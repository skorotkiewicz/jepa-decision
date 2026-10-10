"""Supervised text JEPA: predict a frozen semantic target, then rank candidates."""

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

from schema import Question


@dataclass
class Example:
    state: str
    question: Question
    label: int = 0

    def context(self) -> str:
        return f"Question: {self.question.instructions}\nState: {self.state}"

    def targets(self) -> list[str]:
        _, descriptions = self.question.options()
        return [f"Question: {self.question.instructions}\nDecision: {text}" for text in descriptions]


class DecisionJEPA(nn.Module):
    def __init__(self, encoder: str, device: str = "cpu", target: str | None = None):
        super().__init__()
        self.context_encoder = SentenceTransformer(encoder, device=device, trust_remote_code=False)
        # A frozen pretrained teacher keeps the semantic targets stationary.
        # This is supervised latent prediction, not self-supervised I-JEPA.
        self.target_encoder = (
            SentenceTransformer(target, device=device, trust_remote_code=False)
            if target else deepcopy(self.context_encoder)
        )
        self.target_encoder.requires_grad_(False)
        dimension = self.context_encoder.get_sentence_embedding_dimension()
        self.predictor = nn.Sequential(
            nn.LayerNorm(dimension), nn.Linear(dimension, dimension),
            nn.GELU(), nn.Linear(dimension, dimension),
        ).to(device)
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        self.temperature = 1.0
        self.model_id = "jepa-decision"
        self.train(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def embed(self, encoder, texts: list[str]) -> tuple[torch.Tensor, int]:
        # Never silently discard the end of a state, including a negation.
        tokenized = encoder.tokenizer(texts, truncation=False, padding=False)
        lengths = [len(ids) for ids in tokenized["input_ids"]]
        if max(lengths) > encoder.max_seq_length:
            raise ValueError(f"input exceeds encoder limit of {encoder.max_seq_length} tokens")
        embeddings = []
        for start in range(0, len(texts), 32):
            features = batch_to_device(encoder.tokenize(texts[start:start + 32]), encoder.device)
            embeddings.append(encoder(features)["sentence_embedding"])
        return F.normalize(torch.cat(embeddings), dim=-1), sum(lengths)

    def forward(self, examples: list[Example], baseline: bool = False):
        context_encoder = self.target_encoder if baseline else self.context_encoder
        context, tokens = self.embed(context_encoder, [example.context() for example in examples])
        predicted = context if baseline else F.normalize(context + self.predictor(context), dim=-1)
        target_texts = [text for example in examples for text in example.targets()]
        with torch.no_grad():
            targets, target_tokens = self.embed(self.target_encoder, target_texts)
        logits, groups = [], []
        offset = 0
        for index, example in enumerate(examples):
            count = len(example.question.options()[0])
            group = targets[offset:offset + count]
            groups.append(group)
            logits.append(20.0 * (group @ predicted[index]))
            offset += count
        return logits, predicted, groups, tokens + target_tokens

    def loss(self, examples: list[Example]) -> torch.Tensor:
        logits, predictions, targets, _ = self(examples)
        losses = []
        for example, scores, prediction, group in zip(examples, logits, predictions, targets):
            label = torch.tensor([example.label], device=scores.device)
            ranking = F.cross_entropy(scores[None], label)
            latent = 1.0 - (prediction * group[example.label]).sum()
            loss = ranking + 0.5 * latent
            if example.question.type == "score":
                # Ordered bins: penalize probability mass crossing each ordinal boundary.
                cdf = scores.softmax(-1).cumsum(-1)[:-1]
                desired = (torch.arange(len(cdf), device=scores.device) >= example.label).float()
                loss = loss + 0.2 * F.mse_loss(cdf, desired)
            losses.append(loss)
        return torch.stack(losses).mean()

    def save_weights(self, directory: Path):
        self.context_encoder.save_pretrained(str(directory / "context"))
        torch.save(self.predictor.state_dict(), directory / "predictor.pt")

    def restore_weights(self, directory: Path):
        device = str(self.context_encoder.device)
        self.context_encoder = SentenceTransformer(str(directory / "context"), device=device, trust_remote_code=False)
        self.predictor.load_state_dict(torch.load(directory / "predictor.pt", map_location=device, weights_only=True))
        self.train(False)

    @classmethod
    def load(cls, directory: str, device: str = "cpu"):
        path = Path(directory)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata.get("format") != 1:
            raise ValueError("unsupported checkpoint format")
        model = cls(str(path / "context"), device, target=str(path / "target"))
        model.predictor.load_state_dict(torch.load(path / "predictor.pt", map_location=device, weights_only=True))
        model.temperature = float(metadata["temperature"])
        if not 0.05 <= model.temperature <= 20.0:
            raise ValueError("invalid checkpoint temperature")
        model.model_id = metadata["model_id"]
        return model.eval()


@torch.inference_mode()
def collect_logits(model: DecisionJEPA, examples: list[Example], batch_size: int, baseline: bool = False):
    model.eval()
    collected = []
    for start in range(0, len(examples), batch_size):
        logits, _, _, _ = model(examples[start:start + batch_size], baseline=baseline)
        collected.extend(scores.cpu() for scores in logits)
    return collected


def fit_temperature(logits: list[torch.Tensor], examples: list[Example]) -> float:
    # ponytail: one global temperature, per-type calibration when held-out NLL warrants it.
    # A bounded grid includes 1.0, so calibration cannot worsen its own set's NLL.
    temperatures = torch.cat([torch.logspace(-1.30103, 1.30103, 161), torch.ones(1)])
    losses = torch.zeros(len(temperatures))
    for scores, example in zip(logits, examples):
        scaled = scores[None] / temperatures[:, None]
        losses += -scaled.log_softmax(-1)[:, example.label]
    return float(temperatures[losses.argmin()].clamp(0.05, 20.0))


def metrics(logits: list[torch.Tensor], examples: list[Example], temperature: float = 1.0) -> dict:
    probabilities = [(scores / temperature).softmax(-1) for scores in logits]
    correct = [int(int(p.argmax()) == e.label) for p, e in zip(probabilities, examples)]
    confidence = [float(p.max()) for p in probabilities]
    nll = [-float((scores / temperature).log_softmax(-1)[e.label]) for scores, e in zip(logits, examples)]
    brier = [float((p - F.one_hot(torch.tensor(e.label), len(p))).square().sum()) for p, e in zip(probabilities, examples)]
    ece = 0.0
    for index in range(15):
        members = [i for i, c in enumerate(confidence) if int(min(c * 15, 14)) == index]
        if members:
            ece += abs(sum(correct[i] - confidence[i] for i in members)) / len(examples)
    report = {
        "count": len(examples), "accuracy": sum(correct) / len(examples),
        "nll": sum(nll) / len(examples), "brier": sum(brier) / len(examples), "ece": ece,
    }
    score_errors = [
        abs(float(p @ torch.arange(len(p), dtype=p.dtype)) - e.label)
        for p, e in zip(probabilities, examples) if e.question.type == "score"
    ]
    if score_errors:
        report["score_mae"] = sum(score_errors) / len(score_errors)
    return report
