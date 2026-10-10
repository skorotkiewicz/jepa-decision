"""Supervised text JEPA with JAX, a frozen semantic teacher, and candidate ranking."""

from dataclasses import dataclass
import json
from pathlib import Path

from flax import linen as nn, serialization
from huggingface_hub import file_exists, snapshot_download
import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer, FlaxAutoModel

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


def normalize(values):
    return values / jnp.maximum(jnp.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


class Predictor(nn.Module):
    dimension: int

    @nn.compact
    def __call__(self, context):
        hidden = nn.LayerNorm()(context)
        hidden = nn.gelu(nn.Dense(self.dimension)(hidden))
        residual = nn.Dense(self.dimension, kernel_init=nn.initializers.zeros)(hidden)
        return normalize(context + residual)


def encoder_path(encoder: str) -> Path:
    path = Path(encoder)
    if not path.is_dir():
        # Download only native Flax or safetensors, never a pickle/.bin checkpoint.
        weights = "flax_model.msgpack" if file_exists(encoder, "flax_model.msgpack") else "model.safetensors"
        path = Path(snapshot_download(encoder, allow_patterns=[
            weights, "*.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model",
        ]))
    if not any((path / name).is_file() for name in ("flax_model.msgpack", "model.safetensors")):
        raise ValueError("encoder needs native Flax or unsharded safetensors weights, not PyTorch .bin")
    modules_file = path / "modules.json"
    if modules_file.exists():
        supported = {
            "sentence_transformers.models.Transformer", "sentence_transformers.models.Pooling",
            "sentence_transformers.models.Normalize",
        }
        if any(module["type"] not in supported for module in json.loads(modules_file.read_text())):
            raise ValueError("encoder has unsupported SentenceTransformer modules")
    pooling_file = path / "1_Pooling" / "config.json"
    if pooling_file.exists():
        pooling = json.loads(pooling_file.read_text())
        if not pooling.get("pooling_mode_mean_tokens") or any(
            value for key, value in pooling.items()
            if key.startswith("pooling_mode_") and key != "pooling_mode_mean_tokens"
        ):
            raise ValueError("encoder must use mean-token pooling")
    return path


class DecisionJEPA:
    def __init__(self, encoder: str, device: str = "cpu", seed: int = 42):
        platform, _, index = device.partition(":")
        self.device = jax.devices(platform)[int(index or 0)]
        path = encoder_path(encoder)
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False)
        with jax.default_device(self.device):
            self.encoder = FlaxAutoModel.from_pretrained(
                path, from_pt=not (path / "flax_model.msgpack").exists(),
                dtype=jnp.float32, trust_remote_code=False,
            )
            # Immutable pretrained parameters are the teacher, excluded from optimization.
            # This is supervised latent prediction, not self-supervised I-JEPA.
            self.target_params = jax.device_put(self.encoder.params, self.device)
            dimension = self.encoder.config.hidden_size
            self.predictor = Predictor(dimension)
            self.params = {
                "context": self.target_params,
                "predictor": self.predictor.init(jax.random.PRNGKey(seed), jnp.zeros((1, dimension)))["params"],
            }
        limit_file = path / "sentence_bert_config.json"
        limit = json.loads(limit_file.read_text())["max_seq_length"] if limit_file.exists() else 256
        self.max_seq_length = min(limit, self.encoder.config.max_position_embeddings)
        self.temperature = 1.0
        self.model_id = "jepa-decision"
        self._embed = jax.jit(self.encode)
        self._infer = jax.jit(self.predict)

    def tokenize(self, texts: list[str]):
        # Never silently discard the end of a state, including a negation.
        raw = self.tokenizer(texts, truncation=False, padding=False)
        lengths = [len(ids) for ids in raw["input_ids"]]
        if max(lengths) > self.max_seq_length:
            raise ValueError(f"input exceeds encoder limit of {self.max_seq_length} tokens")
        features = self.tokenizer.pad(raw, padding="max_length", max_length=self.max_seq_length, return_tensors="np")
        return jax.device_put(dict(features), self.device), sum(lengths)

    def encode(self, params, tokens, dropout_rng=None):
        hidden = self.encoder(
            **tokens, params=params, train=dropout_rng is not None, dropout_rng=dropout_rng,
        ).last_hidden_state
        mask = tokens["attention_mask"][..., None]
        pooled = (hidden * mask).sum(axis=1) / jnp.maximum(mask.sum(axis=1), 1)
        return normalize(pooled)

    def prepare(self, examples: list[Example]):
        if not examples:
            raise ValueError("batch cannot be empty")
        contexts, tokens = self.tokenize([example.context() for example in examples])
        target_texts = [text for example in examples for text in example.targets()]
        embeddings = []
        for start in range(0, len(target_texts), 32):
            features, count = self.tokenize(target_texts[start:start + 32])
            embeddings.append(self._embed(self.target_params, features))
            tokens += count
        targets = jax.lax.stop_gradient(jnp.concatenate(embeddings))
        counts = [len(example.question.options()[0]) for example in examples]
        width = max(counts)
        groups, offset = [], 0
        for count in counts:
            groups.append(jnp.pad(targets[offset:offset + count], ((0, width - count), (0, 0))))
            offset += count
        batch = {
            "contexts": contexts, "targets": jnp.stack(groups),
            "mask": np.arange(width)[None, :] < np.asarray(counts)[:, None],
            "labels": np.asarray([example.label for example in examples], dtype=np.int32),
            "ordered": np.asarray([example.question.type == "score" for example in examples]),
        }
        return jax.device_put(batch, self.device), tokens

    def predict(self, params, batch, dropout_rng=None):
        context = self.encode(params["context"], batch["contexts"], dropout_rng)
        predicted = self.predictor.apply({"params": params["predictor"]}, context)
        scores = 20.0 * jnp.einsum("bkd,bd->bk", batch["targets"], predicted)
        return jnp.where(batch["mask"], scores, -jnp.inf), predicted

    def __call__(self, examples: list[Example], baseline: bool = False):
        batch, tokens = self.prepare(examples)
        if baseline:
            predicted = self._embed(self.target_params, batch["contexts"])
            scores = 20.0 * jnp.einsum("bkd,bd->bk", batch["targets"], predicted)
        else:
            scores, predicted = self._infer(self.params, batch)
        counts = [len(example.question.options()[0]) for example in examples]
        return (
            [scores[i, :count] for i, count in enumerate(counts)], predicted,
            [batch["targets"][i, :count] for i, count in enumerate(counts)], tokens,
        )

    def loss(self, params, batch, dropout_rng=None):
        scores, predicted = self.predict(params, batch, dropout_rng)
        ranking = optax.softmax_cross_entropy_with_integer_labels(scores, batch["labels"])
        correct_targets = batch["targets"][jnp.arange(len(predicted)), batch["labels"]]
        latent = 1.0 - (predicted * correct_targets).sum(axis=-1)
        # Ordered bins: penalize mass crossing each ordinal boundary, ignoring padding.
        positions = jnp.arange(scores.shape[-1])[None, :]
        boundaries = positions < (batch["mask"].sum(axis=-1, keepdims=True) - 1)
        desired = positions >= batch["labels"][:, None]
        error = (jax.nn.softmax(scores).cumsum(axis=-1) - desired) ** 2
        ordinal = (error * boundaries).sum(axis=-1) / jnp.maximum(boundaries.sum(axis=-1), 1)
        return (ranking + 0.5 * latent + 0.2 * ordinal * batch["ordered"]).mean()

    def save_encoder(self, directory: Path):
        path = directory / "encoder"
        self.encoder.save_pretrained(path, params=self.target_params)
        self.tokenizer.save_pretrained(path)
        (path / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": self.max_seq_length}))

    def save_weights(self, directory: Path):
        (directory / "parameters.msgpack").write_bytes(serialization.to_bytes(self.params))

    def restore_weights(self, directory: Path):
        self.params = jax.device_put(
            serialization.from_bytes(self.params, (directory / "parameters.msgpack").read_bytes()), self.device,
        )

    @classmethod
    def load(cls, directory: str, device: str = "cpu"):
        path = Path(directory)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata.get("format") != 2 or metadata.get("backend") != "jax":
            raise ValueError("checkpoint is not JAX format 2; retrain old PyTorch checkpoints")
        model = cls(str(path / "encoder"), device)
        model.restore_weights(path)
        model.temperature = float(metadata["temperature"])
        if not 0.05 <= model.temperature <= 20.0:
            raise ValueError("invalid checkpoint temperature")
        model.model_id = metadata["model_id"]
        return model


def collect_logits(model: DecisionJEPA, examples: list[Example], batch_size: int, baseline: bool = False):
    collected = []
    for start in range(0, len(examples), batch_size):
        logits, _, _, _ = model(examples[start:start + batch_size], baseline=baseline)
        collected.extend(np.asarray(scores) for scores in logits)
    return collected


def fit_temperature(logits: list, examples: list[Example]) -> float:
    # ponytail: one global temperature, per-type calibration when held-out NLL warrants it.
    # Include 1.0 so calibration cannot worsen its own set's NLL.
    temperatures = jnp.concatenate([jnp.geomspace(0.05, 20.0, 161), jnp.ones(1)])
    losses = jnp.zeros(len(temperatures))
    for scores, example in zip(logits, examples):
        scaled = jnp.asarray(scores)[None] / temperatures[:, None]
        losses -= jax.nn.log_softmax(scaled)[:, example.label]
    return float(jnp.clip(temperatures[losses.argmin()], 0.05, 20.0))


def metrics(logits: list, examples: list[Example], temperature: float = 1.0) -> dict:
    probabilities = [np.asarray(jax.nn.softmax(jnp.asarray(scores) / temperature)) for scores in logits]
    correct = [int(int(p.argmax()) == e.label) for p, e in zip(probabilities, examples)]
    confidence = [float(p.max()) for p in probabilities]
    nll = [-float(jax.nn.log_softmax(jnp.asarray(scores) / temperature)[e.label]) for scores, e in zip(logits, examples)]
    brier = [float(((p - np.eye(len(p))[e.label]) ** 2).sum()) for p, e in zip(probabilities, examples)]
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
        abs(float(p @ np.arange(len(p))) - e.label)
        for p, e in zip(probabilities, examples) if e.question.type == "score"
    ]
    if score_errors:
        report["score_mae"] = sum(score_errors) / len(score_errors)
    return report
