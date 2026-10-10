"""Train on JSONL decisions and evaluate against the untouched pretrained encoder."""

import argparse
import json
from pathlib import Path
import random
import uuid

import torch

from model import DecisionJEPA, Example, collect_logits, fit_temperature, metrics
from schema import TrainingRecord, state_text


def read_split(path: str) -> tuple[list[Example], set[tuple[str, str]]]:
    examples, identities = [], set()
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = TrainingRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            state = state_text(record.state)
            identities.add(("state", state.strip()))
            if record.group is not None:
                identities.add(("group", record.group.strip()))
            for name, question in record.questions.items():
                value = record.answers[name]
                label = question.options()[0].index(value) if question.type == "choice" else int(value)
                examples.append(Example(state, question, label))
    if not examples:
        raise ValueError(f"{path}: split is empty")
    return examples, identities


def load_splits(paths: dict[str, str]) -> dict[str, list[Example]]:
    splits, seen = {}, {}
    for name, path in paths.items():
        examples, identities = read_split(path)
        for previous, previous_ids in seen.items():
            if identities & previous_ids:
                raise ValueError(f"data leakage between {previous} and {name}: repeated state or group")
        splits[name], seen[name] = examples, identities
    return splits


def report(logits, examples, temperature=1.0):
    result = metrics(logits, examples, temperature)
    result["by_type"] = {}
    for kind in ("noul", "choice", "score"):
        indices = [i for i, example in enumerate(examples) if example.question.type == kind]
        if indices:
            result["by_type"][kind] = metrics(
                [logits[i] for i in indices], [examples[i] for i in indices], temperature,
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "validation", "calibration", "test"):
        parser.add_argument(f"--{split}", required=True, help="Labeled JSONL, disjoint states and groups")
    parser.add_argument("--output", required=True, help="New checkpoint directory")
    parser.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or not 1 <= args.batch_size <= 64:
        parser.error("epochs must be positive and batch-size must be between 1 and 64")
    output = Path(args.output)
    if output.exists():
        parser.error("output already exists; choose a new directory")
    splits = load_splits({name: getattr(args, name) for name in ("train", "validation", "calibration", "test")})
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = DecisionJEPA(args.encoder, args.device)
    # Validate every sequence before producing any checkpoint files.
    for examples in splits.values():
        collect_logits(model, examples, args.batch_size)
    output.mkdir(parents=True)
    model.target_encoder.save_pretrained(str(output / "target"))
    optimizer = torch.optim.AdamW([
        {"params": model.context_encoder.parameters(), "lr": 2e-5},
        {"params": model.predictor.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    total_steps = args.epochs * ((len(splits["train"]) + args.batch_size - 1) // args.batch_size)
    warmup_steps = max(1, int(total_steps * 0.1))

    def learning_rate(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    best_nll, stale, best_epoch = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        examples = list(splits["train"])
        random.shuffle(examples)
        epoch_loss = 0.0
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss; no usable checkpoint was finalized")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach()) * len(batch)
        validation_logits = collect_logits(model, splits["validation"], args.batch_size)
        validation = report(validation_logits, splits["validation"])
        print(json.dumps({"epoch": epoch, "loss": epoch_loss / len(examples), "validation": validation}), flush=True)
        if validation["nll"] < best_nll:
            best_nll, stale, best_epoch = validation["nll"], 0, epoch
            model.save_weights(output)
        else:
            stale += 1
            if stale >= 3:
                break
    model.restore_weights(output)
    calibration_logits = collect_logits(model, splits["calibration"], args.batch_size)
    model.temperature = fit_temperature(calibration_logits, splits["calibration"])
    test_logits = collect_logits(model, splits["test"], args.batch_size)
    baseline_calibration = collect_logits(model, splits["calibration"], args.batch_size, baseline=True)
    baseline_temperature = fit_temperature(baseline_calibration, splits["calibration"])
    baseline_test = collect_logits(model, splits["test"], args.batch_size, baseline=True)
    model.model_id = f"jepa-decision-{uuid.uuid4().hex[:8]}"
    results = {
        "validation": report(collect_logits(model, splits["validation"], args.batch_size), splits["validation"]),
        "test_uncalibrated": report(test_logits, splits["test"]),
        "test": report(test_logits, splits["test"], model.temperature),
        "pretrained_baseline_test": report(baseline_test, splits["test"], baseline_temperature),
    }
    metadata = {
        "format": 1, "model_id": model.model_id, "encoder": args.encoder,
        "temperature": model.temperature, "baseline_temperature": baseline_temperature,
        "best_epoch": best_epoch, "seed": args.seed,
        "split_counts": {name: len(examples) for name, examples in splits.items()},
        "metrics": results,
    }
    # Written last: an interrupted run cannot be mistaken for a finished checkpoint.
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metadata, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
