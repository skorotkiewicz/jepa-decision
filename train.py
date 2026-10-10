"""Train on JSONL decisions and evaluate against the untouched pretrained encoder."""

import argparse
import json
from pathlib import Path
import random
import sys
import uuid

import jax
import jax.numpy as jnp
import numpy as np
import optax

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
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
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
    demo = all(
        Path(getattr(args, name)).resolve() == Path(__file__).resolve().parent / "data" / f"{name}.jsonl"
        for name in splits
    )
    if demo:
        print("Training bundled synthetic demo data. These metrics do not measure real-world accuracy.", file=sys.stderr)
    random.seed(args.seed)
    model = DecisionJEPA(args.encoder, args.device, seed=args.seed)
    rng = jax.device_put(jax.random.PRNGKey(args.seed), model.device)
    # Check lengths without evaluating test labels or creating checkpoint files.
    for examples in splits.values():
        for example in examples:
            model.tokenize([example.context(), *example.targets()])
    output.mkdir(parents=True)
    model.save_encoder(output)
    total_steps = args.epochs * ((len(splits["train"]) + args.batch_size - 1) // args.batch_size)
    warmup_steps = max(1, int(total_steps * 0.1))

    def learning_rate(step):
        return jnp.where(
            step < warmup_steps, (step + 1) / warmup_steps,
            jnp.maximum(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)),
        )

    parameter_groups = {
        "context": jax.tree.map(lambda _: "context", model.params["context"]),
        "predictor": jax.tree.map(lambda _: "predictor", model.params["predictor"]),
    }
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform({
            "context": optax.adamw(lambda step: 2e-5 * learning_rate(step), weight_decay=0.01),
            "predictor": optax.adamw(lambda step: 1e-3 * learning_rate(step), weight_decay=0.01),
        }, parameter_groups),
    )
    optimizer_state = optimizer.init(model.params)

    @jax.jit
    def train_step(params, optimizer_state, batch, rng):
        loss, gradients = jax.value_and_grad(model.loss)(params, batch, rng)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        return optax.apply_updates(params, updates), optimizer_state, loss, optax.global_norm(gradients)

    best_nll, stale, best_epoch = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        examples = list(splits["train"])
        random.shuffle(examples)
        epoch_loss = 0.0
        for start in range(0, len(examples), args.batch_size):
            examples_batch = examples[start:start + args.batch_size]
            batch, _ = model.prepare(examples_batch)
            rng, dropout_rng = jax.random.split(rng)
            params, next_state, loss, gradient_norm = train_step(model.params, optimizer_state, batch, dropout_rng)
            if not np.isfinite(float(loss)) or not np.isfinite(float(gradient_norm)):
                raise ValueError("non-finite loss or gradient; no usable checkpoint was finalized")
            model.params, optimizer_state = params, next_state
            epoch_loss += float(loss) * len(examples_batch)
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
    model.model_id = f"jepa-decision-{'demo-' if demo else ''}{uuid.uuid4().hex[:8]}"
    results = {
        "validation": report(collect_logits(model, splits["validation"], args.batch_size), splits["validation"]),
        "test_uncalibrated": report(test_logits, splits["test"]),
        "test": report(test_logits, splits["test"], model.temperature),
        "pretrained_baseline_test": report(baseline_test, splits["test"], baseline_temperature),
    }
    metadata = {
        "format": 2, "backend": "jax", "demo": demo, "model_id": model.model_id, "encoder": args.encoder,
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
