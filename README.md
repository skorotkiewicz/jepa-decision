# JEPA decisions with JAX

A supervised text JEPA adaptation of the API in `example.md`. A fine-tuned context encoder predicts the embedding of the correct decision description. A frozen pretrained teacher encodes candidate descriptions. Training combines candidate cross-entropy, latent alignment, and an ordinal loss for scores.

This is not self-supervised I-JEPA, and switching frameworks does not improve accuracy by itself. No task dataset or trained checkpoint is included. The offline tests verify computation and the API, not semantic accuracy.

## Setup

Use Python 3.12 and a virtual environment:

```sh
uv venv --python 3.12
uv pip install -r requirements.txt
```

The default encoder is `sentence-transformers/all-MiniLM-L6-v2`. Transformers converts its safetensors directly to Flax arrays without installing PyTorch. The previous MPNet default has no Flax implementation in the pinned Transformers release. Supported encoders need a Flax implementation, native Flax or unsharded safetensors weights, and mean-token pooling. Additional SentenceTransformer modules are rejected rather than silently skipped.

The pinned Transformers 4.x release supplies the Flax encoder. Do not upgrade to Transformers 5 without checking its JAX support.

## Training data

Each JSONL row is an API request plus an `answers` object. Boolean answers are JSON booleans, choice answers are option keys, and score answers are zero-based integer bins:

```json
{"state":"Please refund my order.","questions":{"refund":{"type":"noul","instructions":"Does the customer request a refund?"},"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments and refunds","technical":"Bugs and integrations"}},"mood":{"type":"score","instructions":"How frustrated?","criteria":["Calm","Frustrated","Very angry"]}},"answers":{"refund":true,"team":"billing","mood":1},"group":"conversation-123"}
```

Provide four nonempty, disjoint files. Repeated states or explicit `group` values across splits are rejected. Use groups for conversations, customers, or source documents so paraphrases do not leak. Include negation, Unicode, hard alternatives, unseen question templates, and representative option counts in the held-out data.

```sh
.venv/bin/python train.py \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --calibration data/calibration.jsonl \
  --test data/test.jsonl \
  --output checkpoints/run-1
```

The context encoder and predictor have separate AdamW learning rates with warmup, decay, and gradient clipping. JAX `jit` compiles the training step, and `value_and_grad` computes derivatives. Teacher parameters never enter the optimizer. Tokenization and frozen-target preparation happen outside that step. Variable batch sizes and option counts can trigger additional compilations.

Validation NLL selects the checkpoint. A separate calibration split fits a bounded global temperature. Test metrics include accuracy, NLL, Brier score, ECE, score MAE, and per-type results. The report also evaluates the unchanged pretrained encoder with its own calibration temperature. Do not claim an improvement unless independent test results support it. Calibration is not a guarantee under distribution shift.

CPU is the default. `--device gpu` or `--device gpu:0` requires a compatible accelerator and a separately installed JAX CUDA stack. GPU execution is not covered by the CPU checks.

## Serving

```sh
JEPA_CHECKPOINT=checkpoints/run-1 .venv/bin/uvicorn api:app --port 8001
```

`POST /v1/decisions` keeps the `example.md` request and answer shapes, including the `vlang-research` model alias. The response identifies the actual local checkpoint. Confidence is the maximum calibrated option probability. A score is the expected zero-based bin index, not an argmax. Long inputs receive HTTP 422 rather than silent truncation.

There is no text generation, so `output_tokens` is zero. Input tokens count the encoded contexts and candidate descriptions. `cost` is zero API billing, not a claim of free computation. Startup fails if `JEPA_CHECKPOINT` is missing or incomplete.

JAX checkpoints use format 2, native Flax teacher weights, and a msgpack parameter tree. The earlier PyTorch format is incompatible and must be retrained. Load only trusted checkpoints.

## Checks

```sh
uv pip install httpx
.venv/bin/python -m unittest -v test_jepa
```

Tests build a tiny local Flax BERT, with no model download. They check gradients, teacher immutability, loss reduction, variable options, option permutation, calibration, checkpoints, training, and HTTP responses.

## Research basis

- [I-JEPA, Assran et al., 2023](https://arxiv.org/abs/2301.08243): predict semantic representations rather than reconstructing inputs. Its image results do not establish text decision accuracy. This adaptation uses labeled outcomes and a frozen teacher instead of masked image blocks and an EMA teacher.
- [Sentence-BERT, Reimers and Gurevych, 2019](https://arxiv.org/abs/1908.10084): pretrained sentence embeddings support semantic candidate matching. Similarity alone does not establish entailment or handle every negation.
- [Guo et al., 2017](https://arxiv.org/abs/1706.04599): temperature scaling adjusts confidence separately from class ranking.
- [JAX training guide](https://docs.jax.dev/en/latest/the-training-cookbook.html): functional parameter trees, compiled steps, and automatic differentiation.
