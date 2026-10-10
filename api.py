"""Serve trained JEPA decisions without generating text."""

from contextlib import asynccontextmanager
import os
from threading import Lock
import uuid

from fastapi import FastAPI, HTTPException, Request
import jax
import numpy as np

from model import DecisionJEPA, Example
from schema import DecisionRequest, state_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint = os.environ.get("JEPA_CHECKPOINT")
    if not checkpoint:
        raise RuntimeError("set JEPA_CHECKPOINT to a completed training directory")
    app.state.model = DecisionJEPA.load(checkpoint, os.environ.get("JEPA_DEVICE", "cpu"))
    app.state.lock = Lock()
    yield


app = FastAPI(title="JEPA decisions", lifespan=lifespan)


@app.post("/v1/decisions")
def decisions(payload: DecisionRequest, request: Request):
    model = request.app.state.model
    if payload.model not in {"jepa-decision", "vlang-research", model.model_id}:
        raise HTTPException(404, "unknown model")
    examples = [Example(state_text(payload.state), question) for question in payload.questions.values()]
    # ponytail: serial inference, use a batch queue if measured throughput requires it.
    with request.app.state.lock:
        try:
            logits, _, _, input_tokens = model(examples)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        answers = {
            name: question.answer(np.asarray(jax.nn.softmax(scores / model.temperature)).tolist())
            for (name, question), scores in zip(payload.questions.items(), logits)
        }
    return {
        "model": model.model_id, "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0, "cost": 0.0},
        "id": f"gen-dec-{uuid.uuid4()}", "provider": "JEPA-local",
    }
