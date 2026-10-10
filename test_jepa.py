"""Offline backend checks. Tiny random weights test mechanics, not accuracy."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from transformers import BertConfig, BertTokenizerFast, FlaxBertModel

from api import app
from model import DecisionJEPA, Example, fit_temperature, metrics
from schema import Question
from train import load_splits


class JepaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="jepa-test-"))
        cls.encoder = cls.root / "encoder"
        cls.encoder.mkdir()
        vocabulary = (
            "[PAD] [UNK] [CLS] [SEP] [MASK] question state decision yes no the answer is "
            "true false refund please my order has not been delivered billing payments "
            "technical bugs sales pricing calm frustrated angry one two three four zoë café : ."
        ).split()
        (cls.encoder / "vocab.txt").write_text("\n".join(vocabulary) + "\n")
        BertTokenizerFast(vocab_file=str(cls.encoder / "vocab.txt")).save_pretrained(cls.encoder)
        config = BertConfig(
            vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        )
        FlaxBertModel(config, seed=7, input_shape=(1, 8)).save_pretrained(cls.encoder)
        (cls.encoder / "sentence_bert_config.json").write_text('{"max_seq_length": 32}')
        cls.questions = {
            "refund": Question(type="noul", instructions="Is the order a refund?"),
            "route": Question(type="choice", instructions="Which team?", criteria={
                "billing": "Payments", "technical": "Bugs", "sales": "Pricing",
            }),
            "mood": Question(type="score", instructions="How angry?", criteria=["Calm", "Frustrated", "Angry"]),
        }
        cls.examples = [Example("Please refund my order, Zoë.", question, 1) for question in cls.questions.values()]

    def setUp(self):
        self.model = DecisionJEPA(str(self.encoder))

    def test_gradients_training_and_frozen_teacher(self):
        batch, _ = self.model.prepare(self.examples)
        original_teacher = [np.array(leaf) for leaf in jax.tree.leaves(self.model.target_params)]
        value_and_grad = jax.jit(jax.value_and_grad(self.model.loss))
        before, gradients = value_and_grad(self.model.params, batch)
        self.assertGreater(float(optax.global_norm(gradients["context"])), 0)
        self.assertGreater(float(optax.global_norm(gradients["predictor"])), 0)
        self.assertEqual(set(self.model.params), {"context", "predictor"})
        optimizer = optax.adam(0.01)
        state = optimizer.init(self.model.params)

        @jax.jit
        def step(params, state):
            loss, grads = jax.value_and_grad(self.model.loss)(params, batch)
            updates, state = optimizer.update(grads, state, params)
            return optax.apply_updates(params, updates), state, loss

        for _ in range(20):
            self.model.params, state, loss = step(self.model.params, state)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertLess(float(self.model.loss(self.model.params, batch)), float(before))
        for original, current in zip(original_teacher, jax.tree.leaves(self.model.target_params)):
            np.testing.assert_array_equal(original, current)

    def test_ragged_options_and_permutation(self):
        single = Question(type="choice", instructions="Which team?", criteria={"billing": "Payments"})
        logits, _, _, _ = self.model([*self.examples, Example("refund", single)])
        self.assertEqual([len(row) for row in logits], [2, 3, 3, 1])
        for scores in logits:
            self.assertAlmostEqual(float(jax.nn.softmax(scores).sum()), 1.0, places=6)
        question = self.questions["route"]
        reversed_question = question.model_copy(update={"criteria": dict(reversed(list(question.criteria.items())))})
        rows, _, _, _ = self.model([Example("refund", question), Example("refund", reversed_question)])
        np.testing.assert_allclose(rows[0], np.asarray(rows[1])[::-1], atol=2e-5)

    def test_calibration(self):
        logits = [jnp.array([8.0, -8.0]), jnp.array([8.0, -8.0])]
        examples = [Example("one", self.questions["refund"], 0), Example("two", self.questions["refund"], 1)]
        temperature = fit_temperature(logits, examples)
        self.assertTrue(0.05 <= temperature <= 20.0)
        self.assertLessEqual(metrics(logits, examples, temperature)["nll"], metrics(logits, examples)["nll"])

    def test_checkpoint_roundtrip(self):
        checkpoint = self.root / "roundtrip"
        checkpoint.mkdir()
        expected = self.model(self.examples)[0]
        self.model.save_encoder(checkpoint)
        self.model.save_weights(checkpoint)
        (checkpoint / "metadata.json").write_text(json.dumps({
            "format": 2, "backend": "jax", "temperature": 1.25, "model_id": "jepa-test",
        }))
        loaded = DecisionJEPA.load(str(checkpoint))
        for original, restored in zip(expected, loaded(self.examples)[0]):
            np.testing.assert_allclose(original, restored, atol=1e-6)
        self.assertEqual(loaded.temperature, 1.25)
        self.assertEqual(loaded.model_id, "jepa-test")
        (checkpoint / "metadata.json").write_text('{"format": 1}')
        with self.assertRaisesRegex(ValueError, "retrain"):
            DecisionJEPA.load(str(checkpoint))

    def test_api_contract_and_limits(self):
        payload = {
            "model": "vlang-research", "state": "Please refund my order, Zoë.",
            "questions": {name: question.model_dump() for name, question in self.questions.items()},
        }
        with patch.dict(os.environ, {"JEPA_CHECKPOINT": str(self.encoder)}), patch(
            "api.DecisionJEPA.load", return_value=self.model,
        ), TestClient(app) as client:
            response = client.post("/v1/decisions", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result["usage"]["output_tokens"], 0)
            self.assertGreater(result["usage"]["input_tokens"], 0)
            answers = result["answers"]
            self.assertTrue(0 <= answers["refund"]["noul"] <= 1)
            for kind in ("route", "mood"):
                self.assertAlmostEqual(sum(answers[kind]["probabilities"].values()), 1, places=6)
            self.assertIn(answers["route"]["choice"], self.questions["route"].criteria)
            self.assertAlmostEqual(answers["mood"]["score"], sum(
                int(index) * probability for index, probability in answers["mood"]["probabilities"].items()
            ))
            self.assertEqual(client.post("/v1/decisions", json={**payload, "model": "unknown"}).status_code, 404)
            self.assertEqual(client.post("/v1/decisions", json={**payload, "state": "refund " * 64}).status_code, 422)
            self.assertEqual(client.post("/v1/decisions", json={**payload, "questions": {}}).status_code, 422)

    def test_bundled_demo_splits(self):
        data = Path(__file__).resolve().parent / "data"
        splits = load_splits({name: str(data / f"{name}.jsonl") for name in (
            "train", "validation", "calibration", "test",
        )})
        self.assertEqual({name: len(rows) for name, rows in splits.items()}, {
            "train": 56, "validation": 18, "calibration": 18, "test": 18,
        })
        for rows in splits.values():
            self.assertEqual({row.question.type for row in rows}, {"noul", "choice", "score"})

    def test_training_cli_and_split_leakage(self):
        paths = {}
        for split, state in zip(("train", "validation", "calibration", "test"), ("one", "two", "three", "four")):
            path = self.root / f"{split}.jsonl"
            path.write_text(json.dumps({
                "state": f"refund order {state}",
                "questions": {name: q.model_dump() for name, q in self.questions.items()},
                "answers": {"refund": True, "route": "billing", "mood": 1},
            }) + "\n")
            paths[split] = str(path)
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_splits({"train": paths["train"], "validation": paths["train"]})
        output = self.root / "trained"
        repo = Path(__file__).resolve().parent
        env = {**os.environ, "PYTHON": sys.executable, "JAX_PLATFORMS": "cpu", "JEPA_CHECKPOINT": str(output)}
        argv = [str(repo / "train.sh"), "--encoder", str(self.encoder), "--output", str(output), "--epochs", "1", "--batch-size", "3"]
        for split, path in paths.items():
            argv += [f"--{split}", path]
        trained = subprocess.run(argv, cwd=self.root, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(trained.returncode, 0, trained.stdout + trained.stderr)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["backend"], "jax")
        self.assertEqual(metadata["best_epoch"], 1)
        self.assertEqual(metadata["metrics"]["test"]["count"], 3)
        self.assertEqual(set(metadata["metrics"]["test"]["by_type"]), {"noul", "choice", "score"})
        self.assertEqual(len(DecisionJEPA.load(str(output))(self.examples)[0]), 3)

        log_path = self.root / "server.log"
        with socket.socket() as listener, log_path.open("w") as log:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            server = subprocess.Popen(
                [str(repo / "run.sh"), "--fd", str(listener.fileno())],
                cwd=self.root, env=env, pass_fds=(listener.fileno(),), stdout=log, stderr=log,
            )
            try:
                with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=1, trust_env=False) as client:
                    deadline = time.monotonic() + 60
                    while True:
                        try:
                            if client.get("/openapi.json").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        if server.poll() is not None or time.monotonic() >= deadline:
                            self.fail(f"server did not start: {log_path.read_text()}")
                        time.sleep(0.1)
                    response = client.post("/v1/decisions", json={
                        "model": "vlang-research", "state": "Please refund my order, Zoë.",
                        "questions": {name: q.model_dump() for name, q in self.questions.items()},
                    }, timeout=30)
                    self.assertEqual(response.status_code, 200, response.text)
                    result = response.json()
                    self.assertEqual(result["model"], metadata["model_id"])
                    self.assertEqual(set(result["answers"]), set(self.questions))
                    self.assertEqual(result["usage"]["output_tokens"], 0)
            finally:
                server.terminate()
                server.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
