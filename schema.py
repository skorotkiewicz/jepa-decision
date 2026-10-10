"""The decision contract from example.md, shared by training and serving."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictInt, StrictStr, model_validator


def state_text(state: JsonValue) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["noul", "choice", "score"]
    instructions: str = Field(min_length=1, max_length=4096)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def check_criteria(self):
        if not self.instructions.strip():
            raise ValueError("instructions cannot be blank")
        if self.type == "noul":
            if self.criteria is not None and (
                not isinstance(self.criteria, dict) or set(self.criteria) != {"false", "true"}
            ):
                raise ValueError("noul criteria must contain exactly false and true")
        elif self.type == "choice":
            if not isinstance(self.criteria, dict) or not 1 <= len(self.criteria) <= 255:
                raise ValueError("choice requires 1 to 255 named options")
        elif not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 255:
            raise ValueError("score requires 2 to 255 ordered descriptions")
        if self.criteria is not None:
            values = list(self.criteria.values()) if isinstance(self.criteria, dict) else self.criteria
            if any(not v.strip() or len(v) > 4096 for v in values):
                raise ValueError("option descriptions must contain 1 to 4096 characters")
            if len({v.strip() for v in values}) != len(values):
                raise ValueError("option descriptions must be distinct")
            if isinstance(self.criteria, dict) and any(
                not key.strip() or len(key) > 255 for key in self.criteria
            ):
                raise ValueError("option keys must contain 1 to 255 characters")
        return self

    def options(self) -> tuple[list[str], list[str]]:
        if self.type == "noul":
            criteria = self.criteria or {"false": "No. The answer is false.", "true": "Yes. The answer is true."}
            return ["false", "true"], [criteria["false"], criteria["true"]]
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))], list(self.criteria)
        return list(self.criteria), list(self.criteria.values())

    def answer(self, probabilities: list[float]) -> dict:
        keys, descriptions = self.options()
        if len(probabilities) != len(keys):
            raise ValueError("probability count does not match options")
        if self.type == "noul":
            return {"type": "noul", "noul": probabilities[1]}
        answer = {
            "type": self.type,
            "probabilities": dict(zip(keys, probabilities)),
            "confidence": max(probabilities),
        }
        if self.type == "choice":
            answer["choice"] = keys[max(range(len(keys)), key=probabilities.__getitem__)]
        else:
            answer["score"] = sum(i * p for i, p in enumerate(probabilities))
            answer["legend"] = dict(zip(keys, descriptions))
        return answer


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(default="jepa-decision", min_length=1, max_length=255)
    state: JsonValue
    questions: dict[str, Question] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def check_size(self):
        try:
            text = state_text(self.state)
            encoded = json.dumps(self.model_dump(), ensure_ascii=False, allow_nan=False)
        except (ValueError, RecursionError) as error:
            raise ValueError("state must be finite JSON") from error
        if not text.strip() or len(text) > 64_000:
            raise ValueError("state must contain 1 to 64000 characters")
        if len(encoded.encode("utf-8")) > 262_144:
            raise ValueError("request exceeds 256 KiB")
        if any(not key.strip() or len(key) > 255 for key in self.questions):
            raise ValueError("question keys must contain 1 to 255 characters")
        return self


class TrainingRecord(DecisionRequest):
    answers: dict[str, StrictBool | StrictInt | StrictStr]
    group: str | None = None

    @model_validator(mode="after")
    def check_answers(self):
        if set(self.answers) != set(self.questions):
            raise ValueError("answers must match question keys exactly")
        for name, question in self.questions.items():
            value = self.answers[name]
            keys, _ = question.options()
            if question.type == "noul":
                valid = type(value) is bool
            elif question.type == "score":
                valid = type(value) is int and 0 <= value < len(keys)
            else:
                valid = type(value) is str and value in keys
            if not valid:
                raise ValueError(f"invalid answer for {name}")
        if self.group is not None and not self.group.strip():
            raise ValueError("group cannot be blank")
        return self
