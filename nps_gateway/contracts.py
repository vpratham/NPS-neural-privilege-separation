"""Provider-independent, deliberately small action protocol."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

MAX_PROPOSAL_BYTES = 32_768
MAX_CONTENT_CHARS = 8_000


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "strict")).hexdigest()


def strict_json(raw: str, *, limit: int = MAX_PROPOSAL_BYTES) -> Any:
    if not isinstance(raw, str):
        raise ValueError("invalid_json_type")
    try:
        if len(raw.encode("utf-8", "strict")) > limit:
            raise ValueError("input_too_large")

        def object_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate_json_key")
                result[key] = value
            return result

        def constant(_):
            raise ValueError("nonfinite_json")

        return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("malformed_json") from exc


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", value):
        raise ValueError("invalid_identifier")
    return value


def content(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_CONTENT_CHARS:
        raise ValueError("invalid_content")
    try:
        value.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise ValueError("invalid_content") from exc
    return value


def parse_proposal(raw: str) -> dict:
    proposal = strict_json(raw)
    if not isinstance(proposal, dict) or set(proposal) != {"name", "arguments"}:
        raise ValueError("proposal_schema")
    if proposal["name"] != "write_note":
        raise ValueError("tool_not_allowed")
    args = proposal["arguments"]
    if not isinstance(args, dict) or set(args) != {"note_id", "content"}:
        raise ValueError("argument_schema")
    identifier(args["note_id"])
    content(args["content"])
    return proposal


class ModelAdapter(Protocol):
    """A model only returns an untrusted proposal. It receives no capability."""

    def generate(self, messages: list[dict], tools: list[dict]) -> str: ...


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    run_id: str
    side_effect: bool = False

    @property
    def allowed(self) -> bool:
        return self.status == "executed"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_verified_record",
            "description": "Read the record selected by the application. Already completed for this task.",
            "parameters": {
                "type": "object",
                "properties": {"source_id": {"type": "string"}, "record_id": {"type": "string"}},
                "required": ["source_id", "record_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Copy the verified record value exactly to the user-selected note.",
            "parameters": {
                "type": "object",
                "properties": {"note_id": {"type": "string"}, "content": {"type": "string"}},
                "required": ["note_id", "content"],
                "additionalProperties": False,
            },
        },
    },
]
