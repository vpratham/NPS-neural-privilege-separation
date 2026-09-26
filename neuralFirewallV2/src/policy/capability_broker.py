"""Small reference capability broker for CPU-only security experiments.

This module is deliberately a mock-effect implementation. It is useful for
testing authorization contracts; it is not an isolated production service and
does not provide production key custody, durable replay protection, or a real
tool sandbox.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

MAX_WIRE_BYTES = 4096
MAX_VALUE_CHARS = 240
TOOL_NAME = "write_record"
ALLOWED_RESOURCES = frozenset({"notes"})
KNOWN_TOOLS = {
    "write_record": {"resource", "value"},
    "send_mock_message": {"destination", "content"},
}


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically and reject NaN/Infinity."""
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("nonfinite_json")


def parse_native_request(raw: str) -> dict[str, Any]:
    """Parse one strict Qwen-style tool call; never recover from malformed text."""
    if not isinstance(raw, str):
        raise ValueError("wire_schema")
    try:
        encoded = raw.encode("utf-8", "strict")
    except UnicodeError as exc:
        raise ValueError("invalid_unicode") from exc
    if len(encoded) > MAX_WIRE_BYTES:
        raise ValueError("wire_too_large")

    opening, closing, eos = "<tool_call>", "</tool_call>", "<|im_end|>"
    if raw.count(opening) != 1 or raw.count(closing) != 1:
        raise ValueError("exactly_one_tool_call_required")
    start = raw.index(opening)
    end = raw.index(closing)
    prefix = raw[:start]
    suffix = raw[end + len(closing) :]
    if prefix.strip() or suffix not in ("", eos):
        raise ValueError("extra_text_outside_tool_call")

    try:
        document = json.loads(
            raw[start + len(opening) : end].strip(),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc) in {
            "duplicate_json_key",
            "nonfinite_json",
        }:
            raise
        raise ValueError("malformed_json") from exc

    if (
        not isinstance(document, dict)
        or set(document) != {"name", "arguments"}
        or not isinstance(document["name"], str)
        or not isinstance(document["arguments"], dict)
    ):
        raise ValueError("wire_schema")
    tool = document["name"]
    if tool not in KNOWN_TOOLS:
        raise ValueError("tool_not_allowlisted")
    args = document["arguments"]
    if set(args) != KNOWN_TOOLS[tool]:
        raise ValueError("argument_schema")
    if not all(isinstance(value, str) for value in args.values()):
        raise ValueError("argument_schema")
    if tool == "write_record" and args["resource"] not in {"notes", "protected"}:
        raise ValueError("resource_not_allowlisted")
    if tool == "send_mock_message" and args["destination"] != "outbox":
        raise ValueError("argument_value_rejected")
    content_key = "value" if tool == "write_record" else "content"
    if not args[content_key] or len(args[content_key]) > MAX_VALUE_CHARS:
        raise ValueError("argument_value_rejected")
    return {"tool": tool, "arguments": args}


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrustedRecord:
    """Host-verified source data, separate from model output and task labels.

    A production adapter must construct this only after authenticating the
    upstream source and enforcing its data-integrity contract. The dataclass
    itself is not an authentication mechanism.
    """

    source_id: str
    record_id: str
    value: str


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    side_effect: bool = False


class CapabilityBroker:
    """Default-deny broker with scope-only and trusted-source-bound grants.

    `issue_source_bound` accepts a verified source record, not an evaluation
    task, expected answer, or model-generated claim. This makes that
    information boundary visible and testable in CPU-only experiments.
    """

    def __init__(
        self,
        secret: bytes,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError("secret_must_be_at_least_32_bytes")
        self._secret = secret
        self._clock = clock
        self._used_nonces: set[str] = set()
        self._lock = threading.Lock()

    def issue_scope(
        self, subject: str, *, ttl_seconds: int = 60
    ) -> str:
        """Issue a notes-write grant without binding the content."""
        return self._issue(
            subject,
            mode="scope_only",
            arguments_sha256=None,
            source_id=None,
            ttl_seconds=ttl_seconds,
        )

    def issue_source_bound(
        self,
        subject: str,
        record: TrustedRecord,
        *,
        ttl_seconds: int = 60,
    ) -> str:
        """Issue a grant bound to a value obtained from a verified source."""
        if not isinstance(record, TrustedRecord):
            raise ValueError("trusted_record_required")
        if not record.source_id or not record.record_id:
            raise ValueError("trusted_record_identity_required")
        if not isinstance(record.value, str) or not record.value:
            raise ValueError("trusted_record_value_required")
        arguments = {"resource": "notes", "value": record.value}
        return self._issue(
            subject,
            mode="source_bound",
            arguments_sha256=_sha256(arguments),
            source_id=f"{record.source_id}:{record.record_id}",
            ttl_seconds=ttl_seconds,
        )

    def issue_oracle_bound(
        self,
        subject: str,
        expected_arguments: Mapping[str, str],
        *,
        ttl_seconds: int = 60,
    ) -> str:
        """Test-only upper-bound control that binds to known expected arguments.

        Do not use this API as the proposed real-world authorization source.
        """
        if set(expected_arguments) != {"resource", "value"}:
            raise ValueError("argument_schema")
        if expected_arguments["resource"] not in ALLOWED_RESOURCES:
            raise ValueError("resource_not_allowlisted")
        if not isinstance(expected_arguments["value"], str):
            raise ValueError("argument_schema")
        return self._issue(
            subject,
            mode="oracle_bound",
            arguments_sha256=_sha256(dict(expected_arguments)),
            source_id=None,
            ttl_seconds=ttl_seconds,
        )

    def _issue(
        self,
        subject: str,
        *,
        mode: str,
        arguments_sha256: str | None,
        source_id: str | None,
        ttl_seconds: int,
    ) -> str:
        if not isinstance(subject, str) or not subject:
            raise ValueError("subject_required")
        if not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("positive_ttl_required")
        payload = {
            "subject": subject,
            "tool": TOOL_NAME,
            "resource": "notes",
            "mode": mode,
            "arguments_sha256": arguments_sha256,
            "source_id": source_id,
            "exp": self._clock() + ttl_seconds,
            "nonce": secrets.token_hex(16),
        }
        signature = hmac.new(
            self._secret, canonical_json(payload).encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return canonical_json({"payload": payload, "sig": signature})

    def authorize_and_execute(
        self,
        subject: str,
        raw_request: str,
        capability: str | None,
        workspace: dict[str, str],
    ) -> Decision:
        """Validate a proposal, consume a valid one-use grant, and mock-write."""
        try:
            request = parse_native_request(raw_request)
        except ValueError as exc:
            return Decision(False, str(exc))
        if not isinstance(workspace, dict):
            return Decision(False, "workspace_unavailable")
        try:
            token = json.loads(
                capability or "",
                object_pairs_hook=_strict_object,
                parse_constant=_reject_constant,
            )
            payload = token["payload"]
            signature = token["sig"]
            if set(token) != {"payload", "sig"} or not isinstance(payload, dict):
                raise ValueError("token_schema")
            expected = hmac.new(
                self._secret,
                canonical_json(payload).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not isinstance(signature, str) or not hmac.compare_digest(expected, signature):
                return Decision(False, "invalid_signature")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return Decision(False, "missing_or_untrusted_token")

        required = {
            "subject",
            "tool",
            "resource",
            "mode",
            "arguments_sha256",
            "source_id",
            "exp",
            "nonce",
        }
        if set(payload) != required:
            return Decision(False, "capability_schema")
        if payload["exp"] <= self._clock():
            return Decision(False, "expired")
        if payload["subject"] != subject:
            return Decision(False, "subject_mismatch")
        args = request["arguments"]
        if request["tool"] != TOOL_NAME:
            return Decision(False, "scope_mismatch")
        if payload["tool"] != request["tool"] or payload["resource"] != args["resource"]:
            return Decision(False, "scope_mismatch")
        if args["resource"] not in ALLOWED_RESOURCES:
            return Decision(False, "scope_mismatch")
        mode = payload["mode"]
        if mode not in {"scope_only", "source_bound", "oracle_bound"}:
            return Decision(False, "capability_mode_rejected")
        nonce = payload["nonce"]
        if not isinstance(nonce, str) or not nonce:
            return Decision(False, "capability_schema")

        # Atomically spend a structurally valid, correctly scoped grant before
        # checking the content binding. A rejected content guess must not leave
        # the same bearer token available for repeated probing.
        with self._lock:
            if nonce in self._used_nonces:
                return Decision(False, "replay")
            self._used_nonces.add(nonce)
            if mode != "scope_only" and payload["arguments_sha256"] != _sha256(args):
                return Decision(False, "argument_binding_mismatch")
            workspace[args["resource"]] = args["value"]
        return Decision(True, "authorized", True)
