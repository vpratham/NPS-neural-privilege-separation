"""Buffered runtime for deterministic source read permissions."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Iterator, Protocol


def policy_digest(policy: str) -> str:
    return hashlib.sha256(policy.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Step:
    features: list[float]
    token_id: int
    is_eos: bool


class Adapter(Protocol):
    identity: dict
    policy: str

    def iter_steps(self, task: str, context: str, max_new_tokens: int) -> Iterator[Step]: ...
    def decode(self, token_ids: list[int]) -> str: ...


class Firewall:
    """Release only complete output from a host-configured permission adapter."""

    mode = "permissions"

    def __init__(self, adapter: Adapter, *, timeout_seconds: float | None = None):
        from .read_permissions import ReadPermissionAdapter

        if type(adapter) is not ReadPermissionAdapter:
            raise ValueError("Permission firewall requires a read-permission adapter")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.adapter = adapter
        self.timeout_seconds = timeout_seconds
        self._binding = (json.loads(json.dumps(adapter.identity)), policy_digest(adapter.policy))

    def run(self, task: str, context: str = "", *, max_new_tokens: int = 128) -> dict:
        result = {"status": "error", "output": None, "mode": self.mode, "enforced": True,
                  "reason": "runtime_error", "events": []}
        stream = None
        tokens = []
        try:
            if self._binding != (self.adapter.identity, policy_digest(self.adapter.policy)):
                raise ValueError("Permission adapter identity changed")
            if (not isinstance(task, str) or not task.strip() or context != ""
                    or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 32768):
                raise ValueError("Invalid permission request")
            deadline = time.monotonic() + self.timeout_seconds if self.timeout_seconds else None
            stream = iter(self.adapter.iter_steps(task, "", max_new_tokens))
            for index, frame in enumerate(stream):
                if index >= max_new_tokens:
                    raise ValueError("Adapter exceeded generation budget")
                if deadline is not None and time.monotonic() >= deadline:
                    result["reason"] = "generation_timeout"
                    break
                if type(frame.token_id) is not int or frame.token_id < 0 or type(frame.is_eos) is not bool:
                    raise ValueError("Invalid adapter frame")
                if (not isinstance(frame.features, list)
                        or len(frame.features) != self.adapter.identity["feature_dim"]):
                    raise ValueError("Invalid adapter telemetry shape")
                result["events"].append({"alarm": False})
                if frame.is_eos:
                    output = self.adapter.decode(tokens)
                    if not isinstance(output, str):
                        raise ValueError("Adapter decode did not return text")
                    result.update(status="allowed", reason="read_permissions_enforced", output=output)
                    break
                tokens.append(frame.token_id)
            else:
                result.update(status="incomplete", reason="generation_ended_without_eos")
        except Exception as error:
            result.update(error_type=type(error).__name__)
        finally:
            if stream is not None and callable(getattr(stream, "close", None)):
                try:
                    stream.close()
                except Exception as error:
                    result.update(status="error", output=None, reason="cleanup_error",
                                  error_type=type(error).__name__)
        return result
