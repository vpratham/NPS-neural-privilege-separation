"""Buffered generation gate. No model output escapes before successful EOS."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Iterator, Protocol

from .observer import Observer


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
    """A model-portable observer contract; profiles remain model-specific.

    The host owns adapter, profile, policy and mode. Untrusted requests may
    supply task/context only. This class does not execute tools.
    """

    def __init__(self, adapter: Adapter, profile: dict | None = None, *, mode: str = "enforce",
                 timeout_seconds: float | None = None):
        if mode not in ("enforce", "monitor", "permissions"):
            raise ValueError("Invalid enforcement mode")
        if mode == "permissions":
            from .read_permissions import ReadPermissionAdapter
            if profile is not None or type(adapter) is not ReadPermissionAdapter:
                raise ValueError("Permission mode requires a read-permission adapter and no anomaly profile")
        else:
            Observer(profile)  # Reject malformed profiles before touching the model.
        if timeout_seconds is not None and (isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Invalid generation timeout")
        self.timeout_seconds = timeout_seconds
        self.profile = json.loads(json.dumps(profile, allow_nan=False))
        self.adapter = adapter
        self.mode = mode
        self._permission_binding = (json.loads(json.dumps(adapter.identity)), policy_digest(adapter.policy))
        self._calibrated_horizon = None
        self._check_binding()

    @classmethod
    def from_artifact(cls, adapter: Adapter, artifact: dict, *, mode: str = "enforce"):
        """Deployment constructor: validate provenance and bind fitted horizon."""
        from .artifacts import validate_profile
        checked = validate_profile(artifact)
        instance = cls(adapter, checked["profile"], mode=mode)
        instance._calibrated_horizon = checked["max_new_tokens"]
        return instance

    def _check_binding(self):
        if self.profile is None:
            if self._permission_binding != (self.adapter.identity, policy_digest(self.adapter.policy)):
                raise ValueError("Permission adapter identity changed")
            return
        if self.profile["identity"] != self.adapter.identity:
            raise ValueError("Profile does not match the model/sensor identity; recapture and recalibrate")
        if self.profile["policy_sha256"] != policy_digest(self.adapter.policy):
            raise ValueError("Profile does not match the trusted policy")
        if self.profile["feature_dim"] != self.adapter.identity.get("feature_dim"):
            raise ValueError("Profile feature dimension does not match sensor")

    def run(self, task: str, context: str = "", *, max_new_tokens: int | None = None) -> dict:
        events: list[dict] = []
        tokens: list[int] = []
        stream = None
        result = {"status": "error", "output": None, "mode": self.mode,
                  "enforced": self.mode != "monitor", "reason": "runtime_error", "events": events}
        try:
            deadline = time.monotonic() + self.timeout_seconds if self.timeout_seconds is not None else None
            self._check_binding()
            if max_new_tokens is None:
                max_new_tokens = self._calibrated_horizon or 128
            if not isinstance(task, str) or not task.strip() or not isinstance(context, str):
                raise ValueError("task must be non-empty text and context must be text")
            if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 32768:
                raise ValueError("max_new_tokens must be between 1 and 32768")
            if self._calibrated_horizon is not None and max_new_tokens != self._calibrated_horizon:
                raise ValueError("Generation horizon differs from the calibrated profile")
            observer = Observer(self.profile) if self.profile is not None else None
            stream = iter(self.adapter.iter_steps(task, context, max_new_tokens))
            for index, frame in enumerate(stream):
                if index >= max_new_tokens:
                    raise ValueError("adapter exceeded generation budget")
                if not isinstance(frame, Step) or type(frame.token_id) is not int or frame.token_id < 0 or type(frame.is_eos) is not bool:
                    raise ValueError("invalid adapter frame")
                if deadline is not None and time.monotonic() >= deadline:
                    result.update(reason="generation_timeout")
                    break
                if observer is None:
                    if (not isinstance(frame.features, list)
                            or len(frame.features) != self.adapter.identity["feature_dim"]
                            or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in frame.features)):
                        raise ValueError("Invalid permission-path telemetry")
                    events.append({"alarm": False})
                    if frame.is_eos:
                        output = self.adapter.decode(tokens)
                        if not isinstance(output, str):
                            raise ValueError("adapter decode did not return text")
                        result.update(status="allowed", reason="read_permissions_enforced", output=output)
                        break
                    tokens.append(frame.token_id)
                    continue
                event = observer.step(frame.features)
                events.append(event)
                valid = not event.get("sensor_error", False) and all(
                    isinstance(event.get(key), (int, float)) and math.isfinite(event[key])
                    for key in ("innovation_energy", "max_standardized_residual", "cusum", "threshold"))
                if not valid:
                    result["reason"] = "invalid_telemetry"
                    break
                if event["alarm"] and self.mode == "enforce":
                    result.update(status="blocked", reason="state_anomaly")
                    break
                if frame.is_eos:
                    output = self.adapter.decode(tokens)
                    if not isinstance(output, str):
                        raise ValueError("adapter decode did not return text")
                    result.update(status="allowed" if self.mode == "enforce" else "monitored",
                                  reason="completed", output=output)
                    break
                tokens.append(frame.token_id)
            else:
                result.update(status="incomplete", reason="generation_ended_without_eos")
        except Exception as error:
            # Do not serialize exception messages: providers can include prompt,
            # generated tokens or credentials in them.
            result.update(status="error", output=None, reason="runtime_error", error_type=type(error).__name__)
        finally:
            if stream is not None and callable(getattr(stream, "close", None)):
                try:
                    stream.close()
                except Exception as error:
                    result.update(status="error", output=None, reason="cleanup_error", error_type=type(error).__name__)
        # Invalid numbers are never written as non-standard JSON Infinity/NaN.
        for event in events:
            for key, value in event.items():
                if isinstance(value, float) and not math.isfinite(value):
                    event[key] = None
        result["alarm_observed"] = any(event["alarm"] for event in events)
        result["observed_steps"] = len(events)
        return result
