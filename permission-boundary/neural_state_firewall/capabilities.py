"""Host-assigned, per-instance capability gate for model outputs and actions.

This provider-independent gate separates read, disclose and action grants.
It validates proposals; it never executes an action or treats free prose as a
trusted result. The host/broker remains responsible for authorizing effects.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

from .artifacts import digest, loads

REFUSAL_TEXT = "I can't provide an answer for this request."
MAX_RESPONSE_BYTES = 65_536
MAX_QUOTE_CHARS = 16_384
MAX_TASK_CHARS = 16_000
MAX_SOURCE_CHARS = 80_000
MAX_EVIDENCE_CHARS = 160_000
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


class ModelProvider(Protocol):
    def generate(self, task: str, evidence: Mapping[str, str], *,
                 allowed_actions: Mapping[str, frozenset[str]], max_new_tokens: int) -> str: ...


@dataclass(frozen=True, init=False)
class CapabilityProfile:
    """Immutable grants owned by the host for one model instance."""

    instance_id: str
    readable_sources: frozenset[str]
    disclosable_sources: frozenset[str]
    actions: Mapping[str, frozenset[str]]
    sha256: str

    def __init__(self, instance_id: str, *, readable_sources=(), disclosable_sources=(), actions=None):
        if not isinstance(instance_id, str) or not _IDENTIFIER.fullmatch(instance_id):
            raise ValueError("invalid_instance_id")
        readable = self._names(readable_sources, "readable_sources")
        disclosable = self._names(disclosable_sources, "disclosable_sources")
        if not disclosable <= readable:
            raise ValueError("disclosure_requires_read_permission")
        if actions is None:
            actions = {}
        if not isinstance(actions, Mapping):
            raise ValueError("invalid_actions")
        action_fields = {}
        for name, fields in actions.items():
            if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
                raise ValueError("invalid_action_name")
            action_fields[name] = frozenset(self._names(fields, "action_arguments"))
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "readable_sources", readable)
        object.__setattr__(self, "disclosable_sources", disclosable)
        object.__setattr__(self, "actions", MappingProxyType(action_fields))
        object.__setattr__(self, "sha256", digest({"instance_id": instance_id,
                                                   "readable_sources": sorted(readable),
                                                   "disclosable_sources": sorted(disclosable),
                                                   "actions": {name: sorted(fields) for name, fields in sorted(action_fields.items())}}))

    @staticmethod
    def _names(values, label):
        if isinstance(values, (str, bytes)):
            raise ValueError(f"invalid_{label}")
        try:
            values = tuple(values)
        except TypeError as exc:
            raise ValueError(f"invalid_{label}") from exc
        if any(not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) for value in values):
            raise ValueError(f"invalid_{label}")
        if len(set(values)) != len(values):
            raise ValueError(f"duplicate_{label}")
        return frozenset(values)


class CapabilityBoundary:
    """Enforces one profile around any provider implementing `generate`.

    Accepted outputs are exact quotes from explicitly disclosable evidence,
    fixed refusals, or allowlisted action proposals with exact argument keys.
    Arbitrary prose is blocked. Proposals require a separate trusted broker.
    """

    mode = "capabilities"
    ready = True

    def __init__(self, provider: ModelProvider, profile: CapabilityProfile, *, max_new_tokens: int = 128):
        if not callable(getattr(provider, "generate", None)):
            raise ValueError("provider_must_implement_generate")
        if not isinstance(profile, CapabilityProfile):
            raise ValueError("host_capability_profile_required")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 32768:
            raise ValueError("invalid_generation_limit")
        self.provider = provider
        self.profile = profile
        self.max_new_tokens = max_new_tokens

    def run(self, task: str, evidence: Mapping[str, str], *, max_new_tokens: int | None = None) -> dict:
        limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if (not isinstance(task, str) or not task.strip() or len(task) > MAX_TASK_CHARS
                or not isinstance(evidence, Mapping)
                or any(not isinstance(source, str) or not isinstance(text, str)
                       or len(text) > MAX_SOURCE_CHARS
                       for source, text in evidence.items())
                or not set(evidence) <= self.profile.readable_sources
                or sum(len(text) for text in evidence.values()) > MAX_EVIDENCE_CHARS
                or type(limit) is not int or not 1 <= limit <= self.max_new_tokens):
            return self._blocked("request_exceeds_instance_capabilities")
        try:
            raw = self.provider.generate(task, dict(evidence),
                                         allowed_actions=dict(self.profile.actions),
                                         max_new_tokens=limit)
            if not isinstance(raw, str) or len(raw.encode("utf-8", "strict")) > MAX_RESPONSE_BYTES:
                return self._blocked("invalid_response_size")
            proposal = loads(raw)
            if not isinstance(proposal, dict):
                return self._blocked("invalid_proposal")
            kind = proposal.get("kind")
            if kind == "quote":
                if set(proposal) != {"kind", "source_id", "text"}:
                    return self._blocked("invalid_proposal")
                source, text = proposal["source_id"], proposal["text"]
                if not isinstance(source, str) or source not in self.profile.disclosable_sources:
                    return self._blocked("disclosure_not_allowed")
                if (not isinstance(text, str) or not text or len(text) > MAX_QUOTE_CHARS
                        or text not in evidence.get(source, "")):
                    return self._blocked("quote_not_in_authorized_evidence")
                return self._result("allowed", "verified_disclosable_quote", output=text,
                                    source_id=source)
            if kind == "action":
                if set(proposal) != {"kind", "name", "arguments"}:
                    return self._blocked("invalid_proposal")
                name, arguments = proposal["name"], proposal["arguments"]
                if not isinstance(name, str) or name not in self.profile.actions:
                    return self._blocked("action_not_allowed")
                if not isinstance(arguments, dict) or set(arguments) != self.profile.actions[name]:
                    return self._blocked("action_arguments_not_allowed")
                return self._result("action_proposed", "broker_authorization_required",
                                    action_proposal={"name": name, "arguments": arguments})
            if kind == "refusal" and set(proposal) == {"kind"}:
                return self._result("refused", "model_refused", output=REFUSAL_TEXT)
            return self._blocked("output_type_not_allowed")
        except Exception as error:
            return self._result("error", "provider_or_parse_error", error_type=type(error).__name__)

    def _result(self, status, reason, *, output=None, source_id=None, action_proposal=None, error_type=None):
        result = {"status": status, "mode": self.mode, "enforced": True,
                  "instance_id": self.profile.instance_id, "capability_sha256": self.profile.sha256,
                  "reason": reason, "output": output, "source_id": source_id,
                  "action_proposal": action_proposal}
        if error_type is not None:
            result["error_type"] = error_type
        return result

    def _blocked(self, reason):
        return self._result("blocked", reason)
