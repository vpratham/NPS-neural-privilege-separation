"""Deterministic gate for model-proposed outputs and actions.

The provider is untrusted with respect to authority. It can quote supplied
evidence or propose a host-allowlisted action; it cannot execute that action.
The application must send proposals to its trusted action broker.
"""
from __future__ import annotations

import json
from typing import Mapping, Protocol


REFUSAL_TEXT = "I can't provide an answer for this request."
MAX_RESPONSE_BYTES = 65_536
MAX_QUOTE_CHARS = 16_384


class ModelProvider(Protocol):
    def generate(self, task: str, evidence: Mapping[str, str], *,
                 allowed_actions: frozenset[str], max_new_tokens: int) -> str: ...


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _loads(raw: str) -> dict:
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("invalid response size")
    value = json.loads(raw, object_pairs_hook=_unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid JSON number")))
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    return value


class CapabilityFirewall:
    """Model-agnostic output gate; all capabilities come from host config.

    `quote` responses are released only when their text is an exact substring
    of host-supplied evidence. `action` responses are proposals for a separate
    broker, never executed here. Arbitrary model prose is rejected.
    """

    mode = "capability"
    ready = True

    def __init__(self, provider: ModelProvider, *, allowed_actions=(), max_new_tokens: int = 128):
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 32768:
            raise ValueError("max_new_tokens must be between 1 and 32768")
        actions = tuple(allowed_actions)
        if any(not isinstance(name, str) or not name.strip() for name in actions) or len(set(actions)) != len(actions):
            raise ValueError("allowed action names must be unique non-empty strings")
        if not callable(getattr(provider, "generate", None)):
            raise ValueError("provider must implement generate")
        self.provider = provider
        self.allowed_actions = frozenset(actions)
        self.max_new_tokens = max_new_tokens

    def run(self, task: str, context: str = "", *, max_new_tokens: int | None = None) -> dict:
        evidence = {"context": context} if isinstance(context, str) and context else {}
        return self.run_with_evidence(task, evidence, max_new_tokens=max_new_tokens)

    def run_with_evidence(self, task: str, evidence: Mapping[str, str], *,
                          max_new_tokens: int | None = None) -> dict:
        limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if (not isinstance(task, str) or not task.strip() or not isinstance(evidence, Mapping)
                or any(not isinstance(source, str) or not source or not isinstance(text, str)
                       for source, text in evidence.items())
                or isinstance(limit, bool) or not isinstance(limit, int)
                or not 1 <= limit <= self.max_new_tokens):
            return self._blocked("invalid_request")
        try:
            raw = self.provider.generate(task, dict(evidence),
                                         allowed_actions=self.allowed_actions,
                                         max_new_tokens=limit)
            proposal = _loads(raw)
            kind = proposal.get("kind")
            if kind == "quote":
                if set(proposal) != {"kind", "source_id", "text"}:
                    return self._blocked("invalid_proposal")
                source, text = proposal["source_id"], proposal["text"]
                if not isinstance(source, str) or source not in evidence:
                    return self._blocked("source_not_authorized")
                if (not isinstance(text, str) or not text or len(text) > MAX_QUOTE_CHARS
                        or text not in evidence[source]):
                    return self._blocked("quote_not_in_evidence")
                return {"status": "allowed", "mode": self.mode, "enforced": True,
                        "reason": "verified_exact_quote", "output": text,
                        "source_id": source, "action_proposal": None}
            if kind == "action":
                if set(proposal) != {"kind", "name", "arguments"}:
                    return self._blocked("invalid_proposal")
                name, arguments = proposal["name"], proposal["arguments"]
                if not isinstance(name, str) or name not in self.allowed_actions:
                    return self._blocked("action_not_allowed")
                if not isinstance(arguments, dict):
                    return self._blocked("invalid_proposal")
                return {"status": "action_proposed", "mode": self.mode, "enforced": True,
                        "reason": "broker_authorization_required", "output": None,
                        "source_id": None, "action_proposal": {"name": name, "arguments": arguments}}
            if kind == "refusal" and set(proposal) == {"kind"}:
                return {"status": "refused", "mode": self.mode, "enforced": True,
                        "reason": "model_refused", "output": REFUSAL_TEXT,
                        "source_id": None, "action_proposal": None}
            return self._blocked("output_type_not_allowed")
        except Exception as error:
            # Provider errors may contain prompts or secrets; expose only type.
            return {"status": "error", "mode": self.mode, "enforced": True,
                    "reason": "provider_or_parse_error", "error_type": type(error).__name__,
                    "output": None, "source_id": None, "action_proposal": None}

    def _blocked(self, reason: str) -> dict:
        return {"status": "blocked", "mode": self.mode, "enforced": True,
                "reason": reason, "output": None, "source_id": None,
                "action_proposal": None}
