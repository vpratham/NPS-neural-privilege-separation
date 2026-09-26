"""Small, defensive adapters for tool-capable chat providers.

The gateway accepts only the canonical ``{"name", "arguments"}`` proposal.
These adapters deliberately reject ordinary text and ambiguous provider replies
instead of trying to interpret them.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import math
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .contracts import canonical_json, strict_json

MAX_RESPONSE_BYTES = 1_048_576


class ProviderError(RuntimeError):
    """A deliberately non-diagnostic provider failure safe for callers to expose."""

    def __init__(self) -> None:
        super().__init__("provider_error")


class _NoRedirect(HTTPRedirectHandler):
    """Make every redirect fail; credentials must never follow a new origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _default_opener(request: Request, *, timeout: float):
    # Empty proxy configuration is required even for loopback URLs: environment
    # proxy settings can otherwise route local requests through an untrusted host.
    return build_opener(ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout)


def _validated_base_url(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("invalid_provider_url")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("invalid_provider_url")
    if parsed.query or parsed.fragment:
        raise ValueError("invalid_provider_url")
    host = parsed.hostname
    if host is None:
        raise ValueError("invalid_provider_url")
    if parsed.scheme == "http":
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = host.lower() == "localhost"
        if not is_loopback:
            raise ValueError("insecure_provider_url")
    return value.rstrip("/")


def _read_json(response: Any) -> dict[str, Any]:
    content_length = response.headers.get("Content-Length") if hasattr(response, "headers") else None
    try:
        if content_length is not None and int(content_length) > MAX_RESPONSE_BYTES:
            raise ValueError("response_too_large")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ProviderError() from exc
    if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
        raise ProviderError()
    try:
        decoded = raw.decode("utf-8", "strict")
        parsed = strict_json(decoded, limit=MAX_RESPONSE_BYTES)
    except (UnicodeError, ValueError) as exc:
        raise ProviderError() from exc
    if not isinstance(parsed, dict):
        raise ProviderError()
    return parsed


class _HTTPAdapter:
    def __init__(self, base_url: str, *, model: str, timeout: float = 30.0,
                 max_tokens: int = 256, opener: Callable[..., Any] | None = None) -> None:
        self.base_url = _validated_base_url(base_url)
        if not isinstance(model, str) or not model:
            raise ValueError("invalid_model")
        if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("invalid_timeout")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("invalid_max_tokens")
        self.model = model
        self.timeout = float(timeout)
        self.max_tokens = max_tokens
        self._opener = opener or _default_opener

    def _post(self, path: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if headers:
            request_headers.update(headers)
        try:
            body = canonical_json(payload).encode("ascii")
            request = Request(self.base_url + path, data=body, headers=request_headers, method="POST")
            response = self._opener(request, timeout=self.timeout)
            try:
                return _read_json(response)
            finally:
                close = getattr(response, "close", None)
                if close is not None:
                    close()
        except ProviderError:
            raise
        except (HTTPError, URLError, OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError) as exc:
            raise ProviderError() from exc


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and value != ""


def _no_prose(message: dict[str, Any]) -> None:
    # Tool-call responses must be machine proposals, never a mixed prose/action response.
    if message.get("content") not in (None, ""):
        raise ProviderError()


def _assistant_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ProviderError()
    _no_prose(message)
    return message


def _no_provider_error(reply: dict[str, Any]) -> None:
    if "error" in reply:
        raise ProviderError()


def _canonical_proposal(name: Any, arguments: Any) -> str:
    if not _nonempty_text(name) or not isinstance(arguments, dict):
        raise ProviderError()
    try:
        return canonical_json({"name": name, "arguments": arguments})
    except (TypeError, ValueError) as exc:
        raise ProviderError() from exc


def _ollama_messages(messages: list[dict]) -> list[dict]:
    if not isinstance(messages, list):
        raise ProviderError()
    converted: list[dict] = []
    for source in messages:
        if not isinstance(source, dict) or not isinstance(source.get("role"), str):
            raise ProviderError()
        message = copy.deepcopy(source)
        if message["role"] == "tool":
            name = message.pop("name", None)
            message.pop("tool_call_id", None)
            if not _nonempty_text(name):
                raise ProviderError()
            message["tool_name"] = name
        if message["role"] == "assistant" and "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list):
                raise ProviderError()
            normalized = []
            for call in calls:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    raise ProviderError()
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = strict_json(arguments)
                    except ValueError as exc:
                        raise ProviderError() from exc
                if not isinstance(arguments, dict) or not _nonempty_text(function.get("name")):
                    raise ProviderError()
                normalized.append({"function": {"name": function["name"], "arguments": arguments}})
            message["tool_calls"] = normalized
        converted.append(message)
    return converted


class OllamaAdapter(_HTTPAdapter):
    """Native Ollama ``/api/chat`` adapter using its structured tool calls."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", *, model: str,
                 timeout: float = 30.0, max_tokens: int = 256,
                 opener: Callable[..., Any] | None = None) -> None:
        super().__init__(base_url, model=model, timeout=timeout, max_tokens=max_tokens, opener=opener)

    def generate(self, messages: list[dict], tools: list[dict]) -> str:
        reply = self._post("/api/chat", {
            "model": self.model,
            "messages": _ollama_messages(messages),
            "tools": tools,
            "stream": False,
            "options": {"num_predict": self.max_tokens, "temperature": 0},
        })
        _no_provider_error(reply)
        if reply.get("done") is not True or reply.get("done_reason") != "stop":
            raise ProviderError()
        message = _assistant_message(reply.get("message"))
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
            raise ProviderError()
        function = calls[0].get("function")
        if not isinstance(function, dict):
            raise ProviderError()
        return _canonical_proposal(function.get("name"), function.get("arguments"))


class ChatCompletionsAdapter(_HTTPAdapter):
    """Generic OpenAI-compatible ``/v1/chat/completions`` adapter.

    It keeps canonical gateway history unchanged. Compatible APIs return tool
    arguments as a JSON string and tool result messages retain ``tool_call_id``.
    """

    def __init__(self, base_url: str, *, model: str, api_key: str | None = None,
                 timeout: float = 30.0, max_tokens: int = 256,
                 opener: Callable[..., Any] | None = None) -> None:
        super().__init__(base_url, model=model, timeout=timeout, max_tokens=max_tokens, opener=opener)
        path = urlsplit(self.base_url).path.rstrip("/")
        if path not in {"", "/v1"}:
            raise ValueError("invalid_provider_url")
        self._chat_path = "/chat/completions" if path == "/v1" else "/v1/chat/completions"
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise ValueError("invalid_api_key")
        self.api_key = api_key

    def generate(self, messages: list[dict], tools: list[dict]) -> str:
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else None
        reply = self._post(self._chat_path, {
            "model": self.model,
            "messages": copy.deepcopy(messages),
            "tools": copy.deepcopy(tools),
            "max_tokens": self.max_tokens,
            "stream": False,
            "temperature": 0,
        }, headers)
        _no_provider_error(reply)
        choices = reply.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ProviderError()
        choice = choices[0]
        if choice.get("finish_reason") != "tool_calls":
            raise ProviderError()
        message = _assistant_message(choice.get("message"))
        calls = message.get("tool_calls")
        if (not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict)
                or calls[0].get("type") != "function"):
            raise ProviderError()
        function = calls[0].get("function")
        if not isinstance(function, dict):
            raise ProviderError()
        raw_arguments = function.get("arguments")
        if not isinstance(raw_arguments, str):
            raise ProviderError()
        try:
            arguments = strict_json(raw_arguments)
        except ValueError as exc:
            raise ProviderError() from exc
        return _canonical_proposal(function.get("name"), arguments)


class ScriptedAdapter:
    """Offline static response adapter for integration tests and local demos only."""

    def __init__(self, proposal: str) -> None:
        if not isinstance(proposal, str):
            raise ValueError("invalid_proposal")
        self.proposal = proposal

    def generate(self, messages: list[dict], tools: list[dict]) -> str:
        return self.proposal
