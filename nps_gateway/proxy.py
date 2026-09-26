"""Small OpenAI Chat Completions compatible, buffered firewall endpoint."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import time
from typing import Any

from .contracts import strict_json
from .firewall import PromptInjectionFirewall

MAX_REQUEST_BYTES = 256_000


def process_chat_request(raw: bytes, firewall: PromptInjectionFirewall, *, model: str) -> tuple[int, dict, str]:
    """Validate one text chat request and construct a gated response.

    Client-supplied system/developer policy is rejected. Trusted policy comes
    from service configuration; prior turns and explicit retrieved text are
    passed as untrusted context for this single-turn integration surface.
    """
    if not isinstance(raw, bytes) or len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("request_too_large")
    try:
        payload = strict_json(raw.decode("utf-8", "strict"), limit=MAX_REQUEST_BYTES)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid_request_json") from exc
    if not isinstance(payload, dict) or set(payload) - {
        "model", "messages", "stream", "max_tokens", "temperature", "nps_untrusted_context",
    }:
        raise ValueError("unsupported_request_fields")
    if payload.get("stream", False) is not False:
        raise ValueError("streaming_not_supported")
    if "max_tokens" in payload and (type(payload["max_tokens"]) is not int or not 1 <= payload["max_tokens"] <= 8192):
        raise ValueError("invalid_max_tokens")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 32:
        raise ValueError("invalid_messages")
    texts: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content"}:
            raise ValueError("unsupported_message_shape")
        role, content = message.get("role"), message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str) or len(content) > 16_000:
            # Trusted system policy is service-owned, never caller-owned.
            raise ValueError("invalid_or_untrusted_message_role")
        texts.append((role, content))
    last_user = next((index for index in range(len(texts) - 1, -1, -1) if texts[index][0] == "user"), None)
    if last_user is None:
        raise ValueError("user_task_required")
    task = texts[last_user][1]
    context = [f"Prior {role} turn: {content}" for role, content in texts[:last_user]]
    supplied_context = payload.get("nps_untrusted_context", [])
    if not isinstance(supplied_context, list) or any(not isinstance(item, str) for item in supplied_context):
        raise ValueError("invalid_untrusted_context")
    context.extend(supplied_context)
    result = firewall.complete(task=task, untrusted_context=context)
    response = {
        "id": "chatcmpl-" + secrets.token_hex(12),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result.answer if result.allowed else None},
            "finish_reason": "stop" if result.allowed else "content_filter",
        }],
    }
    status = 200 if result.status in {"allowed", "blocked"} else 502
    decision = result.status + ":" + result.reason
    return status, response, decision


class FirewallHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(*, firewall: PromptInjectionFirewall, model: str, host: str = "127.0.0.1", port: int = 8765):
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("bind_loopback_only_add_authentication_and_tls_for_remote_use")

    class Handler(BaseHTTPRequestHandler):
        server_version = "NPS-Firewall/0.1"

        def log_message(self, format, *args):
            # Request prompts and model outputs are sensitive; do not log them.
            return

        def _send(self, status: int, body: dict, decision: str):
            encoded = json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-NPS-Firewall-Decision", decision[:120])
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ready", "model": model}, "health")
            else:
                self._send(404, {"error": {"message": "not found", "type": "not_found"}}, "rejected")

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found", "type": "not_found"}}, "rejected")
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid_content_length")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("incomplete_request")
                status, response, decision = process_chat_request(raw, firewall, model=model)
            except (ValueError, UnicodeError) as exc:
                self._send(400, {"error": {"message": str(exc), "type": "invalid_request_error"}}, "rejected")
                return
            except Exception:
                # No provider exception details or request contents cross boundary.
                self._send(502, {"error": {"message": "firewall check failed", "type": "firewall_error"}}, "error")
                return
            self._send(status, response, decision)

    return FirewallHTTPServer((host, port), Handler)
