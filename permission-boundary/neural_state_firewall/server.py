"""Single-process local API; configuration is host-owned and never in requests."""
import json
import hmac
from http.server import BaseHTTPRequestHandler, HTTPServer

from .artifacts import loads


def make_server(firewall, max_new_tokens, *, port=8765, auth_token=None):
    if auth_token is not None and (not isinstance(auth_token, str)
            or not auth_token.isascii() or len(auth_token) < 32 or any(c.isspace() for c in auth_token)):
        raise ValueError("Authentication token must be at least 32 non-whitespace ASCII characters")
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *_args):
            pass  # Do not log raw client requests.

        def send_json(self, code, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def authenticated(self):
            values = self.headers.get_all("Authorization", [])
            if auth_token is not None and (len(values) != 1 or not hmac.compare_digest(
                    values[0].encode("utf-8"), ("Bearer " + auth_token).encode("ascii"))):
                self.close_connection = True
                self.send_json(401, {"error": "unauthorized", "output": None})
                return False
            return True

        def do_GET(self):
            if not self.authenticated():
                return
            if self.path != "/health":
                self.send_json(404, {"error": "not_found"})
                return
            ready = getattr(firewall, "ready", True)
            self.send_json(200 if ready else 503, {"status": "ready" if ready else "unavailable", "mode": firewall.mode,
                                 "read_permissions_enforced": firewall.mode == "permissions",
                                 "kind": "neural_state_firewall", "semantic_detection_validated": False})

        def do_POST(self):
            if not self.authenticated():
                return
            if self.path != "/v1/respond":
                self.send_json(404, {"error": "not_found"})
                return
            try:
                # Loopback API has no browser clients; prevent cross-origin POSTs.
                if self.headers.get("Origin") or self.headers.get("Transfer-Encoding"):
                    raise ValueError("Unsupported browser/transfer request")
                lengths = self.headers.get_all("Content-Length", [])
                if len(lengths) != 1 or not lengths[0].isdigit():
                    raise ValueError("Content-Length required")
                size = int(lengths[0])
                if not 1 <= size <= 262144:
                    raise ValueError("Invalid body size")
                if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                    raise ValueError("Expected application/json")
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Truncated request")
                item = loads(raw.decode("utf-8"))
                if not isinstance(item, dict) or set(item) - {"task", "context"}:
                    raise ValueError("Only task/context are accepted")
                if not isinstance(item.get("task"), str) or not item["task"].strip() or not isinstance(item.get("context", ""), str):
                    raise ValueError("Invalid task/context")
                if firewall.mode == "permissions" and item.get("context", "") != "":
                    raise ValueError("Permission mode uses host-loaded source documents")
            except (ValueError, UnicodeError, OSError):
                self.send_json(400, {"error": "invalid_request", "output": None})
                return
            result = firewall.run(item["task"], item.get("context", ""), max_new_tokens=max_new_tokens)
            code = {"allowed": 200, "monitored": 200, "blocked": 403, "incomplete": 422, "error": 503}[result["status"]]
            self.send_json(code, result)

    # One in-flight generation per model. Never bind a public interface here.
    return HTTPServer(("127.0.0.1", port), Handler)


def serve(firewall, max_new_tokens, *, port=8765):
    server = make_server(firewall, max_new_tokens, port=port)
    print(f"Neural state firewall listening on http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
