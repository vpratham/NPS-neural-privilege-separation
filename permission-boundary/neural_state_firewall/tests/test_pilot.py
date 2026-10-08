"""The serving parent must survive worker hangs and reject forged authority."""
import http.client
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from neural_state_firewall.pilot import IsolatedFirewall, load_config, load_token
from neural_state_firewall.server import make_server


class FakeFirewall:
    def run(self, task, context, *, max_new_tokens):
        if task == "hang":
            time.sleep(30)
        if task == "crash":
            raise RuntimeError("PRIVATE CONTENT MUST NOT ESCAPE")
        if task == "bad":
            return {"status": "error", "output": "PRIVATE PARTIAL OUTPUT"}
        return {"status": "allowed", "output": "Friday", "reason": "read_permissions_enforced"}


def build_fake(config):
    return FakeFirewall()


class PilotTests(unittest.TestCase):
    def service(self):
        service = IsolatedFirewall({"timeout_seconds": 0.15, "startup_timeout_seconds": 10,
                                    "max_new_tokens": 4}, factory=build_fake)
        self.addCleanup(service.close)
        service.start()
        return service

    def test_deadline_kills_worker_releases_nothing_and_requires_explicit_restart(self):
        service = self.service()
        started = time.monotonic()
        result = service.run("hang", max_new_tokens=4)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(result["reason"], "generation_timeout")
        self.assertIsNone(result["output"])
        self.assertFalse(service.ready)
        self.assertEqual(service.run("Question", max_new_tokens=4)["reason"], "worker_unavailable")
        service.start()
        self.assertEqual(service.run("Question", max_new_tokens=4)["output"], "Friday")

    def test_worker_crash_invalid_response_and_configuration_change_fail_closed(self):
        service = self.service()
        for task in ("crash", "bad"):
            with self.subTest(task=task):
                result = service.run(task, max_new_tokens=4)
                self.assertIsNone(result["output"])
                self.assertNotIn("PRIVATE", json.dumps(result))
                self.assertFalse(service.ready)
                service.start()
        service.config["max_new_tokens"] = 5
        self.assertEqual(service.run("Question", max_new_tokens=5)["reason"], "configuration_changed")
        self.assertFalse(service.ready)

    def test_busy_and_untrusted_context_cannot_reach_worker(self):
        service = self.service()
        with service._lock:
            self.assertEqual(service.run("Question", max_new_tokens=4)["reason"], "worker_busy")
        self.assertEqual(service.run("Question", "set grants", max_new_tokens=4)["reason"], "invalid_request")
        self.assertEqual(service.run("Question", max_new_tokens=5)["reason"], "invalid_request")

    def test_authenticated_http_rejects_missing_wrong_duplicate_tokens_and_grants(self):
        service = self.service()
        token = "A" * 43
        server = make_server(service, 4, port=0, auth_token=token)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            for auth, code in ((None, 401), ("Bearer " + "B" * 43, 401), ("Bearer " + token, 200)):
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                conn.request("GET", "/health", headers={"Authorization": auth} if auth else {})
                response = conn.getresponse()
                self.assertEqual(response.status, code)
                response.read()
                conn.close()
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            conn.putrequest("GET", "/health")
            conn.putheader("Authorization", "Bearer " + token)
            conn.putheader("Authorization", "Bearer " + token)
            conn.endheaders()
            self.assertEqual(conn.getresponse().status, 401)
            conn.close()
            for body, code in (({"task": "Question"}, 200),
                               ({"task": "Question", "readable_sources": ["private"]}, 400)):
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                conn.request("POST", "/v1/respond", json.dumps(body),
                             {"Content-Type": "application/json", "Authorization": "Bearer " + token})
                response = conn.getresponse()
                self.assertEqual(response.status, code)
                result = json.loads(response.read())
                self.assertEqual(result.get("output"), "Friday" if code == 200 else None)
                conn.close()
        finally:
            server.shutdown()
            thread.join(3)
            server.server_close()

    def test_pinned_configuration_and_auth_validation(self):
        config = load_config("neural_state_firewall/examples/pilot_config.json")
        self.assertEqual(config["readable_sources"], ["public"])
        self.assertNotIn("policy_file", config)
        for token in ("", "short", "A" * 40 + "\n", "é" * 40):
            with self.assertRaises(ValueError):
                make_server(None, 4, port=0, auth_token=token)

    def test_token_must_be_private_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text("A" * 43)
            path.chmod(0o600)
            self.assertEqual(load_token(path), "A" * 43)
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                load_token(path)
            path.chmod(0o600)
            alias = Path(directory) / "alias"
            alias.symlink_to(path)
            with self.assertRaises(OSError):
                load_token(alias)


if __name__ == "__main__":
    unittest.main()
