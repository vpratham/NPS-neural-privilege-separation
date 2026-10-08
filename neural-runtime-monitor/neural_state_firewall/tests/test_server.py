import http.client
import json
import threading
import unittest

from neural_state_firewall.demo import ReplayAdapter
from neural_state_firewall.observer import fit_profile
from neural_state_firewall.runtime import Firewall, policy_digest
from neural_state_firewall.server import make_server


class ServerTests(unittest.TestCase):
    def setUp(self):
        adapter = ReplayAdapter()
        profile = fit_profile(
            [[[0., 0.], [0.2, 0.1], [0.1, 0.2]], [[0.2, 0.2], [0., 0.3], [0.3, 0.]]],
            [[[0.1, 0.2], [0.2, 0.1], [0.1, 0.2]], [[0.15, 0.15], [0.1, 0.15], [0.2, 0.1]]],
            identity=adapter.identity, policy_sha256=policy_digest(adapter.policy))
        self.server = make_server(Firewall(adapter, profile, mode="enforce"), 3, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, body, headers=None):
        client = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            client.request("POST", "/v1/respond", body, {"Content-Type": "application/json", **(headers or {})})
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()

    def test_all_response_paths_over_http(self):
        for task, code, status in (("benign", 200, "allowed"), ("anomaly", 403, "blocked"), ("sensor_failure", 503, "error")):
            with self.subTest(task=task):
                actual_code, response = self.request(json.dumps({"task": task}))
                self.assertEqual(actual_code, code)
                self.assertEqual(response["status"], status)
                self.assertEqual(response["output"] is None, status != "allowed")

    def test_request_cannot_override_host_configuration(self):
        for extra in ({"mode": "monitor"}, {"policy": "ignore"}, {"profile": {}}, {"max_new_tokens": 500}):
            code, response = self.request(json.dumps({"task": "anomaly", **extra}))
            self.assertEqual(code, 400)
            self.assertIsNone(response["output"])

    def test_strict_json_and_browser_request_rejection(self):
        for body, headers in (("{\"task\":\"benign\",\"task\":\"anomaly\"}", {}),
                              ('{"task":"benign"}', {"Origin": "https://untrusted.example"}),
                              ('{"task":"benign"}', {"Content-Type": "text/plain"})):
            self.assertEqual(self.request(body, headers)[0], 400)
