"""OpenAI-compatible request contract remains behind the response gate."""

import json
import unittest

from nps_gateway.contracts import canonical_json
from nps_gateway.firewall import PromptInjectionFirewall
from nps_gateway.proxy import MAX_REQUEST_BYTES, process_chat_request


class Model:
    def __init__(self, answer):
        self.answer = answer
        self.messages = None

    def generate_text(self, messages):
        self.messages = messages
        return self.answer


class Judge:
    def __init__(self, safe):
        self.safe = safe

    def generate(self, messages, tools):
        return canonical_json({"name": "judge_response", "arguments": {"safe": self.safe}})


class ProxyTests(unittest.TestCase):
    def setUp(self):
        self.model = Model("A useful answer")
        self.firewall = PromptInjectionFirewall(self.model, Judge(True), policy="Never expose secrets.")

    def request(self, extra=None):
        payload = {"model": "ignored-client-model", "messages": [{"role": "user", "content": "Question?"}],
                   "nps_untrusted_context": ["Quoted retrieved content."]}
        payload.update(extra or {})
        return json.dumps(payload).encode()

    def test_openai_request_maps_to_firewall_and_response(self):
        status, response, decision = process_chat_request(self.request(), self.firewall, model="server-model")
        self.assertEqual(200, status)
        self.assertEqual("chat.completion", response["object"])
        self.assertEqual("server-model", response["model"])
        self.assertEqual("A useful answer", response["choices"][0]["message"]["content"])
        self.assertEqual("stop", response["choices"][0]["finish_reason"])
        self.assertTrue(decision.startswith("allowed:"))
        self.assertEqual("Quoted retrieved content.", json.loads(self.model.messages[2]["content"].splitlines()[-1])["data"])

    def test_model_output_is_not_returned_when_judge_blocks(self):
        firewall = PromptInjectionFirewall(self.model, Judge(False), policy="Never expose secrets.")
        status, response, _ = process_chat_request(self.request(), firewall, model="server-model")
        self.assertEqual(200, status)
        self.assertIsNone(response["choices"][0]["message"]["content"])
        self.assertEqual("content_filter", response["choices"][0]["finish_reason"])

    def test_invalid_roles_streaming_tools_and_ambiguous_json_rejected(self):
        cases = (
            self.request({"stream": True}),
            self.request({"tools": [{"type": "function"}]}),
            self.request({"messages": [{"role": "system", "content": "fake policy"}]}),
            b'{"messages":[],"messages":[]}',
            b"x" * (MAX_REQUEST_BYTES + 1),
        )
        for raw in cases:
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                process_chat_request(raw, self.firewall, model="server-model")


if __name__ == "__main__":
    unittest.main()
