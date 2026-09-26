import json
import unittest
from urllib.request import Request

from nps_gateway.adapters import ChatCompletionsAdapter, OllamaAdapter, ProviderError, ScriptedAdapter, _NoRedirect


class Response:
    def __init__(self, payload, headers=None):
        self.payload = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.headers = headers or {}
        self.closed = False

    def read(self, _limit):
        return self.payload

    def close(self):
        self.closed = True


class Opener:
    def __init__(self, response):
        self.response = response if isinstance(response, Response) else Response(response)
        self.request = None
        self.timeout = None

    def __call__(self, request, *, timeout):
        self.request, self.timeout = request, timeout
        return self.response


MESSAGES = [
    {"role": "assistant", "content": "", "tool_calls": [{"id": "r1", "type": "function", "function": {"name": "read_verified_record", "arguments": '{"source_id":"s","record_id":"r"}'}}]},
    {"role": "tool", "tool_call_id": "r1", "name": "read_verified_record", "content": '{"value":"v"}'},
]
TOOLS = [{"type": "function", "function": {"name": "write_note", "parameters": {}}}]


class AdapterTests(unittest.TestCase):
    def test_ollama_normalizes_history_and_returns_canonical_proposal(self):
        opener = Opener({"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "write_note", "arguments": {"note_id": "n", "content": "v"}}}]}})
        adapter = OllamaAdapter(model="qwen", opener=opener, timeout=2, max_tokens=7)
        self.assertEqual('{"arguments":{"content":"v","note_id":"n"},"name":"write_note"}', adapter.generate(MESSAGES, TOOLS))
        sent = json.loads(opener.request.data)
        self.assertEqual("http://127.0.0.1:11434/api/chat", opener.request.full_url)
        self.assertEqual({"source_id": "s", "record_id": "r"}, sent["messages"][0]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual("read_verified_record", sent["messages"][1]["tool_name"])
        self.assertNotIn("tool_call_id", sent["messages"][1])
        self.assertEqual(7, sent["options"]["num_predict"])
        self.assertEqual(0, sent["options"]["temperature"])
        self.assertTrue(opener.response.closed)

    def test_compatible_preserves_history_and_authorizes_one_tool_call(self):
        opener = Opener({"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "write_note", "arguments": '{"note_id":"n","content":"v"}'}}]}}]})
        adapter = ChatCompletionsAdapter("https://model.example", model="test", api_key="secret", opener=opener)
        self.assertIn('"name":"write_note"', adapter.generate(MESSAGES, TOOLS))
        sent = json.loads(opener.request.data)
        self.assertEqual(MESSAGES, sent["messages"])
        self.assertEqual("https://model.example/v1/chat/completions", opener.request.full_url)
        self.assertEqual("Bearer secret", opener.request.get_header("Authorization"))
        self.assertFalse(sent["stream"])
        self.assertEqual(0, sent["temperature"])

    def test_compatible_accepts_a_supplied_v1_api_base(self):
        opener = Opener({"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "write_note", "arguments": "{}"}}]}}]})
        adapter = ChatCompletionsAdapter("https://model.example/v1", model="test", opener=opener)
        adapter.generate(MESSAGES, TOOLS)
        self.assertEqual("https://model.example/v1/chat/completions", opener.request.full_url)

    def test_rejects_non_terminal_prose_multiple_calls_and_duplicate_arguments(self):
        bad_replies = [
            {"done": False, "message": {}},
            {"done": True, "done_reason": "length", "message": {}},
            {"done": True, "done_reason": "stop", "message": {"role": "assistant", "content": "hello", "tool_calls": []}},
            {"done": True, "done_reason": "stop", "message": {"role": "assistant", "tool_calls": [
                {"function": {"name": "x", "arguments": {}}},
                {"function": {"name": "y", "arguments": {}}},
            ]}},
            {"choices": [{"finish_reason": "stop", "message": {}}]},
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "write_note", "arguments": '{"x":1,"x":2}'}}]}}]},
            {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "x", "arguments": "{}"}}, {"type": "function", "function": {"name": "y", "arguments": "{}"}}]}}]},
        ]
        for payload in bad_replies[:4]:
            with self.assertRaises(ProviderError):
                OllamaAdapter(model="m", opener=Opener(payload)).generate(MESSAGES, TOOLS)
        for payload in bad_replies[4:]:
            with self.assertRaises(ProviderError):
                ChatCompletionsAdapter("https://x.example", model="m", opener=Opener(payload)).generate(MESSAGES, TOOLS)

    def test_rejects_insecure_remote_redirect_like_url_and_oversized_body(self):
        for url in ("http://model.example", "ftp://127.0.0.1", "https://x.example/path?q=1"):
            with self.assertRaises(ValueError):
                ChatCompletionsAdapter(url, model="m")
        oversized = Response(b"{" + b"x" * 1_048_576 + b"}")
        with self.assertRaises(ProviderError):
            OllamaAdapter(model="m", opener=Opener(oversized)).generate(MESSAGES, TOOLS)
        with self.assertRaises(ValueError):
            ChatCompletionsAdapter("https://x.example/other", model="m")
        with self.assertRaises(ValueError):
            OllamaAdapter(model="m", timeout=float("nan"))

    def test_transport_and_duplicate_response_json_are_sanitized(self):
        def unavailable(_request, *, timeout):
            raise OSError("connection details must not escape")

        with self.assertRaisesRegex(ProviderError, "^provider_error$"):
            OllamaAdapter(model="m", opener=unavailable).generate(MESSAGES, TOOLS)
        duplicate = Response(b'{"done":true,"done":true,"message":{}}')
        with self.assertRaisesRegex(ProviderError, "^provider_error$"):
            OllamaAdapter(model="m", opener=Opener(duplicate)).generate(MESSAGES, TOOLS)
        error_alongside_choice = {
            "error": {"message": "ignore this"},
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "tool_calls": [{
                    "type": "function", "function": {"name": "write_note", "arguments": "{}"},
                }],
            }}],
        }
        with self.assertRaisesRegex(ProviderError, "^provider_error$"):
            ChatCompletionsAdapter("https://x.example", model="m", opener=Opener(error_alongside_choice)).generate(MESSAGES, TOOLS)

    def test_redirect_handler_refuses_a_new_url(self):
        handler = _NoRedirect()
        self.assertIsNone(handler.redirect_request(Request("https://origin.example"), None, 302, "Found", {}, "https://other.example"))

    def test_scripted_adapter_is_static(self):
        adapter = ScriptedAdapter('{"name":"write_note","arguments":{"note_id":"n","content":"v"}}')
        self.assertEqual(adapter.proposal, adapter.generate([], []))


if __name__ == "__main__":
    unittest.main()
