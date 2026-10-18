import json
import unittest
from unittest.mock import patch

from npb_integrated.src.providers import OllamaProvider


class _Response:
    def __init__(self, value):
        self.value = value

    def read(self, _limit):
        return json.dumps(self.value).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class OllamaProviderTests(unittest.TestCase):
    def test_cpu_options_pinned_without_retries(self):
        replies = iter([
            {"models": [{"name": "qwen2.5:3b", "digest": "sha256:fixed"}]},
            {"message": {"content": "12%"}, "prompt_eval_count": 7,
             "eval_count": 2, "done_reason": "stop"},
            {"models": [{"name": "qwen2.5:3b", "size_vram": 0}]},
        ])
        requests = []

        def open_mock(request, timeout):
            requests.append((request, timeout))
            return _Response(next(replies))

        with patch("npb_integrated.src.providers.urlopen", side_effect=open_mock):
            provider = OllamaProvider("qwen2.5:3b", base_url="http://127.0.0.1:11436")
            result = provider.generate([{"role": "user", "content": "answer"}])
        self.assertEqual(provider.calls, 1)
        self.assertEqual(len(requests), 3)
        self.assertTrue(all("/api/pull" not in request.full_url for request, _ in requests))
        payload = json.loads(requests[1][0].data.decode("utf-8"))
        self.assertEqual(payload["options"], {"num_gpu": 0, "num_ctx": 4096, "num_predict": 128,
                                               "temperature": 0, "seed": 17, "num_thread": 4})
        self.assertEqual(result["gpu_status"]["size_vram"], 0)
        self.assertEqual(result["messages"], [{"role": "user", "content": "answer"}])

    def test_nonzero_vram_stops_after_one_post(self):
        replies = iter([
            {"models": [{"name": "qwen2.5:3b", "digest": "sha256:fixed"}]},
            {"message": {"content": "12%"}},
            {"models": [{"name": "qwen2.5:3b", "size_vram": 1}]},
        ])
        with patch("npb_integrated.src.providers.urlopen", side_effect=lambda *_, **__: _Response(next(replies))):
            provider = OllamaProvider("qwen2.5:3b", base_url="http://127.0.0.1:11436")
            with self.assertRaisesRegex(RuntimeError, "gpu_execution"):
                provider.generate([{"role": "user", "content": "answer"}])
        self.assertEqual(provider.calls, 1)
