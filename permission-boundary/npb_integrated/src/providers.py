"""Local-only model provider used by the bounded integrated evaluation."""
from __future__ import annotations

import json
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LocalHFProvider:
    """Greedy CPU inference from an already cached Hugging Face causal model."""

    MAX_INPUT_TOKENS = 4096

    def __init__(self, model_path: str, *, max_new_tokens: int = 192):
        if not isinstance(model_path, str) or not model_path:
            raise ValueError("model_path_required")
        if not 1 <= max_new_tokens <= 192:
            raise ValueError("max_new_tokens_must_be_between_1_and_192")
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._lock = threading.Lock()
        self.calls = 0
        self.stopped = False
        self.max_new_tokens = max_new_tokens
        # local_files_only prevents this prototype from fetching weights at runtime.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        ).to("cpu").eval()
        self.identity = {
            "provider": "local_huggingface",
            "model": model_path,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "device": "cpu",
            "dtype": "float32",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "decoding": "greedy",
            "max_new_tokens": max_new_tokens,
            "local_files_only": True,
            "trust_remote_code": False,
        }

    def _prompt(self, messages: list[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n\n".join(
            f"{message['role'].upper()}: {message['content']}" for message in messages
        ) + "\n\nASSISTANT:"

    def generate(self, messages: list[dict]) -> dict:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages_required")
        start = time.perf_counter()
        prompt = self._prompt(messages)
        with self._lock, self._torch.inference_mode():
            encoded = self.tokenizer(prompt, return_tensors="pt")
            input_tokens = int(encoded["input_ids"].shape[-1])
            if input_tokens > self.MAX_INPUT_TOKENS:
                raise ValueError("model_input_too_long")
            pad_token_id = self.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            self.calls += 1
            output = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
        generated = output[0, input_tokens:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        return {
            "text": text,
            "input_tokens": input_tokens,
            "output_tokens": int(generated.shape[-1]),
            "elapsed_seconds": time.perf_counter() - start,
            "finish_reason": "eos" if len(generated) and int(generated[-1]) in eos_ids else "length",
            "messages": json.loads(json.dumps(messages)),
        }


class OllamaProvider:
    """Pinned, loopback-only Ollama transport with an explicit CPU-only request."""

    MAX_REPLY_BYTES = 4 * 1024 * 1024

    def __init__(self, model: str, *, base_url: str, seed: int = 17, timeout: int = 60):
        parsed = urlparse(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username is not None or parsed.password is not None):
            raise ValueError("loopback_http_base_url_required")
        if not isinstance(model, str) or not model:
            raise ValueError("model_required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.seed = seed
        self.timeout = timeout
        self.calls = 0
        self.stopped = False
        self._lock = threading.Lock()
        tags = self._request_json("GET", "/api/tags")
        models = tags.get("models") if isinstance(tags, dict) else None
        match = next((entry for entry in models or [] if entry.get("name") == model), None)
        if not isinstance(match, dict) or not isinstance(match.get("digest"), str):
            raise ValueError("requested_ollama_model_not_available")
        self.identity = {
            "provider": "ollama_loopback",
            "model": model,
            "model_digest": match["digest"],
            "base_url": self.base_url,
            "device": "cpu",
            "options": self._options(),
            "stream": False,
            "timeout_seconds": timeout,
            "no_pull": True,
        }

    def _options(self) -> dict:
        return {"num_gpu": 0, "num_ctx": 4096, "num_predict": 128,
                "temperature": 0, "seed": self.seed, "num_thread": 4}

    def _request_json(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(self.base_url + path, data=data, method=method,
                          headers={"Content-Type": "application/json"} if data is not None else {})
        with urlopen(request, timeout=self.timeout) as response:  # no retry or redirect loop
            raw = response.read(self.MAX_REPLY_BYTES + 1)
        if len(raw) > self.MAX_REPLY_BYTES:
            raise ValueError("ollama_reply_too_large")
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("invalid_ollama_response")
        return parsed

    def _gpu_status(self) -> dict:
        payload = self._request_json("GET", "/api/ps")
        models = payload.get("models")
        match = next((entry for entry in models or [] if entry.get("name") == self.model), None)
        if not isinstance(match, dict) or not isinstance(match.get("size_vram"), int):
            self.stopped = True
            raise ValueError("ollama_gpu_status_unavailable")
        status = {"model": self.model, "size_vram": match["size_vram"], "cpu_only_verified": match["size_vram"] == 0}
        if not status["cpu_only_verified"]:
            self.stopped = True
            raise RuntimeError("ollama_gpu_execution_detected")
        return status

    def generate(self, messages: list[dict]) -> dict:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages_required")
        start = time.perf_counter()
        payload = {"model": self.model, "messages": messages, "stream": False, "options": self._options()}
        with self._lock:
            if self.stopped or self.calls >= 80:
                raise RuntimeError('provider_stopped_or_call_limit')
            self.calls += 1  # Count an attempted model call even if transport fails.
            response = self._request_json("POST", "/api/chat", payload)
            gpu_status = self._gpu_status()
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("invalid_ollama_chat_response")
        return {
            "text": message["content"], "input_tokens": response.get("prompt_eval_count"),
            "output_tokens": response.get("eval_count"), "elapsed_seconds": time.perf_counter() - start,
            "finish_reason": response.get("done_reason", "unknown"), "gpu_status": gpu_status,
            "messages": json.loads(json.dumps(messages)),
        }
