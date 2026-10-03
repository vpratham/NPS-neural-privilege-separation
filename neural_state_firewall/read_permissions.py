"""Host-assigned document read permissions enforced at every attention layer.

Public queries can read only public keys. Denied queries can read only themselves.
Policy K/V is sealed separately. This restricts information flow, not the semantic
instruction authority of readable evidence. Host code, weights and backend are trusted.
"""
from __future__ import annotations

import re
import torch
import transformers

from .artifacts import digest
from .policy_memory import PolicyMemoryAdapter


def validate_sources(documents, readable_sources):
    if not isinstance(documents, dict) or len(documents) > 64:
        raise ValueError("documents must be a host-loaded mapping of at most 64 sources")
    if (not isinstance(readable_sources, (list, tuple))
            or any(not isinstance(name, str) for name in readable_sources)
            or len(set(readable_sources)) != len(readable_sources)):
        raise ValueError("readable_sources must be a unique host-controlled list")
    for name, text in documents.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name):
            raise ValueError("Invalid source ID")
        if not isinstance(text, str) or len(text) > 80000:
            raise ValueError("Invalid source text")
    if any(not isinstance(name, str) or name not in documents for name in readable_sources):
        raise ValueError("Read grant names an unknown source")
    if sum(len(text) for text in documents.values()) > 160000:
        raise ValueError("Source text budget exceeded")
    return tuple(documents.items()), tuple(sorted(readable_sources))


def permission_mask(readable, start, count, device):
    """Return additive causal ACL mask and public logical position IDs."""
    if (not isinstance(readable, tuple) or any(type(x) is not bool for x in readable)
            or type(start) is not int or type(count) is not int
            or start < 0 or count < 1 or start + count != len(readable)):
        raise ValueError("Invalid read-permission layout")
    public = torch.tensor(readable, dtype=torch.bool, device=device)
    key = torch.arange(len(readable), device=device)
    query = torch.arange(start, start + count, device=device)
    allowed = (public[query, None] & public[None, :] & (key[None, :] <= query[:, None]))
    allowed |= (~public[query, None]) & (key[None, :] == query[:, None])
    if not allowed.any(dim=-1).all():
        raise ValueError("A query has no permitted attention source")
    mask = torch.zeros((count, len(readable)), dtype=torch.float32, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    # Denied length does not shift public RoPE positions. Length/timing and
    # resource-limit failures are outside the confidentiality claim.
    positions = (public.long().cumsum(0) - 1).clamp_min(0)
    positions = torch.where(public, positions, 0)
    return mask[None, None], positions[start:][None]


class ReadPermissionAdapter(PolicyMemoryAdapter):
    SUPPORTED_MODEL_TYPES = ("qwen2", "llama")
    DECODER_VERSION = "greedy_cached_read_permissions_v1"
    DENIED_TOKEN_SLOTS = 256

    def _set_sources(self, documents, readable_sources):
        self._documents, self._readable_sources = validate_sources(documents, readable_sources)
        self._layout = None

    def __init__(self, model_name, *, documents, readable_sources, **kwargs):
        self._set_sources(documents, readable_sources)
        super().__init__(model_name, **kwargs)

    @classmethod
    def from_components(cls, model, tokenizer, *, documents, readable_sources, **kwargs):
        instance = cls.__new__(cls)
        instance._set_sources(documents, readable_sources)
        model.eval()
        instance._configure(model, tokenizer, layers=kwargs.pop("layers", None),
                            projection_dim=kwargs.pop("projection_dim", 8),
                            seed=kwargs.pop("seed", 17), max_context=kwargs.pop("max_context", 2048),
                            device=kwargs.pop("device", str(next(model.parameters()).device)),
                            model_identifier=kwargs.pop("model_identifier", None)
                            or "in_memory:" + cls._hash_model(model),
                            transformers_version=transformers.__version__, **kwargs)
        return instance

    def _configure(self, model, tokenizer, **kwargs):
        from transformers import Qwen2ForCausalLM, LlamaForCausalLM
        family = model.config.model_type
        if (transformers.__version__ != "4.57.6" or family not in self.SUPPORTED_MODEL_TYPES
                or type(model) not in (Qwen2ForCausalLM, LlamaForCausalLM)):
            raise ValueError("Read permissions support Transformers 4.57.6 Qwen2/Llama only")
        if (model.config._attn_implementation not in ("eager", "sdpa")
                or getattr(model.model, "has_sliding_layers", False)
                or getattr(model.config, "rope_scaling", None)):
            raise ValueError("Requires full attention, ordinary RoPE, and eager or SDPA backend")
        self.SENSOR_SITE = f"{family}.decoder_block_output.last_token.v1"
        super()._configure(model, tokenizer, **kwargs)
        self._identity.update(adapter=type(self).__name__,
                              read_permissions_sha256=digest(self._readable_sources),
                              documents_sha256=digest(self._documents),
                              denied_token_slots=self.DENIED_TOKEN_SLOTS,
                              attention_backend=model.config._attn_implementation)

    def _capture_baseline(self):
        state = super()._capture_baseline()
        state["read_permissions"] = digest((self._documents, self._readable_sources))
        state["attention_backend"] = self.model.config._attn_implementation
        state["denied_token_slots"] = self.DENIED_TOKEN_SLOTS
        return state

    def _serialize(self, task, context):
        if not isinstance(task, str) or not task.strip() or len(task) > 16000 or context != "":
            raise ValueError("Use a task and host-loaded documents; request context cannot set source provenance")
        marker = "__NPS_HOST_DOCUMENT_INSERTION__"
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.policy}, {"role": "user", "content": marker}],
            tokenize=False, add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or rendered.count(marker) != 1:
            raise ValueError("Chat template must preserve one document insertion point")
        head, tail = rendered.split(marker)
        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)
        def encode_data(text):
            backend = getattr(self.tokenizer, "backend_tokenizer", None)
            previous = getattr(backend, "encode_special_tokens", None)
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False, split_special_tokens=True)
            finally:
                if previous is not None:
                    backend.encode_special_tokens = previous
            if set(tokens) & set(getattr(self.tokenizer, "all_special_ids", [])):
                raise ValueError("Untrusted data encoded a reserved control token")
            return list(tokens)
        ids = list(encode(head))
        labels = [True] * len(ids)
        for name, text in self._documents:
            tokens = encode_data(f"\n[EVIDENCE {name}]\n{text}\n[/EVIDENCE]\n")
            if name not in self._readable_sources:
                # Fixed host capacity prevents denied length from selecting a
                # different floating-point kernel shape for public queries.
                if len(tokens) > self.DENIED_TOKEN_SLOTS:
                    raise ValueError("Denied source exceeds its fixed token-slot capacity")
                tokens += [self._identity["eos_token_ids"][0]] * (self.DENIED_TOKEN_SLOTS - len(tokens))
            ids.extend(tokens)
            labels.extend([name in self._readable_sources] * len(tokens))
        ending = encode_data("\nTask:\n" + task) + list(encode(tail))
        if not ending:
            raise ValueError("Missing public response position")
        ids.extend(ending)
        labels.extend([True] * len(ending))
        self._layout = tuple(labels)
        return torch.tensor([ids], dtype=torch.long, device=self._device)

    def _forward_generation(self, current, attention_mask, past):
        self._validate_generation_cache(past)
        start = past.get_seq_length()
        length = start + current.shape[1]
        if self._layout is None or length < len(self._layout):
            raise RuntimeError("Missing or inconsistent permission layout")
        self._layout += (True,) * (length - len(self._layout))
        mask, positions = permission_mask(self._layout, start, current.shape[1], current.device)
        calls = [0] * len(self._blocks)
        handles = []

        def guard(index):
            def check(module, args, kwargs):
                actual = kwargs.get("attention_mask")
                if (not isinstance(actual, torch.Tensor) or actual.shape != mask.shape
                        or not torch.equal(actual, mask)):
                    raise RuntimeError("Attention layer did not receive the complete permission mask")
                calls[index] += 1
            return check

        try:
            for index, block in enumerate(self._blocks):
                handles.append(block.self_attn.register_forward_pre_hook(guard(index), with_kwargs=True))
            output = self.model(input_ids=current, attention_mask=mask, position_ids=positions,
                                cache_position=torch.arange(start, length, device=current.device),
                                past_key_values=past, use_cache=True, return_dict=True)
            if any(count != 1 for count in calls):
                raise RuntimeError("Permission mask was not checked exactly once in every layer")
            return output
        finally:
            for handle in handles:
                handle.remove()

    def _cleanup_generation(self):
        self._layout = None
