"""Isolated policy KV memory for batch-one Qwen inference.

This protects memory ownership, not semantic instruction following. Ordinary
causal attention already makes prefix representations independent of suffixes.
The host and model runtime are trusted; Python tensors are not a process sandbox.
"""
from __future__ import annotations

import hashlib

import torch
import transformers
from transformers.cache_utils import Cache, DynamicLayer

from .hf_adapter import HFAdapter


def _check_pair(keys, values):
    if (not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor)
            or keys.ndim != 4 or values.shape != keys.shape or keys.shape[0] != 1
            or any(size < 1 for size in keys.shape)
            or keys.dtype != torch.float32 or values.dtype != keys.dtype
            or values.device != keys.device
            or not torch.isfinite(keys).all() or not torch.isfinite(values).all()):
        raise ValueError("Policy memory requires finite batch-one float32 KV tensors")


class _PolicyLayer(DynamicLayer):
    def __init__(self, keys, values):
        super().__init__()
        _check_pair(keys, values)
        self._policy = (keys.detach().clone(), values.detach().clone())
        self.keys = keys[..., :0, :].detach().clone()
        self.values = values[..., :0, :].detach().clone()
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True
        self._seal = self.policy_digest()

    def policy_digest(self):
        digest = hashlib.sha256()
        for tensor in self._policy:
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
        return digest.hexdigest()

    def verify(self):
        if self.policy_digest() != self._seal:
            raise RuntimeError("Protected policy memory changed")

    def get_seq_length(self):
        return self._policy[0].shape[-2] + self.keys.shape[-2]

    def update(self, key_states, value_states, cache_kwargs=None):
        self.verify()
        _check_pair(key_states, value_states)
        reference = self._policy[0]
        if (key_states.shape[:2] != reference.shape[:2]
                or key_states.shape[-1] != reference.shape[-1]
                or key_states.device != reference.device):
            raise ValueError("Working KV layout differs from protected policy")
        positions = (cache_kwargs or {}).get("cache_position")
        expected = torch.arange(self.get_seq_length(),
                                self.get_seq_length() + key_states.shape[-2],
                                device=key_states.device)
        if (not isinstance(positions, torch.Tensor) or positions.dtype != torch.long
                or not torch.equal(positions, expected)):
            raise ValueError("Policy cache accepts only contiguous append positions")
        # cat allocates: incoming KV and the returned attention view never alias
        # either the policy storage or the stored writable tail.
        self.keys = torch.cat((self.keys, key_states.detach()), dim=-2)
        self.values = torch.cat((self.values, value_states.detach()), dim=-2)
        return (torch.cat((self._policy[0], self.keys), dim=-2),
                torch.cat((self._policy[1], self.values), dim=-2))

    def _unsupported(self, *args, **kwargs):
        raise RuntimeError("Policy cache supports append-only greedy inference")

    # Do not inherit cache mutations that assume policy and tail share storage.
    lazy_initialization = _unsupported
    reset = _unsupported
    crop = _unsupported
    reorder_cache = _unsupported
    batch_repeat_interleave = _unsupported
    batch_select_indices = _unsupported
    offload = _unsupported
    prefetch = _unsupported


class ProtectedPolicyCache(Cache):
    """Owns a sealed prefix plus a separately allocated tail for each layer."""
    def __init__(self, policy_cache):
        source = getattr(policy_cache, "layers", None)
        if not source or any(getattr(layer, "is_sliding", True) for layer in source):
            raise ValueError("A nonempty full-attention policy cache is required")
        layers = [_PolicyLayer(layer.keys, layer.values) for layer in source]
        if len({layer.get_seq_length() for layer in layers}) != 1:
            raise ValueError("Policy cache layers have inconsistent lengths")
        super().__init__(layers=layers)
        self._layer_count = len(layers)
        self._seals = tuple(layer.policy_digest() for layer in layers)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if type(layer_idx) is not int or not 0 <= layer_idx < self._layer_count:
            raise ValueError("Invalid policy-cache layer index")
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def verify(self):
        # ponytail: hashes copy KV to CPU each step; optimize only after measuring
        # overhead, while retaining equivalent mutation-detection coverage.
        if (len(self.layers) != self._layer_count
                or tuple(layer.policy_digest() for layer in self.layers) != self._seals):
            raise RuntimeError("Protected policy memory changed")


class PolicyMemoryAdapter(HFAdapter):
    """Uses the existing decoder and gate with physically separate policy KV.

    Expected to preserve the ordinary model's outputs. This is a structural
    foundation for future read permissions, not a prompt-injection defense.
    """
    DECODER_VERSION = "greedy_cached_isolated_policy_v1"

    def _prepare_generation(self, input_ids):
        if transformers.__version__ != "4.57.6":
            raise RuntimeError("Policy memory currently supports Transformers 4.57.6 only")
        if getattr(self.model.model, "has_sliding_layers", False):
            raise ValueError("Policy memory requires full attention at every layer")
        prefix = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.policy}],
            tokenize=True, add_generation_prompt=False, return_tensors="pt",
        )
        if isinstance(prefix, dict):
            prefix = prefix.get("input_ids")
        if (not isinstance(prefix, torch.Tensor) or prefix.ndim != 2
                or prefix.shape[0] != 1 or not 0 < prefix.shape[1] < input_ids.shape[1]):
            raise ValueError("Tokenizer must provide a nonempty separate system prefix")
        prefix = prefix.to(input_ids.device)
        count = prefix.shape[1]
        if not torch.equal(prefix, input_ids[:, :count]):
            raise ValueError("System-only encoding is not an exact request prefix")
        with torch.no_grad():
            policy_output = self.model(input_ids=prefix, attention_mask=torch.ones_like(prefix),
                                       use_cache=True, return_dict=True)
        cache = ProtectedPolicyCache(policy_output.past_key_values)
        if len(cache.layers) != len(self._blocks):
            raise RuntimeError("Policy cache does not cover every decoder layer")
        cache.verify()
        return input_ids[:, count:], cache

    def _validate_generation_cache(self, cache):
        if not isinstance(cache, ProtectedPolicyCache):
            raise RuntimeError("Decoder replaced the protected policy cache")
        cache.verify()


def compare_policy_memory(adapter, requests, max_new_tokens):
    """Paired development smoke check; no output-policy or attack labels."""
    from .artifacts import digest
    from .runtime import policy_digest

    if not isinstance(adapter, PolicyMemoryAdapter) or not requests:
        raise ValueError("An isolated-memory adapter and nonempty requests are required")
    ordinary = HFAdapter.from_components(
        adapter.model, adapter.tokenizer, policy=adapter.policy, layers=adapter.layers,
        projection_dim=adapter.projection_dim, seed=adapter.seed, max_context=adapter.max_context,
    )
    rows = []
    for request in requests:
        task, context = request["task"], request.get("context", "")
        before = list(ordinary.iter_steps(task, context, max_new_tokens))
        after = list(adapter.iter_steps(task, context, max_new_tokens))
        same_tokens = [s.token_id for s in before] == [s.token_id for s in after]
        a = torch.tensor([s.features for s in before])
        b = torch.tensor([s.features for s in after])
        same_shape = a.shape == b.shape
        rows.append({
            "request_sha256": digest({"task": task, "context": context}),
            "ordinary_tokens": len(before), "isolated_tokens": len(after),
            "ordinary_ended_with_eos": before[-1].is_eos,
            "isolated_ended_with_eos": after[-1].is_eos,
            "identical_tokens": same_tokens,
            "features_close": same_shape and torch.allclose(a, b, atol=1e-5, rtol=1e-5),
            "max_feature_difference": float((a - b).abs().max()) if same_shape else None,
        })
    return {
        "kind": "policy-memory-equivalence-v1",
        "claim_scope": "memory isolation and decoder comparison only; no injection-resistance claim",
        "identity": adapter.identity, "ordinary_identity": ordinary.identity,
        "same_model_object": adapter.model is ordinary.model,
        "policy_sha256": policy_digest(adapter.policy), "max_new_tokens": max_new_tokens,
        "feature_atol": 1e-5, "feature_rtol": 1e-5, "rows": rows,
        "all_match": all(row["identical_tokens"] and row["features_close"] for row in rows),
    }
