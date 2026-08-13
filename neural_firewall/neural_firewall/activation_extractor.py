"""
activation_extractor.py

Thin, model-agnostic extraction layer on top of ModelAdapter. Given a set of
layer indices (typically ProbeBank.all_layer_indices()), runs a forward pass
and returns pooled hidden states ready for probe.score().

NOTE ON REUSE: your prior notebooks (Exp015-18) already have a validated
extraction pipeline (dataset-fingerprint caching, batched extraction, last-
token vs mean-pooling choices, etc). This class is the seam where that code
should slot in — replace `_pool` / `extract` bodies with your existing
logic once the notebook is available; the call signature is designed to
stay stable so firewall.py doesn't need to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .model_interface import ModelAdapter

PoolingStrategy = Literal["last_token", "mean"]


@dataclass
class ExtractedActivations:
    # layer_idx -> (batch, d_model) pooled hidden state
    pooled: dict[int, torch.Tensor]


class ActivationExtractor:
    def __init__(
        self,
        adapter: ModelAdapter,
        layer_indices: set[int],
        pooling: PoolingStrategy = "last_token",
    ):
        self.adapter = adapter
        self.layer_indices = sorted(layer_indices)
        self.pooling = pooling

    def _pool(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        # hidden_state: (batch, seq_len, d_model)
        if self.pooling == "last_token":
            if attention_mask is not None:
                last_idx = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(hidden_state.size(0), device=hidden_state.device)
                return hidden_state[batch_idx, last_idx]
            return hidden_state[:, -1, :]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
                summed = (hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                return summed / counts
            return hidden_state.mean(dim=1)
        raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    @torch.no_grad()
    def extract(self, prompt: str) -> ExtractedActivations:
        """Single-prompt extraction via forward hooks (pre-generation scoring)."""
        captured: dict[int, torch.Tensor] = {}

        def _callback(layer_idx: int, hs: torch.Tensor) -> None:
            captured[layer_idx] = hs

        handles = self.adapter.register_input_hooks(self.layer_indices, _callback)
        try:
            enc = self.adapter.tokenize(prompt)
            # A plain forward pass is enough to populate hidden states via
            # the hooks — no generation needed for pre-generation scoring.
            self.adapter.model(**enc) if hasattr(self.adapter, "model") else None
        finally:
            for h in handles:
                h.remove()

        pooled = {
            idx: self._pool(hs, enc.get("attention_mask"))
            for idx, hs in captured.items()
        }
        return ExtractedActivations(pooled=pooled)

    @torch.no_grad()
    def extract_batch(self, prompts: list[str]) -> ExtractedActivations:
        """Batched extraction matching the Exp019/19 convention.

        Captures decoder-layer INPUTS, then pools the last valid token using
        attention_mask.sum(dim=1) - 1.

        Returns:
            pooled[layer_idx]: Tensor of shape (batch, d_model)
        """
        if not prompts:
            raise ValueError("prompts must not be empty")

        captured: dict[int, torch.Tensor] = {}

        def _callback(layer_idx: int, hs: torch.Tensor) -> None:
            captured[layer_idx] = hs.detach()

        handles = self.adapter.register_input_hooks(
            self.layer_indices,
            _callback,
        )

        try:
            enc = self.adapter.tokenize(prompts,max_length=512)

            if not hasattr(self.adapter, "model"):
                raise RuntimeError(
                    "Batched extraction requires an adapter exposing .model"
                )

            self.adapter.model(**enc)

        finally:
            for h in handles:
                h.remove()

        pooled = {
            idx: self._pool(
                hs,
                enc.get("attention_mask"),
            )
            for idx, hs in captured.items()
        }

        return ExtractedActivations(pooled=pooled)