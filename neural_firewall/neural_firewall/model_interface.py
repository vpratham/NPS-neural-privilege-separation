"""
model_interface.py

Generic adapter boundary between the firewall and any HF-style causal LM.

The firewall (and everything downstream of it: activation_extractor,
intervention, calibration) is written against `ModelAdapter` only. It never
imports a model family by name and never assumes a specific attribute path
(`model.model.layers`, `model.transformer.h`, etc). Adding Llama, Gemma,
Mistral, or anything else later means writing one new adapter class here —
nothing else in the package changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn as nn


@dataclass
class HiddenStateBundle:
    """Hidden states for one forward pass, one layer per entry.

    hidden_states[i] has shape (batch, seq_len, d_model) and corresponds to
    the OUTPUT of decoder layer `layer_indices[i]` (0-indexed, matching
    however the adapter chooses to number layers — the firewall treats
    these indices as opaque keys, not architecture facts).
    """

    hidden_states: dict[int, torch.Tensor]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor | None = None


class ModelAdapter(ABC):
    """Boundary the firewall depends on. Implement this once per model family."""

    @abstractmethod
    def tokenize(self, prompt: str, **kwargs) -> dict[str, torch.Tensor]:
        """Return a dict of tensors suitable for `self.model(**out)`."""

    @abstractmethod
    def num_layers(self) -> int:
        """Number of decoder layers available for hooking."""

    @abstractmethod
    def get_decoder_layer(self, layer_idx: int) -> nn.Module:
        """Return the nn.Module for decoder layer `layer_idx`.

        This is the single method that encodes model-family-specific
        attribute traversal (e.g. `model.model.layers[i]` for Qwen/Llama-
        style models, `model.transformer.h[i]` for GPT-2-style models).
        Everything else in the codebase is written against this method,
        never against the raw attribute path.
        """

    @abstractmethod
    def hidden_size(self) -> int:
        """d_model, used to validate PolicyDirection shapes at load time."""

    @abstractmethod
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def generate_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values=None,
    ):
        """One incremental decoding step for streaming mode.

        Returns (logits, new_past_key_values, hidden_states_by_layer) where
        hidden_states_by_layer is a dict[int, Tensor] of shape (batch, 1,
        d_model) for the newly generated position, keyed the same way as
        `get_decoder_layer` / `num_layers`.
        """

    def register_output_hooks(
        self,
        layer_indices: Sequence[int],
        callback: Callable[[int, torch.Tensor], None],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Default hook registration shared by all adapters.

        callback(layer_idx, output_hidden_state) is invoked on every forward
        pass. Subclasses generally do not need to override this — it's
        implemented once here against `get_decoder_layer`, which subclasses
        DO need to implement correctly.
        """
        handles = []
        for idx in layer_indices:
            layer = self.get_decoder_layer(idx)

            def _hook(module, inputs, output, _idx=idx):
                # Decoder layers commonly return a tuple whose first element
                # is the hidden state; guard both cases defensively.
                hs = output[0] if isinstance(output, tuple) else output
                callback(_idx, hs)

            handles.append(layer.register_forward_hook(_hook))
        return handles

    def register_input_hooks(
        self,
        layer_indices: Sequence[int],
        callback: Callable[[int, torch.Tensor], None],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Register forward-pre-hooks that capture decoder-layer inputs.

        callback(layer_idx, input_hidden_state) is invoked before each
        selected decoder layer runs. The captured tensor is the layer INPUT,
        matching the Exp017/18/19 extraction convention.
        """
        handles = []

        for idx in layer_indices:
            layer = self.get_decoder_layer(idx)

            def _hook(module, inputs, _idx=idx):
                hs = inputs[0]
                callback(_idx, hs)

            handles.append(layer.register_forward_pre_hook(_hook))

        return handles

    def register_pre_hooks_for_intervention(
        self,
        layer_indices: Sequence[int],
        intervene_fn: Callable[[int, torch.Tensor], torch.Tensor],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        """Register forward hooks that can MODIFY hidden states in place.

        intervene_fn(layer_idx, hidden_state) -> modified_hidden_state.
        Used by Mode 3 (intervene). Returns removable handles so callers can
        tear down the intervention after a single generation call.
        """
        handles = []
        for idx in layer_indices:
            layer = self.get_decoder_layer(idx)

            def _hook(module, inputs, output, _idx=idx):
                if isinstance(output, tuple):
                    modified = intervene_fn(_idx, output[0])
                    return (modified,) + tuple(output[1:])
                return intervene_fn(_idx, output)

            handles.append(layer.register_forward_hook(_hook))
        return handles


class HFCausalLMAdapter(ModelAdapter):
    """Adapter for standard decoder-only HF models with a `.layers` list.

    Covers Qwen2(.5), Llama, Mistral out of the box via `layer_path` —
    e.g. "model.layers" (Qwen2.5, Llama, Mistral) or "transformer.h"
    (GPT-2 family). Gemma also follows the "model.layers" convention.
    This is the ONE place that needs a new subclass (or a new `layer_path`)
    for an architecture with a genuinely different structure.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        layer_path: str = "model.layers",
        max_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layer_path = layer_path
        self.max_length = max_length
        self._layers = self._resolve_layers(layer_path)

    def _resolve_layers(self, layer_path: str) -> nn.ModuleList:
        obj = self.model
        for attr in layer_path.split("."):
            obj = getattr(obj, attr)
        return obj

    def tokenize(self, prompt, **kwargs) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=kwargs.get("max_length", self.max_length),
        )
        return {k: v.to(self.device()) for k, v in enc.items()}

    def num_layers(self) -> int:
        return len(self._layers)

    def get_decoder_layer(self, layer_idx: int) -> nn.Module:
        return self._layers[layer_idx]

    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate_step(self, input_ids, attention_mask=None, past_key_values=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
        )
        # out.hidden_states: tuple(len = num_layers + 1), index 0 is the
        # embedding output, index i+1 is the output of decoder layer i.
        hs_by_layer = {i: out.hidden_states[i + 1] for i in range(self.num_layers())}
        return out.logits, out.past_key_values, hs_by_layer


def build_qwen_adapter(model_name: str = "Qwen/Qwen2.5-3B-Instruct", **kwargs) -> HFCausalLMAdapter:
    """Convenience constructor for the validation target.

    This is a thin factory, not a special code path — it returns a plain
    HFCausalLMAdapter with layer_path="model.layers", which is why Llama /
    Mistral / Gemma work through the same adapter class unchanged.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **kwargs,
    )
    
    model.eval()
    return HFCausalLMAdapter(model, tokenizer, layer_path="model.layers")
