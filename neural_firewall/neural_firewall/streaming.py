"""
streaming.py

Phase 4 + Phase 8: continuous, per-token risk tracking during generation,
with the firewall sitting inside the decode loop rather than scoring only
a single pre-generation hidden state.

    hidden state (token t) -> firewall.score_pooled() -> risk(t) -> continue / intervene / block
                                                              |
                                                        next token (t+1)

This is intentionally a plain Python generation loop built on
`ModelAdapter.generate_step`, not a wrapper around `model.generate()` —
Phase 8 explicitly calls for the firewall to be capable of operating
*during* generation, which requires stepping token-by-token so a BLOCK
decision can halt before the next forward pass and an INTERVENE policy's
hooks are already live on the decoder layers for every step.

For Mode.INTERVENE, callers should install the firewall's intervention
hooks (`NeuralFirewall._install_intervention_hooks` — exposed here as
`install_intervention` / `remove_intervention` for the streaming case)
before starting the loop, so every decode step benefits from the same
suppression rather than a single one-shot application.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .firewall import NeuralFirewall, RiskAssessment
from .policy import Mode


@dataclass
class TokenRiskRecord:
    position: int
    token_id: int
    token_text: str
    assessment: RiskAssessment


@dataclass
class StreamingResult:
    output_ids: list[int]
    output_text: str
    trajectory: list[TokenRiskRecord] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str | None = None

    def risk_series(self) -> tuple[list[int], list[float]]:
        """Convenience accessor for visualization.plot_risk_timeline."""
        positions = [r.position for r in self.trajectory]
        risks = [r.assessment.risk_score for r in self.trajectory]
        return positions, risks


class StreamingFirewall:
    """Wraps a NeuralFirewall + its ModelAdapter to run token-by-token
    generation with per-step risk scoring, honoring the firewall's
    configured Mode (detect/warn/intervene/block) at every step rather
    than only on the initial prompt.
    """

    def __init__(self, firewall: NeuralFirewall):
        self.firewall = firewall
        self.adapter = firewall.adapter

    def _pool_step_hidden_states(self, hs_by_layer: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        # Each tensor is (batch, 1, d_model) for a single incremental step —
        # squeeze the sequence dim so score_pooled sees (batch, d_model),
        # matching what ActivationExtractor produces for a full prompt.
        return {idx: hs[:, -1, :] for idx, hs in hs_by_layer.items()}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
        temperature: float = 0.0,
        install_intervention_hooks: bool = True,
    ) -> StreamingResult:
        """Run the decode loop with per-token firewall checks.

        Mode.DETECT / Mode.WARN: never stop generation; risk is recorded
            per token (WARN also populates `assessment.explanation`).
        Mode.BLOCK: generation halts the moment a token's risk exceeds
            threshold; that token is NOT included in output_ids/output_text.
        Mode.INTERVENE: intervention hooks (from `firewall.policy.
            intervention_method`) are installed for the whole loop (if
            `install_intervention_hooks=True`), so every step's forward
            pass already has suppressed/projected activations; risk is
            still recorded post-intervention so you can see how much the
            intervention reduced it (Phase 9's intervention-strength plots).
        """
        tokenizer = getattr(self.adapter, "tokenizer", None)
        enc = self.adapter.tokenize(prompt)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        hooks_installed = False
        if self.firewall.policy.mode == Mode.INTERVENE and install_intervention_hooks:
            self.firewall._install_intervention_hooks()
            hooks_installed = True

        trajectory: list[TokenRiskRecord] = []
        generated_ids: list[int] = []
        past_key_values = None
        stopped_early = False
        stop_reason = None

        try:
            step_input_ids = input_ids
            for position in range(max_new_tokens):
                logits, past_key_values, hs_by_layer = self.adapter.generate_step(
                    step_input_ids, attention_mask, past_key_values
                )

                pooled = self._pool_step_hidden_states(hs_by_layer)
                assessment = self.firewall.score_pooled(pooled)

                next_token_logits = logits[:, -1, :]
                if temperature and temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token_id = int(torch.multinomial(probs, num_samples=1)[0, 0])
                else:
                    next_token_id = int(torch.argmax(next_token_logits, dim=-1)[0])

                token_text = (
                    tokenizer.decode([next_token_id]) if tokenizer is not None else str(next_token_id)
                )

                if self.firewall.policy.mode == Mode.BLOCK and assessment.exceeded_threshold:
                    trajectory.append(TokenRiskRecord(position, next_token_id, token_text, assessment))
                    stopped_early = True
                    stop_reason = f"blocked at position {position}: risk={assessment.risk_score:.3f}"
                    break

                trajectory.append(TokenRiskRecord(position, next_token_id, token_text, assessment))
                generated_ids.append(next_token_id)

                if eos_token_id is not None and next_token_id == eos_token_id:
                    stop_reason = "eos"
                    break

                step_input_ids = torch.tensor([[next_token_id]], device=self.adapter.device())
                if attention_mask is not None:
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)],
                        dim=1,
                    )
        finally:
            if hooks_installed:
                self.firewall._remove_intervention_hooks()

        output_text = (
            tokenizer.decode(generated_ids) if tokenizer is not None and generated_ids else ""
        )

        return StreamingResult(
            output_ids=generated_ids,
            output_text=output_text,
            trajectory=trajectory,
            stopped_early=stopped_early,
            stop_reason=stop_reason,
        )
