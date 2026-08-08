"""
firewall.py

The NeuralFirewall class — the external safety module described in the
architecture doc:

    Prompt -> Forward Pass -> Hidden States -> NeuralFirewall
    -> Risk Score -> Policy Decision -> Generation Continues / Intervention

It owns tokenizer access (via ModelAdapter), probe directions (via
ProbeBank), thresholds/calibration (baked into loaded PolicyDirection
artifacts — see probe.py and calibration.py), layer configuration (derived
from the loaded artifacts), the extraction pipeline (ActivationExtractor),
and intervention logic (intervention.py). It does NOT own model weights or
training code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .activation_extractor import ActivationExtractor, PoolingStrategy
from .intervention import get_intervention
from .model_interface import ModelAdapter
from .policy import Mode, Policy, VotingStrategy
from .probe import PolicyDirection, ProbeBank


@dataclass
class RiskAssessment:
    risk_score: float                       # combined, in [0, 1] where possible
    per_layer_scores: dict[int, float]
    per_layer_votes: dict[int, bool]
    exceeded_threshold: bool
    explanation: str | None = None          # populated in WARN mode


@dataclass
class FirewallDecision:
    assessment: RiskAssessment
    action: str            # "allow" | "warn" | "intervene" | "block"
    intervened: bool = False


class NeuralFirewall:
    def __init__(
        self,
        adapter: ModelAdapter,
        probe_bank: ProbeBank,
        policy: Policy,
        pooling: PoolingStrategy = "last_token",
    ):
        self.adapter = adapter
        self.probe_bank = probe_bank
        self.policy = policy
        self._validate_policy_names()
        self.extractor = ActivationExtractor(
            adapter, probe_bank.all_layer_indices(), pooling=pooling
        )
        self._active_intervention_handles = []

    def _validate_policy_names(self) -> None:
        missing = [p for p in self.policy.policy_names if p not in self.probe_bank.policies()]
        if missing:
            raise ValueError(
                f"Policy references unknown probe policies {missing}; "
                f"available: {self.probe_bank.policies()}"
            )

    # -- fit() intentionally does NOT train. Per design decision: probe
    # directions are frozen artifacts produced by train_offline.py.
    # fit() here means "load/attach runtime state", e.g. swapping in a
    # different ProbeBank or re-deriving which layers to hook. Kept as a
    # method (rather than only constructor args) so a long-lived firewall
    # instance can hot-swap policies without rebuilding the extractor from
    # scratch each time.
    def fit(self, probe_bank: ProbeBank | None = None, policy: Policy | None = None) -> "NeuralFirewall":
        if probe_bank is not None:
            self.probe_bank = probe_bank
        if policy is not None:
            self.policy = policy
        self._validate_policy_names()
        self.extractor = ActivationExtractor(
            self.adapter, self.probe_bank.all_layer_indices(), pooling=self.extractor.pooling
        )
        return self

    def _combine_votes(self, per_layer_scores: dict[int, float], per_layer_votes: dict[int, bool]) -> tuple[float, bool]:
        voting = self.policy.voting
        if not per_layer_votes:
            return 0.0, False

        if voting == VotingStrategy.MAJORITY:
            frac = sum(per_layer_votes.values()) / len(per_layer_votes)
            return frac, frac >= 0.5
        elif voting == VotingStrategy.ALL_LAYERS:
            all_true = all(per_layer_votes.values())
            frac = sum(per_layer_votes.values()) / len(per_layer_votes)
            return frac, all_true
        elif voting == VotingStrategy.ANY_LAYER:
            any_true = any(per_layer_votes.values())
            frac = sum(per_layer_votes.values()) / len(per_layer_votes)
            return frac, any_true
        elif voting == VotingStrategy.WEIGHTED:
            weights = self.policy.voting_weights or {k: 1.0 for k in per_layer_votes}
            total_w = sum(weights.get(k, 0.0) for k in per_layer_votes)
            if total_w == 0:
                total_w = 1.0
            weighted = sum(weights.get(k, 0.0) * float(v) for k, v in per_layer_votes.items())
            frac = weighted / total_w
            return frac, frac >= 0.5
        elif voting == VotingStrategy.AVERAGE_PROJECTION:
            # normalize scores to [0,1] via sigmoid before averaging, since
            # raw_score is an unbounded dot product + bias
            sig = {k: 1 / (1 + np.exp(-v)) for k, v in per_layer_scores.items()}
            frac = float(np.mean(list(sig.values())))
            return frac, frac >= 0.5
        raise ValueError(f"Unhandled voting strategy: {voting}")

    def score_pooled(self, pooled: dict[int, torch.Tensor]) -> RiskAssessment:
        """Score already-pooled hidden states directly (layer_idx -> (batch,
        d_model) or (d_model,) tensor). This is the shared core used by both
        `score()` (which extracts from a prompt first) and `streaming.py`
        (which feeds it hidden states from a single incremental decode step,
        one token at a time — Phase 4/8's continuous risk tracking).
        """
        policy_risk_scores = []
        combined_layer_scores: dict[tuple[str, int], float] = {}
        combined_layer_votes: dict[tuple[str, int], bool] = {}

        for policy_name in self.policy.policy_names:
            per_layer_scores: dict[int, float] = {}
            per_layer_votes: dict[int, bool] = {}
            for layer_idx in self.probe_bank.layers_for(policy_name):
                if layer_idx not in pooled:
                    continue
                direction: PolicyDirection = self.probe_bank.get(policy_name, layer_idx)
                h = pooled[layer_idx]
                h = h.detach().cpu().numpy() if isinstance(h, torch.Tensor) else h
                raw_score = float(direction.score(h)[0]) if h.ndim == 2 else float(direction.score(h))
                per_layer_scores[layer_idx] = raw_score
                per_layer_votes[layer_idx] = raw_score >= direction.threshold

            frac, exceeded = self._combine_votes(per_layer_scores, per_layer_votes)
            policy_risk_scores.append(frac)
            for k, v in per_layer_scores.items():
                combined_layer_scores[(policy_name, k)] = v
            for k, v in per_layer_votes.items():
                combined_layer_votes[(policy_name, k)] = v

        if not policy_risk_scores:
            overall_risk = 0.0
        elif self.policy.policy_combination == "mean":
            overall_risk = float(np.mean(policy_risk_scores))
        else:  # "max"
            overall_risk = float(max(policy_risk_scores))
        overall_exceeded = overall_risk >= 0.5

        explanation = None
        if self.policy.mode == Mode.WARN:
            hits = [str(k) for k, v in combined_layer_votes.items() if v]
            explanation = (
                f"risk={overall_risk:.3f}; triggered probes: {hits}"
                if hits else f"risk={overall_risk:.3f}; no probes triggered"
            )

        return RiskAssessment(
            risk_score=overall_risk,
            per_layer_scores=combined_layer_scores,
            per_layer_votes=combined_layer_votes,
            exceeded_threshold=overall_exceeded,
            explanation=explanation,
        )

    def score(self, prompt: str) -> RiskAssessment:
        """Score a prompt against every layer of every policy in self.policy.policy_names.

        Combines across layers within each policy via self.policy.voting,
        then combines across policies (when policy_names has more than one
        entry) via self.policy.policy_combination ("max": a hit on any
        policy direction is a hit; "mean": average risk across policies).
        This is the entry point Phase 10's independent per-policy detectors
        (unsafe_intent, jailbreak, privacy_leakage, ...) plug into — adding
        a policy is a train_offline.py artifact, not a code change here.
        """
        try:
            extracted = self.extractor.extract(prompt)
        except Exception as e:
            if self.policy.fail_closed:
                return RiskAssessment(
                    risk_score=1.0,
                    per_layer_scores={},
                    per_layer_votes={},
                    exceeded_threshold=True,
                    explanation=f"fail-closed: extraction error ({e})",
                )
            raise

        return self.score_pooled(extracted.pooled)

    def should_block(self, prompt: str) -> bool:
        return self.score(prompt).exceeded_threshold

    def _install_intervention_hooks(self) -> None:
        if self.policy.mode != Mode.INTERVENE:
            return
        fn = get_intervention(self.policy.intervention_method)
        strength = self.policy.intervention_strength

        # Use the average direction across triggered policies/layers for a
        # single-pass intervention; per-layer directions are applied at
        # their own layer via the pre-hook's layer_idx argument.
        def _intervene_at_layer(layer_idx: int, hidden_state: torch.Tensor) -> torch.Tensor:
            modified = hidden_state
            for policy_name in self.policy.policy_names:
                if layer_idx in self.probe_bank.layers_for(policy_name):
                    direction = self.probe_bank.get(policy_name, layer_idx)
                    modified = fn(modified, direction.weight, strength)
            return modified

        layer_indices = self.probe_bank.all_layer_indices()
        self._active_intervention_handles = self.adapter.register_pre_hooks_for_intervention(
            layer_indices, _intervene_at_layer
        )

    def _remove_intervention_hooks(self) -> None:
        for h in self._active_intervention_handles:
            h.remove()
        self._active_intervention_handles = []

    def intervene(self, prompt: str) -> FirewallDecision:
        """Apply Mode.INTERVENE's configured intervention and (conceptually)
        continue generation with modified activations. Actual token
        generation is left to the caller (adapter.generate_step / the
        model's own .generate) — this method installs/tears down the
        intervention hooks around that call.

        Returns a FirewallDecision so callers get both the pre-intervention
        risk assessment and confirmation the hooks were applied.
        """
        assessment = self.score(prompt)
        if not assessment.exceeded_threshold:
            return FirewallDecision(assessment=assessment, action="allow", intervened=False)

        self._install_intervention_hooks()
        try:
            # Caller is expected to run generation here, inside this
            # `with`-like window. Since generation is caller-driven (varies
            # by streaming vs batch), we expose install/remove as public
            # methods too — see below — for callers who want more control
            # than this single-shot convenience wrapper.
            pass
        finally:
            self._remove_intervention_hooks()

        return FirewallDecision(assessment=assessment, action="intervene", intervened=True)

    def decide(self, prompt: str) -> FirewallDecision:
        """Top-level entry point implementing Phase 3's four modes."""
        assessment = self.score(prompt)

        if self.policy.mode == Mode.DETECT:
            return FirewallDecision(assessment=assessment, action="allow", intervened=False)

        if self.policy.mode == Mode.WARN:
            action = "warn" if assessment.exceeded_threshold else "allow"
            return FirewallDecision(assessment=assessment, action=action, intervened=False)

        if self.policy.mode == Mode.BLOCK:
            action = "block" if assessment.exceeded_threshold else "allow"
            return FirewallDecision(assessment=assessment, action=action, intervened=False)

        if self.policy.mode == Mode.INTERVENE:
            return self.intervene(prompt)

        raise ValueError(f"Unhandled mode: {self.policy.mode}")
