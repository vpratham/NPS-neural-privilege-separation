"""
policy.py

Swappable policy configuration, per Phase 7. A Policy binds together which
PolicyDirection(s) to use, how to combine multi-layer/multi-probe votes,
the operating mode, and (if applicable) the intervention method/strength.
Changing a Policy should never require touching firewall.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Mode(Enum):
    DETECT = "detect"        # Phase 3, Mode 1: return risk score only
    WARN = "warn"             # Mode 2: risk score + explanation
    INTERVENE = "intervene"   # Mode 3: modify activations before generation
    BLOCK = "block"           # Mode 4: terminate generation


class VotingStrategy(Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    AVERAGE_PROJECTION = "average_projection"
    ANY_LAYER = "any_layer"        # single-layer or "fail on any" ensembles
    ALL_LAYERS = "all_layers"


@dataclass
class Policy:
    """One configurable operating policy.

    policy_names: which PolicyDirection policies to evaluate (e.g.
        ["unsafe_intent"] or ["unsafe_intent", "jailbreak"])
    voting: how to combine per-layer (and, later, per-policy) scores
    voting_weights: only used when voting == WEIGHTED; layer_idx -> weight
    mode: what the firewall does with the resulting decision
    intervention_method: name from intervention.INTERVENTIONS, required if mode == INTERVENE
    intervention_strength: passed through to the intervention function
    fail_closed: if extraction/scoring errors, block (True) or pass-through (False)
    policy_combination: how to combine risk across multiple policy_names ("max" —
        a hit on any policy is a hit — or "mean")
    """

    policy_names: list[str]
    voting: VotingStrategy = VotingStrategy.MAJORITY
    voting_weights: dict[int, float] = field(default_factory=dict)
    mode: Mode = Mode.DETECT
    intervention_method: str | None = None
    intervention_strength: float = 1.0
    fail_closed: bool = True
    policy_combination: str = "max"

    def __post_init__(self):
        if self.mode == Mode.INTERVENE and self.intervention_method is None:
            raise ValueError("Mode.INTERVENE requires intervention_method to be set.")
        if self.policy_combination not in ("max", "mean"):
            raise ValueError(f"policy_combination must be 'max' or 'mean', got {self.policy_combination!r}")
