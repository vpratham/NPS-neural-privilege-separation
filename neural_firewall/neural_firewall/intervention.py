"""
intervention.py

Activation-space interventions using a learned PolicyDirection, applied via
ModelAdapter.register_pre_hooks_for_intervention. Each function has the
same signature (hidden_state, direction, strength) -> modified hidden_state
so Policy configs (policy.py) can swap between them without touching
firewall.py.

hidden_state: (batch, seq_len, d_model) or (batch, d_model)
direction: unit-norm np.ndarray or torch.Tensor, shape (d_model,)
"""

from __future__ import annotations

from typing import Callable

import torch


def _as_tensor(direction, like: torch.Tensor) -> torch.Tensor:
    if isinstance(direction, torch.Tensor):
        return direction.to(device=like.device, dtype=like.dtype)
    return torch.as_tensor(direction, device=like.device, dtype=like.dtype)


def suppress(hidden_state: torch.Tensor, direction, strength: float = 1.0) -> torch.Tensor:
    """Push activations away from the policy direction: h - strength * (h . d) * d"""
    d = _as_tensor(direction, hidden_state)
    proj_magnitude = (hidden_state * d).sum(dim=-1, keepdim=True)
    return hidden_state - strength * proj_magnitude * d


def reinforce(hidden_state: torch.Tensor, direction, strength: float = 1.0) -> torch.Tensor:
    """Push activations toward the policy direction. Mostly useful for
    ablation studies / adversarial testing of the firewall itself, not for
    safety interventions — included for parity with Phase 5's spec."""
    d = _as_tensor(direction, hidden_state)
    return hidden_state + strength * d


def projection_removal(hidden_state: torch.Tensor, direction, strength: float = 1.0) -> torch.Tensor:
    """Remove the component along `direction` entirely (strength scales how
    much of that component is removed; strength=1.0 is full removal, same
    as orthogonal_projection with alpha=1.0). Kept as a distinct entry
    point per the Exp016 projection-removal naming convention."""
    return suppress(hidden_state, direction, strength=strength)


def orthogonal_projection(hidden_state: torch.Tensor, direction, strength: float = 1.0) -> torch.Tensor:
    """x' = x - alpha * U U^T x for a single direction U (alpha=strength).
    Equivalent to suppress() for a rank-1 U but named separately so a
    future rank-k subspace version (multiple directions) can share the
    Policy API without renaming call sites."""
    return suppress(hidden_state, direction, strength=strength)


def activation_clipping(hidden_state: torch.Tensor, direction, strength: float = 1.0, max_projection: float = 0.0) -> torch.Tensor:
    """Clip the projection onto `direction` to at most `max_projection`
    (in raw dot-product units), rather than removing it entirely. `strength`
    is unused here but kept in the signature for interface uniformity with
    the other intervention functions."""
    d = _as_tensor(direction, hidden_state)
    proj_magnitude = (hidden_state * d).sum(dim=-1, keepdim=True)
    excess = (proj_magnitude - max_projection).clamp(min=0)
    return hidden_state - excess * d


INTERVENTIONS: dict[str, Callable[..., torch.Tensor]] = {
    "reinforce": reinforce,
    "suppress": suppress,
    "projection_removal": projection_removal,
    "orthogonal_projection": orthogonal_projection,
    "activation_clipping": activation_clipping,
}


def get_intervention(name: str) -> Callable[..., torch.Tensor]:
    if name not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention '{name}'. Options: {list(INTERVENTIONS)}")
    return INTERVENTIONS[name]
