"""Model-independent authorization gateway for a trusted record-copy workflow."""

from .contracts import Decision, ModelAdapter
from .runtime import run_copy
from .store import Gateway

__all__ = ["Decision", "Gateway", "ModelAdapter", "run_copy"]
