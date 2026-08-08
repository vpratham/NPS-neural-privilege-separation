"""
cache.py

Generic manifest-based checkpointing, in the same spirit as the per-stage
JSON manifest + incremental-save pattern used across Exp009-018. This is a
fresh implementation (the actual notebook wasn't available to port from) —
swap in your existing Drive-caching / manifest code here if its interface
differs; callers only depend on `Manifest.is_done`, `.mark_done`, and
`.save`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    """Tracks which (stage_key) units of work are complete, so long-running
    jobs (offline training over many layers/policies, batch evaluation over
    many prompts) can resume after interruption instead of restarting.
    """

    path: Path
    completed: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load_or_create(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        if path.exists():
            with open(path) as f:
                completed = json.load(f)
        else:
            completed = {}
        return cls(path=path, completed=completed)

    def is_done(self, stage_key: str) -> bool:
        return stage_key in self.completed

    def mark_done(self, stage_key: str, info: dict | None = None) -> None:
        self.completed[stage_key] = info or True
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.completed, f, indent=2)

    def reset(self) -> None:
        self.completed = {}
        self.save()
