"""
probe.py

Defines the PolicyDirection artifact format and ProbeBank, the runtime-side
loader. This module deliberately contains NO training code — fitting a
PolicyDirection happens in train_offline.py, which is a separate, explicit
build step (per your instruction: no retraining during inference).

Artifact format: one JSON metadata file + one .npy weight file per
(policy_name, layer_idx), so artifacts are easy to diff, version, and check
into Drive/git without pickling model or sklearn internals into the runtime
path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class PolicyDirection:
    """A single probe direction for one (policy, layer) pair.

    weight: unit-norm direction vector, shape (d_model,)
    bias: scalar offset such that raw_score = weight . h + bias
    threshold: calibrated decision threshold on raw_score (see calibration.py)
    policy_name: e.g. "unsafe_intent", "jailbreak", "privacy_leakage"
    layer_idx: which decoder layer output this direction was fit on
    metadata: calibration stats, source experiment, dataset fingerprint, etc.
    """

    weight: np.ndarray
    bias: float
    threshold: float
    policy_name: str
    layer_idx: int
    d_model: int
    metadata: dict = field(default_factory=dict)

    def score(self, hidden_state: np.ndarray) -> np.ndarray:
        """raw_score for one or more hidden vectors, shape (..., d_model) -> (...,)"""
        return hidden_state @ self.weight + self.bias

    def to_files(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.policy_name}__layer{self.layer_idx}"
        np.save(out_dir / f"{stem}.weight.npy", self.weight.astype(np.float32))
        meta = asdict(self)
        meta.pop("weight")
        with open(out_dir / f"{stem}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def from_files(cls, meta_path: Path) -> "PolicyDirection":
        with open(meta_path) as f:
            meta = json.load(f)
        stem = meta_path.name.replace(".meta.json", "")
        weight = np.load(meta_path.parent / f"{stem}.weight.npy")
        if weight.shape[0] != meta["d_model"]:
            raise ValueError(
                f"Artifact {meta_path.name}: weight dim {weight.shape[0]} != "
                f"recorded d_model {meta['d_model']}. Likely a mismatched "
                f"model/artifact pairing."
            )
        return cls(weight=weight, **meta)


class ProbeBank:
    """Loads and organizes a directory of PolicyDirection artifacts.

    Directory layout expected:
        artifacts/
          unsafe_intent__layer14.weight.npy
          unsafe_intent__layer14.meta.json
          unsafe_intent__layer18.weight.npy
          unsafe_intent__layer18.meta.json
          jailbreak__layer14.weight.npy
          ...
    """

    def __init__(self, directions: dict[str, dict[int, PolicyDirection]]):
        # policy_name -> {layer_idx -> PolicyDirection}
        self._directions = directions

    @classmethod
    def load(cls, artifact_dir: str | Path, expected_d_model: int | None = None) -> "ProbeBank":
        artifact_dir = Path(artifact_dir)
        directions: dict[str, dict[int, PolicyDirection]] = {}
        meta_files = sorted(artifact_dir.glob("*.meta.json"))
        if not meta_files:
            raise FileNotFoundError(
                f"No PolicyDirection artifacts found in {artifact_dir}. "
                f"Run train_offline.py to build them first."
            )
        for meta_path in meta_files:
            pd = PolicyDirection.from_files(meta_path)
            if expected_d_model is not None and pd.d_model != expected_d_model:
                raise ValueError(
                    f"Artifact {meta_path.name} was fit for d_model={pd.d_model}, "
                    f"but the active model has hidden_size={expected_d_model}. "
                    f"Artifacts must be re-fit per model family."
                )
            directions.setdefault(pd.policy_name, {})[pd.layer_idx] = pd
        return cls(directions)

    def policies(self) -> list[str]:
        return sorted(self._directions.keys())

    def layers_for(self, policy_name: str) -> list[int]:
        return sorted(self._directions.get(policy_name, {}).keys())

    def get(self, policy_name: str, layer_idx: int) -> PolicyDirection:
        return self._directions[policy_name][layer_idx]

    def all_layer_indices(self) -> set[int]:
        """Union of layers needed across all loaded policies — used by the
        extractor to know which layers to hook."""
        out: set[int] = set()
        for layer_map in self._directions.values():
            out.update(layer_map.keys())
        return out
