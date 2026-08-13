"""
train_offline.py

The ONLY place a PolicyDirection gets fit. Run this as an explicit build/
update step (script or notebook cell) whenever you want to add a policy,
retrain on new data, or re-calibrate thresholds. The runtime firewall never
imports this module's training path — only the artifacts it writes.

This is the seam for your existing Exp017/18 probe-training code
(per-layer logistic-regression directions, dataset-fingerprint caching,
stratified train/calibration/held-out-test/adversarial splits). Port that
logic into `fit_policy_direction` / `run_training_job`; the CLI/manifest
scaffolding around it is designed to stay stable.

Usage:
    python -m neural_firewall.train_offline \\
        --activations activations.npz \\
        --labels labels.npy \\
        --policy-name unsafe_intent \\
        --out-dir artifacts/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .cache import Manifest
from .calibration import calibrate_threshold
from .probe import PolicyDirection


def fit_policy_direction(
    activations: np.ndarray,      # (n_examples, d_model), pooled hidden states for one layer
    labels: np.ndarray,           # (n_examples,), 1 = policy-triggering, 0 = benign
    policy_name: str,
    layer_idx: int,
    calibration_labels: np.ndarray | None = None,
    calibration_activations: np.ndarray | None = None,
    calibration_method: str = "target_fpr",
    calibration_target_fpr: float = 0.02,
    random_state: int = 42,
) -> PolicyDirection:
    """Fit one logistic-regression probe direction and calibrate its threshold.

    Uses a held-out calibration split when provided (recommended — this is
    what Exp017/18 do to avoid overfit thresholds); falls back to the
    training split itself otherwise, with a warning baked into metadata.
    """
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        random_state=random_state,
    )
    clf.fit(activations, labels)

    weight = clf.coef_[0]
    norm = np.linalg.norm(weight)
    unit_weight = weight / norm
    # rescale bias so raw_score = unit_weight . h + bias matches the
    # original decision function w.h + b at the same points
    bias = float(clf.intercept_[0]) / norm

    # Artifacts are persisted as float32 (probe.py PolicyDirection.to_files),
    # so calibrate the threshold using the SAME precision the runtime firewall
    # will score with. Calibrating in float64 and scoring in float32 lets
    # borderline points land on the wrong side of their own threshold —
    # rare in practice, but a real correctness gap, not just test noise.
    unit_weight = unit_weight.astype(np.float32)
    bias = float(np.float32(bias))  # snap to the float32 value, keep it JSON-serializable

    used_fallback_calibration = calibration_activations is None
    cal_acts = calibration_activations if calibration_activations is not None else activations
    cal_labels = calibration_labels if calibration_labels is not None else labels

    raw_scores = (cal_acts.astype(np.float32) @ unit_weight + bias).astype(np.float64)
    cal_result = calibrate_threshold(
        raw_scores, cal_labels, method=calibration_method, target_fpr=calibration_target_fpr
    )

    return PolicyDirection(
        weight=unit_weight.astype(np.float32),
        bias=bias,
        threshold=cal_result.threshold,
        policy_name=policy_name,
        layer_idx=layer_idx,
        d_model=activations.shape[1],
        metadata={
            "calibration_method": cal_result.method,
            "calibration_achieved_fpr": cal_result.achieved_fpr,
            "calibration_achieved_recall": cal_result.achieved_recall,
            "used_fallback_calibration_split": used_fallback_calibration,
            "n_train": int(len(labels)),
            "train_class_balance": float(np.mean(labels)),
        },
    )


def run_training_job(
    activations_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    policy_name: str,
    out_dir: str | Path,
    calibration_activations_by_layer: dict[int, np.ndarray] | None = None,
    calibration_labels: np.ndarray | None = None,
    manifest_path: str | Path | None = None,
) -> list[PolicyDirection]:
    """Fit one PolicyDirection per layer, with manifest-based resumability
    so an interrupted multi-layer run (e.g. many layers on a large model)
    can pick back up without re-fitting completed layers."""
    out_dir = Path(out_dir)
    manifest = Manifest.load_or_create(manifest_path or (out_dir / "_manifest.json"))

    results = []
    for layer_idx, acts in activations_by_layer.items():
        stage_key = f"{policy_name}__layer{layer_idx}"
        if manifest.is_done(stage_key):
            results.append(PolicyDirection.from_files(out_dir / f"{stage_key}.meta.json"))
            continue

        cal_acts = (
            calibration_activations_by_layer.get(layer_idx)
            if calibration_activations_by_layer
            else None
        )
        direction = fit_policy_direction(
            acts, labels, policy_name, layer_idx,
            calibration_activations=cal_acts,
            calibration_labels=calibration_labels,
        )
        direction.to_files(out_dir)
        manifest.mark_done(stage_key, {"threshold": direction.threshold})
        results.append(direction)

    return results


def _cli():
    parser = argparse.ArgumentParser(description="Fit PolicyDirection artifacts offline.")
    parser.add_argument("--activations", required=True, help="npz file: layer_idx -> (n, d_model) array")
    parser.add_argument("--labels", required=True, help=".npy file of 0/1 labels")
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--calibration-activations", default=None)
    parser.add_argument("--calibration-labels", default=None)
    args = parser.parse_args()

    npz = np.load(args.activations)
    activations_by_layer = {int(k): npz[k] for k in npz.files}
    labels = np.load(args.labels)

    cal_by_layer = None
    cal_labels = None
    if args.calibration_activations:
        cal_npz = np.load(args.calibration_activations)
        cal_by_layer = {int(k): cal_npz[k] for k in cal_npz.files}
        cal_labels = np.load(args.calibration_labels)

    results = run_training_job(
        activations_by_layer, labels, args.policy_name, args.out_dir,
        calibration_activations_by_layer=cal_by_layer,
        calibration_labels=cal_labels,
    )
    print(f"Fit {len(results)} PolicyDirection artifacts -> {args.out_dir}")


if __name__ == "__main__":
    _cli()
