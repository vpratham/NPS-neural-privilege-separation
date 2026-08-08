"""
calibration.py

Threshold calibration for PolicyDirection artifacts. Kept separate from
probe.py's training path because calibration is something you'll want to
re-run more often than full retraining (e.g. re-target a recall/FPR
operating point without re-fitting the direction itself).

This is used by train_offline.py at build time to produce the `threshold`
baked into each PolicyDirection artifact — the firewall itself never
calls into this module at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_curve


@dataclass
class CalibrationResult:
    threshold: float
    achieved_fpr: float
    achieved_recall: float
    method: str


def calibrate_threshold(
    raw_scores: np.ndarray,
    labels: np.ndarray,
    method: str = "target_fpr",
    target_fpr: float = 0.05,
    target_recall: float = 0.9,
) -> CalibrationResult:
    """Pick an operating threshold on raw_score = weight . h + bias.

    method="target_fpr": lowest threshold achieving FPR <= target_fpr.
    method="target_recall": highest threshold still achieving recall >= target_recall.
    method="youden": maximizes tpr - fpr (Youden's J statistic).

    This mirrors the ROC-calibrated per-layer thresholds already validated
    in Exp017/18 — replace the body with that exact logic once the
    notebook is available so the two stay numerically identical.
    """
    fpr, tpr, thresholds = roc_curve(labels, raw_scores)

    if method == "target_fpr":
        valid = np.where(fpr <= target_fpr)[0]
        idx = valid[np.argmax(tpr[valid])] if len(valid) else int(np.argmin(fpr))
    elif method == "target_recall":
        valid = np.where(tpr >= target_recall)[0]
        idx = valid[np.argmin(fpr[valid])] if len(valid) else int(np.argmax(tpr))
    elif method == "youden":
        idx = int(np.argmax(tpr - fpr))
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return CalibrationResult(
        threshold=float(thresholds[idx]),
        achieved_fpr=float(fpr[idx]),
        achieved_recall=float(tpr[idx]),
        method=method,
    )
