"""
NPS 36-layer train/calibration/held-out probe sweep.

Uses the EXISTING real XSTest activation splits:
  phase1_real/xstest/train/activations.npz
  phase1_real/xstest/train/labels.npy
  phase1_real/xstest/calibration/activations.npz
  phase1_real/xstest/calibration/labels.npy
  phase1_real/xstest/held_out_test/activations.npz
  phase1_real/xstest/held_out_test/labels.npy

For every layer:
  1. fit LogisticRegression on train
  2. normalize its direction exactly like train_offline.py
  3. calibrate threshold on calibration set at target FPR 0.02
  4. evaluate ONCE on held-out test
  5. compute TPR/FPR/precision/AUC and Wilson CIs

This is the clean probe sweep. It does NOT touch the model.

Selection rule for "robust earliest":
  - held-out FPR <= 0.05
  - held-out TPR >= 0.70
  - held-out ROC-AUC >= 0.90
  - among layers satisfying all three, choose the earliest layer.
The rule is explicit and can be changed via CLI.

Outputs:
  phase1_real/xstest/36layer_probe_sweep/
    sweep.csv
    sweep.json
    selected_layer.json
    artifacts/unsafe_intent__layer*.weight.npy
    artifacts/unsafe_intent__layer*.meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    roc_auc_score,
)
from neural_firewall.calibration import calibrate_threshold
from neural_firewall.probe import PolicyDirection


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * np.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return [float(max(0, center-half)), float(min(1, center+half))]


def fit_direction(X, y, cal_X, cal_y, layer, target_fpr):
    clf = LogisticRegression(
        max_iter=2000,
        C=1.0,
        random_state=42,
    )
    clf.fit(X, y)

    w = clf.coef_[0]
    norm = np.linalg.norm(w)
    if norm == 0:
        raise RuntimeError(f"Zero logistic direction at layer {layer}")

    w = (w / norm).astype(np.float32)
    b = float(np.float32(clf.intercept_[0] / norm))

    cal_scores = (
        cal_X.astype(np.float32) @ w + b
    ).astype(np.float64)

    cal = calibrate_threshold(
        cal_scores,
        cal_y,
        method="target_fpr",
        target_fpr=target_fpr,
    )

    return w, b, float(cal.threshold), cal


def evaluate(X, y, w, b, threshold):
    scores = X.astype(np.float32) @ w + b
    pred = (scores >= threshold).astype(np.int64)

    safe = y == 0
    unsafe = y == 1

    tp = int(np.sum(pred[unsafe] == 1))
    fn = int(np.sum(pred[unsafe] == 0))
    fp = int(np.sum(pred[safe] == 1))
    tn = int(np.sum(pred[safe] == 0))

    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": float(tpr),
        "fpr": float(fpr),
        "precision": float(
            precision_score(y, pred, zero_division=0)
        ),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "tpr_ci": wilson(tp, tp + fn),
        "fpr_ci": wilson(fp, fp + tn),
        "scores": scores,
        "pred": pred,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="phase1_real/xstest",
    )
    p.add_argument("--target-fpr", type=float, default=0.02)
    p.add_argument("--min-tpr", type=float, default=0.70)
    p.add_argument("--min-auc", type=float, default=0.90)
    p.add_argument("--max-fpr", type=float, default=0.05)
    args = p.parse_args()

    root = Path(args.root)
    out = root / "36layer_probe_sweep"
    artifact_dir = out / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_z = np.load(root / "train" / "activations.npz")
    cal_z = np.load(root / "calibration" / "activations.npz")
    test_z = np.load(root / "held_out_test" / "activations.npz")

    train_y = np.load(root / "train" / "labels.npy").astype(np.int64)
    cal_y = np.load(root / "calibration" / "labels.npy").astype(np.int64)
    test_y = np.load(root / "held_out_test" / "labels.npy").astype(np.int64)

    layers = sorted(
        int(k) for k in train_z.files
    )

    rows = []

    for layer in layers:
        X = train_z[str(layer)]
        C = cal_z[str(layer)]
        T = test_z[str(layer)]

        w, b, threshold, cal = fit_direction(
            X, train_y, C, cal_y, layer, args.target_fpr
        )

        result = evaluate(
            T, test_y, w, b, threshold
        )

        direction = PolicyDirection(
            weight=w,
            bias=b,
            threshold=threshold,
            policy_name="unsafe_intent",
            layer_idx=layer,
            d_model=X.shape[1],
            metadata={
                "training_dataset": "XSTest",
                "calibration_target_fpr": args.target_fpr,
                "calibration_achieved_fpr": cal.achieved_fpr,
                "calibration_achieved_recall": cal.achieved_recall,
                "n_train": int(len(train_y)),
                "n_calibration": int(len(cal_y)),
                "n_held_out": int(len(test_y)),
            },
        )
        direction.to_files(artifact_dir)

        row = {
            "layer": layer,
            "threshold": threshold,
            "calibration_fpr": float(cal.achieved_fpr),
            "calibration_tpr": float(cal.achieved_recall),
            **{
                k: v for k, v in result.items()
                if k not in ("scores", "pred")
            },
            "tpr_ci_lo": result["tpr_ci"][0],
            "tpr_ci_hi": result["tpr_ci"][1],
            "fpr_ci_lo": result["fpr_ci"][0],
            "fpr_ci_hi": result["fpr_ci"][1],
            "selected_by_rule": (
                result["fpr"] <= args.max_fpr
                and result["tpr"] >= args.min_tpr
                and result["roc_auc"] >= args.min_auc
            ),
        }
        rows.append(row)

        print(
            f"L{layer:02d} | "
            f"TPR={result['tpr']:.4f} "
            f"FPR={result['fpr']:.4f} "
            f"AUC={result['roc_auc']:.4f} "
            f"Prec={result['precision']:.4f} "
            f"threshold={threshold:.6f}"
        )

    candidates = [
        r for r in rows
        if r["selected_by_rule"]
    ]

    selected = min(
        candidates,
        key=lambda r: r["layer"],
    ) if candidates else None

    summary = {
        "target_fpr": args.target_fpr,
        "selection_rule": {
            "heldout_fpr_max": args.max_fpr,
            "heldout_tpr_min": args.min_tpr,
            "heldout_auc_min": args.min_auc,
            "choose": "earliest layer",
        },
        "selected": selected,
        "candidates": candidates,
        "rows": rows,
    }

    (out / "sweep.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    import csv
    with (out / "sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    (out / "selected_layer.json").write_text(
        json.dumps(selected, indent=2),
        encoding="utf-8",
    )

    print()
    if selected:
        print(
            f"SELECTED EARLIEST ROBUST LAYER: "
            f"{selected['layer']}"
        )
    else:
        print(
            "NO LAYER SATISFIED THE ROBUST-SELECTION RULE."
        )

    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
