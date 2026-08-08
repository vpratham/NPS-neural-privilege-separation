"""
evaluation.py

Batch evaluation of a NeuralFirewall against a labeled prompt set (reusing
the recall/FPR/AUROC framing from Exp017/18). Kept separate from firewall.py
since evaluation is a build/validation-time concern, not part of the
inference path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from .firewall import NeuralFirewall


@dataclass
class EvaluationReport:
    n: int
    precision: float
    recall: float
    f1: float
    fpr: float
    auroc: float | None
    per_prompt_risk: list[float]
    per_prompt_label: list[int]


def evaluate(firewall: NeuralFirewall, prompts: list[str], labels: list[int]) -> EvaluationReport:
    """labels: 1 = should be flagged (unsafe / policy-violating), 0 = benign."""
    risks = []
    preds = []
    for p in prompts:
        assessment = firewall.score(p)
        risks.append(assessment.risk_score)
        preds.append(int(assessment.exceeded_threshold))

    labels_arr = np.array(labels)
    preds_arr = np.array(preds)

    tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())
    tn = int(((preds_arr == 0) & (labels_arr == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    try:
        auroc = roc_auc_score(labels_arr, risks) if len(set(labels)) > 1 else None
    except ValueError:
        auroc = None

    return EvaluationReport(
        n=len(prompts),
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        auroc=auroc,
        per_prompt_risk=risks,
        per_prompt_label=labels,
    )
