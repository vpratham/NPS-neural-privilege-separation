import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

ROOT = Path("phase1_real/xstest")
ARTIFACTS = ROOT / "artifacts"
SPLIT = ROOT / "held_out_test"

LAYERS = [19, 20, 21, 22]
VOTE_K = 2


def load_artifact(layer):
    meta_path = ARTIFACTS / f"unsafe_intent__layer{layer}.meta.json"
    weight_path = ARTIFACTS / f"unsafe_intent__layer{layer}.weight.npy"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    weight = np.load(weight_path).astype(np.float32)

    return weight, float(meta["bias"]), float(meta["threshold"])


def main():
    labels = np.load(SPLIT / "labels.npy").astype(np.int64)

    print("\nXSTEST HELD-OUT EVALUATION")
    print("=" * 70)
    print(f"examples: {len(labels)}")
    print(f"safe:     {(labels == 0).sum()}")
    print(f"unsafe:   {(labels == 1).sum()}")
    print(f"vote rule: {VOTE_K}/{len(LAYERS)}")

    layer_predictions = {}
    layer_scores = {}

    for layer in LAYERS:
        X = np.load(SPLIT / f"layer{layer}.npy").astype(np.float32)

        weight, bias, threshold = load_artifact(layer)

        scores = X @ weight + bias
        predictions = (scores >= threshold).astype(np.int64)

        layer_scores[layer] = scores
        layer_predictions[layer] = predictions

        print(f"\nLayer {layer}")
        print(f"  shape:     {X.shape}")
        print(f"  threshold: {threshold:.9f}")
        print(f"  positives: {predictions.sum()}/{len(predictions)}")

    # ---------------------------------------------------------
    # 2-of-4 ensemble
    # ---------------------------------------------------------

    prediction_matrix = np.stack(
        [layer_predictions[layer] for layer in LAYERS],
        axis=1,
    )

    votes = prediction_matrix.sum(axis=1)
    ensemble_pred = (votes >= VOTE_K).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        ensemble_pred,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(labels, ensemble_pred)
    precision = precision_score(
        labels,
        ensemble_pred,
        zero_division=0,
    )
    recall = recall_score(
        labels,
        ensemble_pred,
        zero_division=0,
    )

    safe_mask = labels == 0
    unsafe_mask = labels == 1

    fpr = (
        fp / safe_mask.sum()
        if safe_mask.sum()
        else float("nan")
    )

    tpr = (
        tp / unsafe_mask.sum()
        if unsafe_mask.sum()
        else float("nan")
    )

    print("\nENSEMBLE")
    print("=" * 70)
    print(f"TP:        {tp}")
    print(f"TN:        {tn}")
    print(f"FP:        {fp}")
    print(f"FN:        {fn}")
    print(f"Accuracy:  {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"TPR:       {tpr:.6f}")
    print(f"FPR:       {fpr:.6f}")

    print("\nVOTE DISTRIBUTION")
    print("=" * 70)

    for n in range(len(LAYERS) + 1):
        count = int((votes == n).sum())
        print(f"{n}/{len(LAYERS)}: {count}")

    # ---------------------------------------------------------
    # Per-example results
    # ---------------------------------------------------------

    metadata_path = SPLIT / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        records = metadata.get("records", [])
    else:
        records = []

    results = []

    for i in range(len(labels)):
        row = {
            "index": i,
            "label": int(labels[i]),
            "votes": int(votes[i]),
            "predicted": int(ensemble_pred[i]),
        }

        if i < len(records):
            row["id"] = records[i].get("id")
            row["type"] = records[i].get("type")
            row["prompt"] = records[i].get("prompt")

        for layer in LAYERS:
            row[f"layer{layer}_score"] = float(
                layer_scores[layer][i]
            )
            row[f"layer{layer}_predicted"] = int(
                layer_predictions[layer][i]
            )

        results.append(row)

    output = ROOT / "heldout_evaluation.json"

    summary = {
        "split": "held_out_test",
        "n": len(labels),
        "layers": LAYERS,
        "vote_k": VOTE_K,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "examples": results,
    }

    output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()