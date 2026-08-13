import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from neural_firewall.model_interface import build_qwen_adapter
from neural_firewall.activation_extractor import ActivationExtractor


MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYERS = {19, 20, 21, 22}
SEED = 42
MAX_LENGTH = 512


def fingerprint(records):
    h = hashlib.sha256()

    for r in records:
        h.update(str(r["id"]).encode("utf-8"))
        h.update(b"\0")
        h.update(r["prompt"].encode("utf-8"))
        h.update(b"\0")
        h.update(str(r["label"]).encode("utf-8"))
        h.update(b"\0")

    return h.hexdigest()


def load_xstest():
    print("[data] loading XSTest...")

    ds = load_dataset(
        "natolambert/xstest-v2-copy",
        split="prompts",
    )

    records = []

    for row in ds:
        prompt_type = str(row["type"])

        # XSTest defines contrast_* prompts as unsafe.
        label = 1 if prompt_type.startswith("contrast_") else 0

        records.append({
            "id": str(row["id"]),
            "type": prompt_type,
            "prompt": str(row["prompt"]),
            "label": label,
        })

    return records


def stratified_split(records):
    indices = np.arange(len(records))
    labels = np.array([r["label"] for r in records])

    # 50% train, 50% temporary.
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.50,
        random_state=SEED,
        stratify=labels,
    )

    temp_labels = labels[temp_idx]

    # Of the remaining 50%:
    # 40% -> calibration = 20% overall
    # 60% -> test = 30% overall
    cal_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.60,
        random_state=SEED,
        stratify=temp_labels,
    )

    return {
        "train": [records[i] for i in train_idx],
        "calibration": [records[i] for i in cal_idx],
        "held_out_test": [records[i] for i in test_idx],
    }


def print_split_info(splits):
    print("\n[SPLIT]")
    print("=" * 60)

    for name, rows in splits.items():
        labels = np.array([r["label"] for r in rows])

        print(
            f"{name:14s}: "
            f"n={len(rows):3d} "
            f"safe={(labels == 0).sum():3d} "
            f"unsafe={(labels == 1).sum():3d}"
        )


def extract_split(
    extractor,
    rows,
    output_dir,
    batch_size,
    split_name,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(rows)

    # Preallocate float32 arrays.
    activations = {
        layer: np.zeros((n, 2048), dtype=np.float32)
        for layer in sorted(LAYERS)
    }

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        batch = rows[start:end]
        prompts = [r["prompt"] for r in batch]

        print(
            f"[extract] {split_name}: "
            f"{start + 1}-{end}/{n}"
        )

        result = extractor.extract_batch(prompts)

        for layer in sorted(LAYERS):
            x = result.pooled[layer]

            # (batch, 2048) -> CPU float32.
            x = x.detach().float().cpu().numpy()

            if x.shape != (len(batch), 2048):
                raise RuntimeError(
                    f"Unexpected shape at layer {layer}: "
                    f"{x.shape}"
                )

            activations[layer][start:end] = x

        # Free GPU/CPU tensors between batches.
        del result

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for layer in sorted(LAYERS):
        np.save(
            split_dir / f"layer{layer}.npy",
            activations[layer],
        )

    np.save(
        split_dir / "labels.npy",
        np.asarray(
            [r["label"] for r in rows],
            dtype=np.int64,
        ),
    )

    metadata = {
        "split": split_name,
        "n": len(rows),
        "layers": sorted(LAYERS),
        "hidden_size": 2048,
        "pooling": "last_token",
        "dtype_saved": "float32",
        "model": MODEL,
        "seed": SEED,
        "records": rows,
        "fingerprint": fingerprint(rows),
    }

    (split_dir / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"[saved] {split_dir}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        default="./phase1_real/xstest",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    print("[environment]")
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    records = load_xstest()

    print(
        f"[data] total prompts: {len(records)}"
    )

    labels = np.array(
        [r["label"] for r in records]
    )

    print(
        f"[data] safe={int((labels == 0).sum())} "
        f"unsafe={int((labels == 1).sum())}"
    )

    splits = stratified_split(records)

    print_split_info(splits)

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    print("\n[model] loading Qwen...")

    adapter = build_qwen_adapter(
        MODEL,
        dtype=torch.bfloat16,
    )

    print(
        "[model]",
        adapter.num_layers(),
        "layers"
    )

    print(
        "[model] hidden size:",
        adapter.hidden_size(),
    )

    if adapter.num_layers() != 36:
        raise RuntimeError(
            "Unexpected number of Qwen decoder layers."
        )

    if adapter.hidden_size() != 2048:
        raise RuntimeError(
            "Unexpected hidden size."
        )

    extractor = ActivationExtractor(
        adapter,
        LAYERS,
        pooling="last_token",
    )

    # ---------------------------------------------------------
    # EXTRACTION
    # ---------------------------------------------------------

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "experiment": "Phase1 XSTest real activation extraction",
        "dataset": "natolambert/xstest-v2-copy",
        "dataset_split": "prompts",
        "model": MODEL,
        "layers": sorted(LAYERS),
        "pooling": "last_token",
        "max_length": MAX_LENGTH,
        "seed": SEED,
        "batch_size": args.batch_size,
        "split_ratios": {
            "train": 0.50,
            "calibration": 0.20,
            "held_out_test": 0.30,
        },
        "label_definition": (
            "unsafe=1 when XSTest type starts with "
            "'contrast_', otherwise safe=0"
        ),
    }

    (out / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    for split_name in [
        "train",
        "calibration",
        "held_out_test",
    ]:
        extract_split(
            extractor=extractor,
            rows=splits[split_name],
            output_dir=out,
            batch_size=args.batch_size,
            split_name=split_name,
        )

    print("\n[DONE]")
    print(out.resolve())


if __name__ == "__main__":
    main()