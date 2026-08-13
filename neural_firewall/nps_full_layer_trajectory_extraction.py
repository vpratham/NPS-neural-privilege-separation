"""
NPS full-layer trajectory experiment
====================================

Captures the LAST-VALID-TOKEN INPUT activation at every decoder layer of
Qwen2.5-3B-Instruct and saves a compact trajectory dataset.

Default input:
    phase1_real/xstest/heldout_evaluation.json

The script deliberately does NOT train new probes. It uses the existing
XSTest-trained unsafe-intent probes only at layers 19-22, while capturing
all 36 layers.

Outputs:
    <out>/
      trajectories.npz
      metadata.json
      summary.json

For each selected prompt:
    - activation[layer] = pooled 2048-D layer-input state
    - unsafe margin is computed at layers 19-22 using the existing probe
      artifacts, where available.

Default selection:
    - all 135 held-out XSTest examples
    - optionally --only-false-negatives
    - optionally --limit N

This is an analysis experiment, not an intervention system.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from neural_firewall.model_interface import build_qwen_adapter


MODEL = "Qwen/Qwen2.5-3B-Instruct"
N_LAYERS = 36
HIDDEN = 2048
PROBE_LAYERS = [19, 20, 21, 22]
DEFAULT_DATA = "phase1_real/xstest/heldout_evaluation.json"


def load_examples(path: Path, only_false_negatives: bool, limit: int | None):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    examples = data["examples"]

    if only_false_negatives:
        examples = [
            x for x in examples
            if int(x["label"]) == 1 and int(x["predicted"]) == 0
        ]

    if limit is not None:
        examples = examples[:limit]

    if not examples:
        raise RuntimeError("No examples selected.")

    return examples


def load_probes(artifact_dir: Path):
    probes = {}

    for layer in PROBE_LAYERS:
        weight_path = artifact_dir / f"unsafe_intent__layer{layer}.weight.npy"
        meta_path = artifact_dir / f"unsafe_intent__layer{layer}.meta.json"

        if not weight_path.exists() or not meta_path.exists():
            print(f"[probe] missing layer {layer}; projection unavailable")
            continue

        weight = np.load(weight_path).astype(np.float32)

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        probes[layer] = {
            "weight": weight,
            "bias": float(meta["bias"]),
            "threshold": float(meta["threshold"]),
        }

    return probes


def capture_batch(model, tokenizer, prompts, layers, max_length):
    """
    Direct reference-style capture:
        decoder-layer INPUT
        -> inputs[0]
        -> last valid token

    This intentionally mirrors the extraction convention already validated
    against the production ActivationExtractor.
    """
    captured = {}

    handles = []

    def make_hook(layer_idx):
        def hook(module, inputs):
            captured[layer_idx] = inputs[0].detach()
        return hook

    for layer_idx in layers:
        layer = model.get_decoder_layer(layer_idx)
        handles.append(
            layer.register_forward_pre_hook(make_hook(layer_idx))
        )

    try:
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

        # Determine model device. The current Qwen adapter uses a single
        # device in the local configuration.
        try:
            device = model.device()
        except Exception:
            device = next(model.model.parameters()).device

        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            model.model(**enc)

        mask = enc["attention_mask"]
        last = mask.sum(dim=1) - 1

        pooled = {}

        for layer_idx in layers:
            hs = captured[layer_idx]

            rows = []
            for i in range(hs.shape[0]):
                rows.append(hs[i, int(last[i].item())])

            pooled[layer_idx] = torch.stack(rows)

        return pooled

    finally:
        for h in handles:
            h.remove()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
    )
    parser.add_argument(
        "--artifacts",
        default="phase1_real/xstest/artifacts",
    )
    parser.add_argument(
        "--out",
        default="phase1_real/xstest/full_layer_trajectories",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--only-false-negatives",
        action="store_true",
    )

    args = parser.parse_args()

    torch.set_grad_enabled(False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    examples = load_examples(
        Path(args.data),
        args.only_false_negatives,
        args.limit,
    )

    prompts = [str(x["prompt"]) for x in examples]

    print("# NPS FULL-LAYER TRAJECTORY EXTRACTION")
    print()
    print(f"model:       {MODEL}")
    print(f"examples:    {len(examples)}")
    print(f"layers:      0-{N_LAYERS - 1}")
    print(f"hidden size: {HIDDEN}")
    print(f"batch size:  {args.batch_size}")
    print(f"max length:  {args.max_length}")

    adapter = build_qwen_adapter(
        MODEL,
        dtype="auto",
    )

    if adapter.num_layers() != N_LAYERS:
        raise RuntimeError(
            f"Expected {N_LAYERS} layers, got {adapter.num_layers()}"
        )

    if adapter.hidden_size() != HIDDEN:
        raise RuntimeError(
            f"Expected hidden size {HIDDEN}, got {adapter.hidden_size()}"
        )

    probes = load_probes(Path(args.artifacts))

    # [examples, layers, hidden]
    activations = np.zeros(
        (len(examples), N_LAYERS, HIDDEN),
        dtype=np.float32,
    )

    for start in range(0, len(prompts), args.batch_size):
        end = min(start + args.batch_size, len(prompts))

        print(
            f"[extract] {start + 1}-{end}/{len(prompts)}",
            flush=True,
        )

        pooled = capture_batch(
            adapter,
            adapter.tokenizer,
            prompts[start:end],
            range(N_LAYERS),
            args.max_length,
        )

        for layer in range(N_LAYERS):
            x = pooled[layer].float().cpu().numpy()

            if x.shape != (end - start, HIDDEN):
                raise RuntimeError(
                    f"Unexpected shape layer {layer}: {x.shape}"
                )

            activations[start:end, layer, :] = x

        del pooled

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Probe margins for the four already-trained layers.
    # ---------------------------------------------------------

    probe_scores = np.full(
        (len(examples), len(PROBE_LAYERS)),
        np.nan,
        dtype=np.float32,
    )

    probe_margins = np.full_like(probe_scores, np.nan)

    for j, layer in enumerate(PROBE_LAYERS):
        if layer not in probes:
            continue

        p = probes[layer]
        scores = (
            activations[:, layer, :]
            @ p["weight"]
            + p["bias"]
        )

        margins = (
            scores - p["threshold"]
        ) / max(abs(p["threshold"]), 1e-8)

        probe_scores[:, j] = scores
        probe_margins[:, j] = margins

    # ---------------------------------------------------------
    # Basic diagnostics.
    # ---------------------------------------------------------

    norms = np.linalg.norm(activations, axis=2)

    finite = bool(np.isfinite(activations).all())

    # Mean activation norm by layer, split by label.
    labels = np.array(
        [int(x["label"]) for x in examples],
        dtype=np.int64,
    )

    norm_summary = {}

    for layer in range(N_LAYERS):
        row = norms[:, layer]

        norm_summary[str(layer)] = {
            "all_mean": float(row.mean()),
            "all_std": float(row.std()),
            "safe_mean": (
                float(row[labels == 0].mean())
                if np.any(labels == 0)
                else None
            ),
            "unsafe_mean": (
                float(row[labels == 1].mean())
                if np.any(labels == 1)
                else None
            ),
        }

    # Save the large numeric artifact separately.
    np.savez_compressed(
        out / "trajectories.npz",
        activations=activations,
        labels=labels,
        probe_scores=probe_scores,
        probe_margins=probe_margins,
        layers=np.arange(N_LAYERS),
        probe_layers=np.array(PROBE_LAYERS),
    )

    metadata = {
        "model": MODEL,
        "n_examples": len(examples),
        "n_layers": N_LAYERS,
        "hidden_size": HIDDEN,
        "pooling": "last_valid_token",
        "captured": "decoder_layer_input",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "probe_layers": PROBE_LAYERS,
        "probe_source": str(Path(args.artifacts)),
        "data_source": str(Path(args.data)),
        "only_false_negatives": args.only_false_negatives,
        "examples": examples,
    }

    (out / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "finite": finite,
        "activation_shape": list(activations.shape),
        "min": float(activations.min()),
        "max": float(activations.max()),
        "mean": float(activations.mean()),
        "std": float(activations.std()),
        "norm_summary": norm_summary,
        "probe_layers": PROBE_LAYERS,
    }

    (out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("[DONE]")
    print(f"saved: {out.resolve()}")
    print(f"activation shape: {activations.shape}")
    print(f"finite: {finite}")
    print()
    print("Probe-margin summary:")
    for j, layer in enumerate(PROBE_LAYERS):
        if np.isnan(probe_margins[:, j]).all():
            continue
        print(
            f"  layer {layer}: "
            f"mean={probe_margins[:, j].mean():+.4f} "
            f"min={probe_margins[:, j].min():+.4f} "
            f"max={probe_margins[:, j].max():+.4f}"
        )


if __name__ == "__main__":
    main()
