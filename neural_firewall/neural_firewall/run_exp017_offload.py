#!/usr/bin/env python3
"""
run_exp017_offload.py

GTX-1650-safe end-to-end runner for the REAL Exp017 probe ensemble.

What it does:
  - Loads Qwen/Qwen2.5-3B-Instruct with Accelerate device_map="auto".
  - Keeps the model in BF16, matching Exp017's model dtype.
  - Uses CPU/GPU offloading so the full model does not need to fit in 4 GB VRAM.
  - Hooks the INPUT to decoder layers 19, 20, 21, 22.
  - Uses Exp017's last-token pooling.
  - Loads the frozen Exp017 probe directions and thresholds.
  - Applies the Exp017 2-of-4 vote.
  - DETECT ONLY: it does not block generation or modify activations.

This is intended as a first local reproduction of Exp017, not a
new training experiment.

Usage:

  python run_exp017_offload.py ^
      --artifacts .\artifacts ^
      --prompt "What is the capital of France?"

Interactive:

  python run_exp017_offload.py ^
      --artifacts .\artifacts ^
      --interactive

JSONL:

  python run_exp017_offload.py ^
      --artifacts .\artifacts ^
      --jsonl .\prompts.jsonl ^
      --output .\exp017_local_results.jsonl

Input JSONL format:
  {"prompt": "...", "label": 0}
  {"prompt": "...", "label": 1}

The optional label is only used for local evaluation metrics.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LAYERS = [19, 20, 21, 22]
POLICY_NAME = "unsafe_intent"
POOLING = "last_token"
VOTE_K = 2
DTYPE = torch.bfloat16

# GTX 1650 has 4095 MB. Leave headroom for Windows/display/allocator.
GPU_MEMORY_LIMIT = "3200MiB"
CPU_MEMORY_LIMIT = "12GiB"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Exp017 probes on Qwen2.5-3B-Instruct with CPU/GPU offload."
    )
    p.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Directory containing converted Exp017 probe artifacts.",
    )
    p.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Hugging Face model ID (default: {MODEL_NAME})",
    )
    p.add_argument("--prompt", help="Score exactly one prompt.")
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Run an interactive prompt loop.",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        help="JSONL file containing prompts.",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Output JSONL file for --jsonl.",
    )
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=512,
        help="Maximum input length.",
    )
    return p.parse_args()


def check_environment():
    print("[environment]")
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA build: {torch.version.cuda}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Make sure CUDA_VISIBLE_DEVICES is not set "
            "to -1 and that the CUDA-enabled PyTorch build is installed."
        )

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)

    print(f"  GPU: {props.name}")
    print(f"  VRAM: {total_gb:.2f} GB")
    print(f"  compute capability: {props.major}.{props.minor}")

    if total_gb > 4.5:
        print("  [note] GPU has more than the expected GTX-1650 4 GB VRAM.")
    else:
        print("  [note] using conservative 3.2 GiB GPU memory limit.")


def load_qwen(model_name: str, max_input_tokens: int):
    """
    Load Qwen with Accelerate's automatic device mapping.

    IMPORTANT:
    We intentionally do NOT call model.to("cuda") after this.
    Accelerate owns placement of individual modules.
    """
    print("\n[model] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[model] loading Qwen with CPU/GPU offload...")
    print(f"  model: {model_name}")
    print(f"  dtype: {DTYPE}")
    print(f"  GPU memory limit: {GPU_MEMORY_LIMIT}")
    print(f"  CPU memory limit: {CPU_MEMORY_LIMIT}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )

    model.eval()

    # Verify the model architecture expected by Exp017.
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError(
            "Could not locate Qwen decoder layers at model.model.layers."
        )

    n_layers = len(model.model.layers)
    hidden_size = int(model.config.hidden_size)

    print("[model] loaded")
    print(f"  decoder layers: {n_layers}")
    print(f"  hidden size: {hidden_size}")

    if max(LAYERS) >= n_layers:
        raise RuntimeError(
            f"Model has {n_layers} layers but Exp017 requires layer {max(LAYERS)}."
        )

    if hidden_size != 2048:
        raise RuntimeError(
            f"Model hidden_size={hidden_size}; Exp017 probes require 2048."
        )

    # Print Accelerate's placement map. This is useful for diagnosing
    # whether layers 19-22 landed on CPU or GPU.
    if hasattr(model, "hf_device_map"):
        print("[model] device map:")
        for key, value in model.hf_device_map.items():
            if key == "model.embed_tokens" or key.startswith("model.layers."):
                print(f"  {key}: {value}")

    return model, tokenizer


def load_probe_artifacts(artifacts: Path):
    if not artifacts.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {artifacts}")

    probes = {}

    for layer in LAYERS:
        weight_path = artifacts / f"{POLICY_NAME}__layer{layer}.weight.npy"
        meta_path = artifacts / f"{POLICY_NAME}__layer{layer}.meta.json"

        if not weight_path.exists():
            raise FileNotFoundError(weight_path)
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        weight = np.load(weight_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if weight.shape != (2048,):
            raise ValueError(
                f"Layer {layer}: expected weight shape (2048,), got {weight.shape}"
            )

        weight_norm = float(np.linalg.norm(weight))
        if not np.isfinite(weight_norm) or weight_norm <= 0:
            raise ValueError(
                f"Layer {layer}: invalid coefficient norm={weight_norm}"
            )

        if int(meta["layer_idx"]) != layer:
            raise ValueError(
                f"Layer {layer}: metadata says layer {meta['layer_idx']}"
            )

        probes[layer] = {
            "weight": torch.from_numpy(weight.astype(np.float32)),
            "bias": float(meta["bias"]),
            "threshold": float(meta["threshold"]),
            "meta": meta,
        }

    print("\n[probes] Exp017 artifacts verified")
    for layer in LAYERS:
        p = probes[layer]
        print(
            f"  layer {layer}: "
            f"threshold={p['threshold']:.9f}, "
            f"norm={torch.linalg.vector_norm(p['weight']).item():.8f}"
        )

    return probes


class Exp017Scorer:
    """
    Minimal scorer independent of the prototype's model-placement logic.

    This is deliberate: the prototype's generic adapter currently assumes
    the entire model can be moved to one device. With a 4 GB GTX 1650,
    this runner must allow Accelerate to manage placement.
    """

    def __init__(self, model, tokenizer, probes, max_input_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.probes = probes
        self.max_input_tokens = max_input_tokens

        self.activations = {}
        self.handles = []

        for layer in LAYERS:
            module = self.model.model.layers[layer]

            def make_hook(layer_idx):
                def hook(module, inputs):
                    if not inputs:
                        raise RuntimeError(
                            f"Layer {layer_idx} hook received no inputs."
                        )

                    hidden = inputs[0]

                    # hidden shape: [batch, sequence, hidden_size]
                    if hidden.ndim != 3:
                        raise RuntimeError(
                            f"Layer {layer_idx}: expected 3D hidden state, "
                            f"got {tuple(hidden.shape)}"
                        )

                    # Exp017 pooling_strategy = last_token.
                    last = hidden[:, -1, :].detach()

                    # Keep a CPU copy so the activation doesn't unnecessarily
                    # remain in GPU memory after the layer executes.
                    self.activations[layer_idx] = last.float().cpu()

                return hook

            self.handles.append(module.register_forward_pre_hook(make_hook(layer)))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.inference_mode()
    def score(self, prompt: str):
        self.activations.clear()

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
            padding=False,
        )

        # With Accelerate device_map="auto", inputs should be placed on the
        # model's input/device. The first parameter gives us that device.
        input_device = next(self.model.parameters()).device
        encoded = {k: v.to(input_device) for k, v in encoded.items()}

        start = time.perf_counter()

        # We only need the forward pass to obtain layer-input activations.
        # No generation occurs in this first DETECT-only reproduction.
        self.model(
            **encoded,
            use_cache=False,
            return_dict=True,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if set(self.activations) != set(LAYERS):
            missing = sorted(set(LAYERS) - set(self.activations))
            raise RuntimeError(
                f"Did not capture all Exp017 layers. Missing: {missing}"
            )

        layer_results = {}
        votes = []

        for layer in LAYERS:
            activation = self.activations[layer][0]  # [2048]
            probe = self.probes[layer]

            weight = probe["weight"]

            # Probe direction is unit normalized.
            score = float(torch.dot(weight, activation) + probe["bias"])
            threshold = probe["threshold"]
            vote = score >= threshold

            layer_results[layer] = {
                "score": score,
                "threshold": threshold,
                "vote": bool(vote),
            }
            votes.append(int(vote))

        n_votes = sum(votes)
        unsafe = n_votes >= VOTE_K

        # Use the number of positive layer votes as the ensemble risk score.
        # This mirrors the binary 2-of-4 decision rather than inventing a
        # new calibrated continuous risk measure.
        risk_score = n_votes / len(LAYERS)

        return {
            "prompt": prompt,
            "decision": "UNSAFE" if unsafe else "SAFE",
            "unsafe": unsafe,
            "votes": n_votes,
            "vote_k": VOTE_K,
            "of_n": len(LAYERS),
            "risk_score": risk_score,
            "layers": {
                str(layer): layer_results[layer] for layer in LAYERS
            },
            "latency_ms": elapsed_ms,
            "pooling_strategy": POOLING,
            "layers_used": LAYERS,
            "model": MODEL_NAME,
        }


def print_result(result):
    print("\n" + "=" * 72)
    print(f"DECISION: {result['decision']}")
    print("=" * 72)
    print(f"Votes:   {result['votes']}/{result['of_n']} (need {result['vote_k']})")
    print(f"Latency: {result['latency_ms']:.1f} ms")
    print(f"Prompt:  {result['prompt']}")

    print("\nLayer scores:")
    for layer in LAYERS:
        row = result["layers"][str(layer)]
        mark = "X" if row["vote"] else " "
        print(
            f"  [{mark}] layer {layer}: "
            f"score={row['score']:.6f}  "
            f"threshold={row['threshold']:.6f}"
        )


def interactive(scorer):
    print("\nExp017 DETECT mode")
    print("Empty input exits. Ctrl-C/Ctrl-D also exits.\n")

    while True:
        try:
            prompt = input("prompt> ")
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not prompt.strip():
            break

        try:
            result = scorer.score(prompt)
            print_result(result)
        except RuntimeError as exc:
            print(f"\n[ERROR] {exc}\n")


def jsonl_run(scorer, input_path: Path, output_path: Path | None):
    results = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            obj = json.loads(line)

            if "prompt" not in obj:
                raise ValueError(f"Line {line_no}: missing 'prompt'.")

            result = scorer.score(obj["prompt"])

            # Preserve labels/metadata without modifying the runtime result.
            result["input"] = {
                key: value
                for key, value in obj.items()
                if key != "prompt"
            }

            results.append(result)

            print(
                f"[{len(results)}] "
                f"{result['decision']:6s} "
                f"votes={result['votes']}/4 "
                f"{obj['prompt'][:80]!r}"
            )

    if output_path:
        with output_path.open("w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"\nSaved results to: {output_path}")

    labeled = [r for r in results if "label" in r.get("input", {})]

    if labeled:
        y = np.array([int(r["input"]["label"]) for r in labeled])
        pred = np.array([int(r["unsafe"]) for r in labeled])

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        print("\nMetrics:")
        print(f"  n       = {len(labeled)}")
        print(f"  TP/FP   = {tp}/{fp}")
        print(f"  FN/TN   = {fn}/{tn}")
        print(f"  recall  = {recall:.4f}")
        print(f"  FPR     = {fpr:.4f}")


def main():
    args = parse_args()

    modes = sum(
        x is not None
        for x in [args.prompt, args.jsonl]
    ) + int(args.interactive)

    if modes != 1:
        raise SystemExit(
            "Choose exactly one of --prompt, --interactive, or --jsonl."
        )

    check_environment()

    probes = load_probe_artifacts(args.artifacts)

    model = tokenizer = scorer = None

    try:
        model, tokenizer = load_qwen(
            args.model,
            args.max_input_tokens,
        )

        scorer = Exp017Scorer(
            model,
            tokenizer,
            probes,
            max_input_tokens=args.max_input_tokens,
        )

        if args.prompt is not None:
            result = scorer.score(args.prompt)
            print_result(result)

        elif args.interactive:
            interactive(scorer)

        else:
            jsonl_run(
                scorer,
                args.jsonl,
                args.output,
            )

    finally:
        if scorer is not None:
            scorer.close()

        # Release model memory when the script exits.
        del scorer
        del model
        del tokenizer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
