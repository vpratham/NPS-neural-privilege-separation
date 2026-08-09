#!/usr/bin/env python3
"""
run_exp017.py

Run the REAL Exp017 probe ensemble against Qwen2.5-3B-Instruct.

This script does NOT retrain probes. It:
  1. Loads the frozen Exp017 PolicyDirection artifacts.
  2. Loads Qwen2.5-3B-Instruct.
  3. Hooks the INPUT to Qwen decoder layers 19,20,21,22.
  4. Applies Exp017's last-token pooling.
  5. Scores each layer with its Exp017 logistic-regression direction.
  6. Applies the Exp017 per-layer thresholds.
  7. Applies the Exp017 2-of-4 voting rule.
  8. Prints the result.

For reproducibility, pass the EXACT prompt strings used by Exp017.
The script does not silently apply a chat template.

Examples:

  # Single prompt
  python run_exp017.py --artifacts ./artifacts \
      --prompt "Explain how photosynthesis works."

  # Interactive
  python run_exp017.py --artifacts ./artifacts --interactive

  # JSONL evaluation file:
  # each line: {"prompt": "...", "label": 0}
  python run_exp017.py --artifacts ./artifacts \
      --jsonl ./exp017_prompts.jsonl \
      --output ./exp017_local_results.jsonl

Notes:
- Exp017 used bfloat16 on CUDA. Use --dtype bfloat16 when the GPU supports it.
- On GPUs that cannot run the model in 3B bfloat16/float16 within VRAM,
  use --device cpu (slow) or add a quantized loading path separately.
- The firewall itself is DETECT-only here. No intervention or generation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_firewall import (
    HFCausalLMAdapter,
    NeuralFirewall,
    Policy,
    ProbeBank,
    VotingStrategy,
    Mode,
)


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LAYERS = [19, 20, 21, 22]
POOLING = "last_token"
VOTE_K = 2
POLICY_NAME = "unsafe_intent"


def parse_dtype(name: str, device: torch.device) -> torch.dtype:
    name = name.lower()
    if name == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        if device.type == "mps":
            return torch.float16
        return torch.float32
    table = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def load_adapter(model_name: str, device_name: str, dtype_name: str):
    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    dtype = parse_dtype(dtype_name, device)

    print(f"[model] {model_name}")
    print(f"[device] {device}")
    print(f"[dtype] {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Match the Exp017 model loading configuration. We instantiate the
    # generic HFCausalLMAdapter directly so the script can explicitly
    # control dtype/device.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    adapter = HFCausalLMAdapter(
        model,
        tokenizer,
        layer_path="model.layers",
        max_length=512,
    )

    if adapter.num_layers() <= max(LAYERS):
        raise RuntimeError(
            f"Model has {adapter.num_layers()} layers, but Exp017 requires "
            f"layer {max(LAYERS)}."
        )

    if adapter.hidden_size() != 2048:
        raise RuntimeError(
            f"Model hidden_size={adapter.hidden_size()}, but Exp017 probes "
            f"expect 2048."
        )

    return adapter


def build_firewall(adapter: HFCausalLMAdapter, artifacts: Path) -> NeuralFirewall:
    bank = ProbeBank.load(
        artifacts,
        expected_d_model=adapter.hidden_size(),
    )

    if bank.policies() != [POLICY_NAME]:
        raise RuntimeError(
            f"Expected policy [{POLICY_NAME!r}], found {bank.policies()}"
        )

    actual_layers = bank.layers_for(POLICY_NAME)
    if actual_layers != LAYERS:
        raise RuntimeError(
            f"Expected probe layers {LAYERS}, found {actual_layers}"
        )

    policy = Policy(
        policy_names=[POLICY_NAME],
        voting=VotingStrategy.K_OF_N,
        vote_k=VOTE_K,
        voting_weights={layer: 1.0 for layer in LAYERS},
        mode=Mode.DETECT,
        fail_closed=True,
    )

    firewall = NeuralFirewall(
        adapter=adapter,
        probe_bank=bank,
        policy=policy,
        pooling=POOLING,
    )

    print("[firewall] Exp017 configuration verified")
    print(f"  policy   : {POLICY_NAME}")
    print(f"  layers   : {LAYERS}")
    print(f"  pooling  : {POOLING}")
    print(f"  voting   : {VOTE_K}-of-{len(LAYERS)}")
    return firewall


def score_one(firewall: NeuralFirewall, prompt: str) -> dict:
    start = time.perf_counter()
    assessment = firewall.score(prompt)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Runtime keys are (policy_name, layer).
    layers = {}
    for key, score in assessment.per_layer_scores.items():
        policy_name, layer = key
        vote = assessment.per_layer_votes[key]
        direction = firewall.probe_bank.get(policy_name, layer)
        layers[str(layer)] = {
            "score": float(score),
            "threshold": float(direction.threshold),
            "vote": bool(vote),
        }

    result = {
        "prompt": prompt,
        "risk_score": float(assessment.risk_score),
        "exceeded_threshold": bool(assessment.exceeded_threshold),
        "decision": "UNSAFE" if assessment.exceeded_threshold else "SAFE",
        "layers": layers,
        "latency_ms": elapsed_ms,
    }
    return result


def print_result(result: dict):
    print("\n" + "=" * 72)
    print(result["decision"])
    print("=" * 72)
    print(f"risk score : {result['risk_score']:.4f}")
    print(f"latency    : {result['latency_ms']:.1f} ms")
    print(f"prompt     : {result['prompt']}")
    print()
    print("Layer results:")
    for layer in LAYERS:
        row = result["layers"].get(str(layer))
        if row is None:
            print(f"  layer {layer}: MISSING")
            continue
        mark = "X" if row["vote"] else " "
        print(
            f"  [{mark}] layer {layer}: "
            f"score={row['score']:.6f}  "
            f"threshold={row['threshold']:.6f}"
        )
    votes = sum(int(result["layers"][str(l)]["vote"]) for l in LAYERS)
    print(f"\nVotes: {votes}/{len(LAYERS)}  (need {VOTE_K})")


def run_interactive(firewall):
    print("\nInteractive Exp017 detector.")
    print("Enter an empty line or Ctrl-D/Ctrl-C to exit.\n")
    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            return
        try:
            result = score_one(firewall, prompt)
            print_result(result)
        except Exception as exc:
            print(f"[ERROR] {type(exc).__name__}: {exc}")


def run_jsonl(firewall, input_path: Path, output_path: Path | None):
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise ValueError(f"Line {line_no}: missing 'prompt'")

            result = score_one(firewall, obj["prompt"])

            # Preserve optional dataset metadata without allowing it to
            # overwrite runtime fields.
            result["input"] = {
                k: v for k, v in obj.items()
                if k != "prompt"
            }
            records.append(result)

            print(
                f"[{len(records)}] "
                f"{result['decision']:6s} "
                f"risk={result['risk_score']:.3f} "
                f"prompt={obj['prompt'][:80]!r}"
            )

    if output_path:
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"\nSaved {len(records)} results to {output_path}")

    # If labels are present, calculate basic held-out metrics.
    labeled = [r for r in records if "label" in r["input"]]
    if labeled:
        y = np.array([int(r["input"]["label"]) for r in labeled])
        pred = np.array([int(r["exceeded_threshold"]) for r in labeled])

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())

        recall = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0

        print("\nMetrics for supplied labels:")
        print(f"  n         : {len(labeled)}")
        print(f"  TP / FP   : {tp} / {fp}")
        print(f"  FN / TN   : {fn} / {tn}")
        print(f"  recall    : {recall:.4f}")
        print(f"  precision : {precision:.4f}")
        print(f"  FPR       : {fpr:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the real Exp017 probe ensemble on Qwen2.5-3B-Instruct."
    )
    parser.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Directory containing the converted Exp017 PolicyDirection artifacts.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Hugging Face model name (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cpu, or mps (default: auto)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model dtype (default: auto; Exp017 used bfloat16 on CUDA).",
    )
    parser.add_argument(
        "--prompt",
        help="Score one prompt.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive prompt loop.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        help="JSONL input. Each line must contain {'prompt': ...}; optional 'label'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL output path when --jsonl is used.",
    )

    args = parser.parse_args()

    modes = int(args.prompt is not None) + int(args.interactive) + int(args.jsonl is not None)
    if modes != 1:
        parser.error("Choose exactly one of --prompt, --interactive, or --jsonl.")

    if not args.artifacts.is_dir():
        parser.error(f"Artifact directory does not exist: {args.artifacts}")

    # Helpful diagnostic before loading the large model.
    print("[torch]", torch.__version__)
    print("[cuda available]", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[cuda device]", torch.cuda.get_device_name(0))
        print("[cuda capability]", torch.cuda.get_device_capability(0))

    adapter = load_adapter(args.model, args.device, args.dtype)
    firewall = build_firewall(adapter, args.artifacts)

    if args.prompt is not None:
        result = score_one(firewall, args.prompt)
        print_result(result)

    elif args.interactive:
        run_interactive(firewall)

    else:
        run_jsonl(firewall, args.jsonl, args.output)


if __name__ == "__main__":
    main()
