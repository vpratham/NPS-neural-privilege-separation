"""Run the frozen Stage 1 candidate-feasibility pilot without response gating."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from datetime import datetime, timezone

from neural_state_firewall.hf_adapter import HFAdapter


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "examples" / "stage1_feasibility_cases.json"
DEFAULT_POLICY = ROOT / "examples" / "stage1_policy.txt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "stage1-feasibility-outputs.json"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MAX_NEW_TOKENS = 64


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases_bytes = DEFAULT_CASES.read_bytes()
    policy = DEFAULT_POLICY.read_text(encoding="utf-8")
    cases = json.loads(cases_bytes)
    if cases.get("kind") != "neural-state-feasibility-pilot-v1" or cases.get("split") != "development":
        raise ValueError("refusing to run a non-development Stage 1 manifest")
    if len(cases.get("cases", [])) != 12:
        raise ValueError("expected the frozen 12-case Stage 1 pilot")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing raw outputs: {args.output}")

    import torch
    import transformers

    adapter = HFAdapter(
        MODEL,
        revision=REVISION,
        policy=policy,
        layers=[6, 18],
        device="cpu",
        local_files_only=True,
    )
    rows = []
    for case in cases["cases"]:
        tokens: list[int] = []
        reached_eos = False
        for frame in adapter.iter_steps(case["task"], case["context"], MAX_NEW_TOKENS):
            if frame.is_eos:
                reached_eos = True
                break
            tokens.append(frame.token_id)
        if not reached_eos:
            raise RuntimeError(f"generation did not reach EOS: {case['case_id']}")
        rows.append({
            "case_id": case["case_id"],
            "output": adapter.decode(tokens),
            "token_count": len(tokens),
            "reached_eos": reached_eos,
        })

    result = {
        "kind": "neural-state-feasibility-output-v1",
        "split": "development",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "revision": REVISION,
        "identity": adapter.identity,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "dtype": "float32",
            "decoding": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "cases_sha256": sha256(cases_bytes),
        "policy_sha256": sha256(policy.encode("utf-8")),
        "rows": rows,
    }
    payload = (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps({
        "output": str(args.output),
        "cases": len(rows),
        "cases_sha256": result["cases_sha256"],
        "results_sha256": sha256(payload),
        "all_reached_eos": all(row["reached_eos"] for row in rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
