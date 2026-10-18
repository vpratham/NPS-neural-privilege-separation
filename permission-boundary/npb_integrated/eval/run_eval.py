"""Bounded, local-only evaluation runner for NPB Integrated."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_ROOT = ROOT.parent
REPOSITORY_ROOT = BOUNDARY_ROOT.parent
sys.path[:0] = [str(BOUNDARY_ROOT), str(REPOSITORY_ROOT), str(ROOT)]

from src.baseline import BaselineAssistant, PROMPT_TEMPLATE as BASELINE_PROMPT
from src.authorization import Authorization
from src.defended import DefendedAssistant, SYSTEM_PROMPT as DEFENDED_SYSTEM_PROMPT
from src.providers import LocalHFProvider, OllamaProvider
from eval.score import score_record, summarize


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_cases(path: Path, maximum: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("no_cases")
    if maximum < 1 or maximum > 40:
        raise ValueError("max_cases_must_be_between_1_and_40")
    return rows[:maximum]


def run_tests() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([
        str(BOUNDARY_ROOT), str(REPOSITORY_ROOT), environment.get("PYTHONPATH", "")
    ])
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "npb_integrated/tests", "-v"],
        cwd=BOUNDARY_ROOT, env=environment, text=True, capture_output=True, check=False,
    )
    if completed.returncode:
        sys.stderr.write(completed.stdout + completed.stderr)
        raise RuntimeError("unit_tests_failed")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("hf", "ollama"), default="hf")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11436")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "cases.jsonl")
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    run_tests()  # Required before model initialization or any inference call.
    cases = read_cases(args.cases, args.max_cases)
    if args.provider == "hf":
        provider = LocalHFProvider(args.model_path)
        provider._torch.manual_seed(args.seed)
    else:
        provider = OllamaProvider(args.model_path, base_url=args.base_url, seed=args.seed)
    names = ('config.json', 'baseline_results.jsonl', 'defended_results.jsonl', 'comparison.json')
    if any((args.output / name).exists() for name in names):
        raise FileExistsError("evaluation_output_must_not_exist")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1, "seed": args.seed, "case_count": len(cases),
        "provider": provider.identity, "cases": str(args.cases), "cases_sha256": sha256(args.cases),
        "code": {str(path.relative_to(ROOT)): sha256(path)
                 for path in sorted((ROOT / "src").glob("*.py")) + [ROOT / "eval" / "run_eval.py", ROOT / "eval" / "score.py"]},
        "baseline_prompt_template": BASELINE_PROMPT,
        "defended_system_prompt": DEFENDED_SYSTEM_PROMPT,
        "execution": {"cpu_only": True, "max_total_calls": len(cases) * 2, "retries": 0},
        "python": platform.python_version(),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    authorization = Authorization()
    systems = (("baseline", BaselineAssistant(provider, authorization)),
               ("defended", DefendedAssistant(provider, authorization)))
    records = {name: [] for name, _ in systems}
    started = time.perf_counter()
    streams = {name: (args.output / (name + '_results.jsonl')).open('x', encoding='utf-8')
               for name, _ in systems}
    try:
        for case in cases:
            for name, assistant in systems:
                began = time.perf_counter()
                result = assistant.answer(case["question"], case["document"])
                record = {"case_id": case.get("id"), "system": name, "case": case, "result": result,
                          "runtime_seconds": time.perf_counter() - began}
                record["scores"] = score_record(case, result)
                records[name].append(record)
                streams[name].write(json.dumps(record, sort_keys=True) + '\n')
                streams[name].flush()
                print(f"{case['id']} {name}: {result['status']}", flush=True)
                if getattr(provider, 'stopped', False):
                    raise RuntimeError('cpu_verification_failed_evaluation_stopped')
    finally:
        for stream in streams.values():
            stream.close()
        authorization.close()
    all_records = records["baseline"] + records["defended"]
    model_attempts = provider.calls
    if model_attempts > len(cases) * 2:
        raise RuntimeError("model_call_limit_exceeded")
    errors = sum(1 for row in all_records if row["result"].get("status") == "error")
    comparison = {
        "schema_version": 1, "runtime_seconds": time.perf_counter() - started,
        "case_count": len(cases), "model_calls_attempted": model_attempts,
        "errors": errors,
        "baseline": summarize(cases, records["baseline"]),
        "defended": summarize(cases, records["defended"]),
    }
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
