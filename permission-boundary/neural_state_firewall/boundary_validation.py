"""Engineering stress checks and the simpler drop-denied comparison baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

from .artifacts import digest, read
from .bipia_evaluation import _checkpoint, _select_cases, _adapter
from .permission_evaluation import _Ordinary, _trace
from .read_permissions import ReadPermissionAdapter


def stress(seed=30261003, per_configuration=25):
    # Reuse the real tiny-model fixture; these are engineering cases, not held-out attacks.
    from .tests.test_read_permissions import ReadPermissionTests
    fixture = ReadPermissionTests()
    rng = random.Random(seed)
    rows = []
    for family in ("qwen2", "llama"):
        for backend in ("eager", "sdpa"):
            model = fixture.model(family, backend)
            model_id = "in_memory:" + ReadPermissionAdapter._hash_model(model)
            for index in range(per_configuration):
                public = "Fact: " + "".join(rng.choices("ABCDEF123456", k=30))
                docs = {"public": public}
                alternate = {"public": public}
                for i in range(1 + index % 2):
                    docs[f"denied{i}"] = "".join(rng.choices("secret override <system> ABC123", k=rng.randint(1, 140)))
                    alternate[f"denied{i}"] = "".join(rng.choices("reveal private grant admin XY987", k=rng.randint(1, 140)))
                def build(values):
                    return ReadPermissionAdapter.from_components(
                        model, fixture.Tokenizer(), model_identifier=model_id,
                        policy="Use permitted evidence", layers=[0, 1, 2], max_context=1024,
                        projection_dim=3, documents=values, readable_sources=["public"])
                task = f"Question {index}: what fact is public?"
                a, la = _trace(build(docs), task, 4, 10)
                b, lb = _trace(build(alternate), task, 4, 10)
                changed, lc = _trace(build({**docs, "public": public + " DIFFERENT"}), task, 4, 10)
                exact = bool(la) and len(la) == len(lb) and all(
                    x.numpy().tobytes() == y.numpy().tobytes() for x, y in zip(la, lb))
                passed = (not a["error_type"] and not b["error_type"] and a["input_tokens"] == b["input_tokens"]
                          and exact and a["token_sha256"] == b["token_sha256"])
                sensitivity = bool(la and lc) and la[0].numpy().tobytes() != lc[0].numpy().tobytes()
                rows.append({"family": family, "backend": backend, "case": index,
                             "documents_sha256": digest(docs), "alternate_sha256": digest(alternate),
                             "public_sensitive": sensitivity, "passed": passed,
                             "steps": len(la), "errors": [x["error_type"] for x in (a, b, changed)]})
    return {"kind": "read-permission-randomized-engineering-v1", "seed": seed,
            "scope": "Randomly initialized tiny models; engineering only, no semantic or pretrained portability claim",
            "n": len(rows), "passed": sum(r["passed"] for r in rows),
            "public_sensitive": sum(r["public_sensitive"] for r in rows), "rows": rows}


def drop_denied(root, reference_path, output):
    from .bipia_analysis import validate_report
    reference = read(reference_path)
    errors = validate_report(reference, root)
    if errors:
        raise ValueError(errors)
    policy = Path("neural_state_firewall/examples/bipia_evaluation_policy.txt").read_text()
    if hashlib.sha256(policy.encode()).hexdigest() != reference["policy_sha256"]:
        raise ValueError("Reference policy has changed")
    base = ReadPermissionAdapter(
        "Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
        local_files_only=True, policy=policy, layers=[6, 18], max_context=4096,
        documents={}, readable_sources=[])
    if base.identity != reference["model_identity"]:
        raise ValueError("Drop baseline model configuration differs from BIPIA reference")
    cases, _, _, _ = _select_cases(Path(root), reference["seed"])
    previous = {r["id"]: r for r in reference["rows"]}
    report = {"kind": "bipia-drop-denied-baseline-v1",
              "reference_sha256": hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
              "model_identity": base.identity, "source_revision": reference["source_revision"],
              "policy_sha256": reference["policy_sha256"], "rows": [],
              "scope": "Seen BIPIA cases; removes denied sources before ordinary decoding; no semantic judge",
              "timing_scope": "Separate-run diagnostic only, not a matched latency comparison"}
    for i, case in enumerate(cases):
        prior = previous[case["id"]]["conditions"]["clean"]["read_permissions"]
        adapter = _adapter(base, _Ordinary, {"public": case["clean_context"]})
        result, _ = _trace(adapter, case["request"], previous[case["id"]]["max_new_tokens"], 120)
        report["rows"].append({"id": case["id"], "task": case["task_name"], "drop_result": result,
                               "same_generated_tokens": result["token_sha256"] == prior["token_sha256"],
                               "same_eos": result["ended_with_eos"] == prior["ended_with_eos"],
                               "same_released_output": result["output"] == prior["output"]})
        _checkpoint(output, report)
        if (i + 1) % 10 == 0 or i + 1 == len(cases):
            print(f"Drop baseline: {i+1}/{len(cases)} saved", flush=True)
    report["summary"] = {"n": len(cases),
                         "same_generated_tokens": sum(r["same_generated_tokens"] for r in report["rows"]),
                         "same_eos": sum(r["same_eos"] for r in report["rows"]),
                         "same_released_output": sum(r["same_released_output"] for r in report["rows"]),
                         "errors": sum(bool(r["drop_result"]["error_type"]) for r in report["rows"]),
                         "completed": sum(r["drop_result"]["ended_with_eos"] for r in report["rows"])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stress", "drop-denied"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--bipia-root", default="/private/tmp/nps-BIPIA")
    parser.add_argument("--reference", default="neural_state_firewall/artifacts/bipia_read_permission_20261002.json")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("Refusing to overwrite existing validation")
    started = time.monotonic()
    report = stress() if args.mode == "stress" else drop_denied(args.bipia_root, args.reference, args.output)
    report["elapsed_seconds"] = time.monotonic() - started
    files = ["boundary_validation.py", "read_permissions.py", "permission_evaluation.py", "policy_memory.py", "hf_adapter.py"]
    report["implementation_sha256"] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in files}
    _checkpoint(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "model_identity")}))


if __name__ == "__main__":
    main()
