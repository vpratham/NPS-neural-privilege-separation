"""Bounded Colab-only baseline qualification; no local pretrained inference.

Upload the documented git archive to /content/nps-day1-source.tar.gz, then run
this file with `colab exec -s nps-day1-20261004 -f .../colab_day1.py`.
Only previously seen synthetic development fixtures are consumed.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
import time


BASELINE = "3ebbb4c6ad22e26a614b3591cf11864f97ba6f3a"
ARCHIVE_SHA256 = "664752df3cdf8fd7241187e32c317d2b0fb6e0daf4c37a857617cae74c71e0dc"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def summarize_gpu(report):
    """Separate completed development mechanics from any efficacy claim."""
    rows = report["rows"]
    attempts = [r["arms"][arm] for r in rows
                for arm in ("ordinary", "isolated_memory", "read_permissions")]
    attempts += [r["denied_variant"] for r in rows]
    complete = sum(r["ended_with_eos"] and r["error_type"] is None for r in attempts)
    structural = sum(r["structural_check_passed"] is True for r in rows)
    return {"cases": len(rows), "attempts": len(attempts), "eos_without_error": complete,
            "errors": sum(r["error_type"] is not None for r in attempts),
            "structural_checks_passed": structural,
            "development_mechanics_passed": bool(rows) and structural == len(rows)
                and report["structural_checks_passed"] is True and complete == len(attempts),
            "semantic_security_approved": False, "production_approved": False}


def main():
    if platform.system() != "Linux" or not Path("/content").is_dir():
        raise RuntimeError("This job requires a Colab Linux runtime; do not run model evaluation locally")
    archive = Path("/content/nps-day1-source.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Source archive does not match the pinned baseline")
    root = Path("/content/nps-day1-" + BASELINE[:12] + "-pyenv4")
    root.mkdir(exist_ok=True)
    with (root / ".run.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = root / "day1-report.json"
        if output.exists():
            raise FileExistsError("Preserve the previous run; this job refuses to repeat it")
        with tarfile.open(archive) as bundle:
            bundle.extractall(root, filter="data")
        result = {"kind": "colab-day1-preflight-v1", "baseline_commit": BASELINE,
                  "source_archive_sha256": ARCHIVE_SHA256, "python": platform.python_version(),
                  "platform": platform.platform(), "commands": [], "gpu_summary": None,
                  "scope": "Seen synthetic development mechanics; no fresh-test or production qualification",
                  "execution": "One sequential supervisor; cooperative exclusive lease; no automatic retries",
                  "semantic_security_approved": False, "production_approved": False}
        started = time.monotonic()
        environment = {**os.environ, "HF_HOME": str(root / "hf-cache"),
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false"}

        def run(label, arguments, timeout):
            print(json.dumps({"phase": label, "state": "started"}), flush=True)
            before = time.monotonic()
            with (root / (label + ".log")).open("x") as log:
                process = subprocess.Popen(arguments, cwd=root, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                timed_out = False
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            row = {"phase": label, "exit_code": process.returncode, "timed_out": timed_out,
                   "wall_seconds": round(time.monotonic() - before, 3)}
            result["commands"].append(row)
            print(json.dumps(row), flush=True)
            if process.returncode or timed_out:
                raise RuntimeError("Remote phase failed: " + label)

        try:
            run("hardware", ["nvidia-smi"], 20)
            # Colab omits ensurepip; use the host pip's supported --python target.
            run("venv", [sys.executable, "-m", "venv", "--without-pip", str(root / ".venv")], 90)
            python = str(root / ".venv/bin/python")
            run("install", [sys.executable, "-m", "pip", "--python", python, "install", "-r",
                            "neural_state_firewall/requirements.txt"], 900)
            run("environment", [sys.executable, "-m", "pip", "--python", python, "freeze"], 30)
            run("firewall_tests", [python, "-m", "unittest", "discover", "-s",
                                   "neural_state_firewall/tests", "-v"], 300)
            run("gateway_tests", [python, "-m", "unittest", "discover", "-s", "tests/gateway", "-v"], 120)
            setup = ("import json, torch, transformers; "
                     "assert torch.cuda.is_available(), 'CUDA is required; CPU fallback is prohibited'; "
                     "torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; "
                     "torch.use_deterministic_algorithms(True); "
                     "print(json.dumps({'torch':torch.__version__, 'transformers':transformers.__version__, "
                     "'cuda':torch.version.cuda, 'gpu':torch.cuda.get_device_name(0)}), flush=True); "
                     "from neural_state_firewall.cli import main; raise SystemExit(main())")
            examples = "neural_state_firewall/examples/"
            run("gpu_development", [python, "-c", setup, "evaluate-permissions", "--model", MODEL,
                "--revision", REVISION, "--device", "cuda:0", "--policy", examples + "permission_policy.txt",
                "--documents", examples + "permission_documents.json", "--read-permissions", examples + "read_permissions.json",
                "--cases", examples + "permission_cases.json", "--layers", "6,18", "--max-context", "2048",
                "--max-new-tokens", "64", "--timeout-seconds", "60", "--output", "gpu-development.json"], 600)
            report = json.loads((root / "gpu-development.json").read_text())
            result["gpu_summary"] = summarize_gpu(report)
            result["adapter_identity"] = report["reference_configuration_identity"]
            result["gpu_report_sha256"] = hashlib.sha256((root / "gpu-development.json").read_bytes()).hexdigest()
        finally:
            result["total_wall_seconds"] = round(time.monotonic() - started, 3)
            result["evidence_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in root.glob("*.log")}
            with output.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
            # Only explicit reports/logs; never credentials, model weights or arbitrary runtime files.
            with tarfile.open(root / "day1-evidence.tar.gz", "w:gz") as bundle:
                for path in sorted(root.glob("*.log")) + [output, root / "gpu-development.json"]:
                    if path.is_file():
                        bundle.add(path, arcname=path.name)
            print(json.dumps({"report": str(output), "bundle": str(root / "day1-evidence.tar.gz")}), flush=True)
        if not result["gpu_summary"]["development_mechanics_passed"]:
            raise RuntimeError("GPU mechanics or completion gate failed; inspect preserved evidence")


if __name__ == "__main__":
    main()
