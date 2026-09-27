"""Local capture, calibration, generation and demonstration commands."""
import argparse
import json
import sys
from pathlib import Path

from . import artifacts
from .runtime import Firewall


def build_adapter(args):
    from .hf_adapter import HFAdapter
    return HFAdapter(args.model, policy=Path(args.policy).read_text(),
                     layers=[int(value) for value in args.layers.split(",")],
                     projection_dim=args.projection_dim, seed=args.seed,
                     device=args.device, max_context=args.max_context,
                     revision=args.revision, local_files_only=args.local_files_only)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Activation state-space monitor with buffered response release")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Offline synthetic trajectories; verifies gate mechanics only")
    fit = commands.add_parser("fit", help="Fit benign training dynamics and held-out benign threshold")
    fit.add_argument("--training", required=True)
    fit.add_argument("--calibration", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--quantile", type=float, default=1.0)
    for name in ("capture", "run", "serve"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", required=True)
        sub.add_argument("--policy", required=True, help="Host-controlled policy file")
        sub.add_argument("--layers", default="0", help="Zero-based decoder block OUTPUT indices")
        sub.add_argument("--projection-dim", type=int, default=8)
        sub.add_argument("--seed", type=int, default=17)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--max-context", type=int, default=2048)
        sub.add_argument("--revision")
        sub.add_argument("--local-files-only", action="store_true")
        if name == "capture":
            sub.add_argument("--requests", required=True)
            sub.add_argument("--output", required=True)
            sub.add_argument("--max-new-tokens", type=int, default=128)
        else:
            sub.add_argument("--profile", required=True)
            sub.add_argument("--mode", choices=("enforce", "monitor"), default="enforce")
            if name == "run":
                sub.add_argument("--requests", required=True)
            else:
                sub.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            from .demo import demo
            print(json.dumps(demo(), indent=2, allow_nan=False))
            return 0
        if args.command == "fit":
            result = artifacts.fit(artifacts.read(args.training), artifacts.read(args.calibration), quantile=args.quantile)
            artifacts.write(args.output, result)
            print(json.dumps({"written": args.output, "calibration": result["profile"]["calibration"]}))
            return 0
        adapter = build_adapter(args)
        if args.command == "capture":
            result = artifacts.capture(adapter, artifacts.requests(args.requests), args.max_new_tokens)
            artifacts.write(args.output, result)
            print(json.dumps({"written": args.output, "traces": len(result["traces"])}))
            return 0
        artifact = artifacts.validate_profile(artifacts.read(args.profile))
        firewall = Firewall.from_artifact(adapter, artifact, mode=args.mode)
        if args.command == "serve":
            from .server import serve
            serve(firewall, artifact["max_new_tokens"], port=args.port)
            return 0
        results = []
        for item in artifacts.requests(args.requests):
            result = firewall.run(item["task"], item.get("context", ""), max_new_tokens=artifact["max_new_tokens"])
            print(json.dumps(result, allow_nan=False))
            results.append(result)
        return 2 if any(result["status"] in ("error", "incomplete") for result in results) else 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 2
