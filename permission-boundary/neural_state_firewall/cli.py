"""Commands for the deterministic read-permission boundary."""
import argparse
import json
from pathlib import Path

from . import artifacts
from .runtime import Firewall


def build_adapter(args):
    from .read_permissions import ReadPermissionAdapter
    grants = artifacts.read(args.read_permissions)
    if not isinstance(grants, dict) or set(grants) != {"readable_sources"}:
        raise ValueError("Permission file must contain only readable_sources")
    return ReadPermissionAdapter(
        args.model, documents=artifacts.read(args.documents),
        readable_sources=grants["readable_sources"], policy=Path(args.policy).read_text(),
        layers=[int(value) for value in args.layers.split(",")], device=args.device,
        max_context=args.max_context, revision=args.revision,
        local_files_only=args.local_files_only)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Host-controlled model read permissions")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "serve", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", required=True)
        sub.add_argument("--policy", required=True)
        sub.add_argument("--documents", required=True)
        sub.add_argument("--read-permissions", required=True)
        sub.add_argument("--layers", default="0")
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--max-context", type=int, default=2048)
        sub.add_argument("--revision")
        sub.add_argument("--local-files-only", action="store_true")
        sub.add_argument("--max-new-tokens", type=int, default=64)
        sub.add_argument("--timeout-seconds", type=float, default=60)
        if name == "run":
            sub.add_argument("--requests", required=True)
        elif name == "serve":
            sub.add_argument("--port", type=int, default=8765)
        else:
            sub.add_argument("--cases", required=True)
            sub.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        adapter = build_adapter(args)
        if args.command == "evaluate":
            from .permission_evaluation import evaluate
            report = evaluate(adapter, artifacts.read(args.cases), args.max_new_tokens,
                              timeout_seconds=args.timeout_seconds)
            artifacts.write(args.output, report)
            print(json.dumps({"written": args.output, "summary": report["summary"],
                              "production_eligible": False}))
            return 0 if report["structural_checks_passed"] else 2
        firewall = Firewall(adapter, timeout_seconds=args.timeout_seconds)
        if args.command == "serve":
            from .server import serve
            serve(firewall, args.max_new_tokens, port=args.port)
            return 0
        results = [firewall.run(item["task"], item.get("context", ""),
                                max_new_tokens=args.max_new_tokens)
                   for item in artifacts.requests(args.requests)]
        for result in results:
            print(json.dumps(result, allow_nan=False))
        return 2 if any(row["status"] in ("error", "incomplete") for row in results) else 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__,
                          "message": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
