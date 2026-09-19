#!/usr/bin/env python3
"""Convert NFW-01 safe/refusal CSVs to the NFW-002 development data contract.

This is deliberately an explicit preparation step. It is useful for a Colab
smoke test, but is not a substitute for reviewed source and paraphrase-family
annotations in a frozen research dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def row_id(prefix: str, position: int, prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{position:05d}-{digest}"


def convert(path: Path, intent_label: int, source_prefix: str) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"prompt", "category"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        result = []
        for position, raw in enumerate(reader):
            prompt = (raw.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(f"{path}: empty prompt at row {position + 2}")
            category = (raw.get("category") or "unknown").strip()
            identifier = row_id(source_prefix, position, prompt)
            result.append({
                "id": identifier,
                "messages": [{"role": "user", "content": prompt}],
                "intent_label": intent_label,
                # CSV source has no verified prompt-family linkage. Unique groups
                # avoid an invented assertion that rows are independent families.
                "group_id": f"unannotated-{identifier}",
                "source": f"{source_prefix}:{category}",
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safe-csv", type=Path, required=True)
    parser.add_argument("--refusal-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = convert(args.safe_csv, 0, "nfw01_safe") + convert(args.refusal_csv, 1, "nfw01_refusal")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Input CSVs generated duplicate IDs")
    if not rows:
        raise ValueError("No rows found in source CSVs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(rows), "purpose": "development_only"}))


if __name__ == "__main__":
    main()
