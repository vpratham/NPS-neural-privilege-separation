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


def derive_intent_label(raw: dict[str, str], path: Path, row_number: int) -> int:
    """Use the source's per-row refusal annotation; never label a whole file."""
    expected_refusal = (raw.get("expected_refusal") or "").strip()
    source_label = (raw.get("label") or "").strip().lower()
    if expected_refusal in {"0", "1"}:
        return int(expected_refusal)
    if source_label in {"safe", "benign", "allow", "allowed"}:
        return 0
    if source_label in {"refusal", "unsafe", "harmful", "block", "blocked"}:
        return 1
    raise ValueError(
        f"{path}: row {row_number} has no recognized per-row safety label "
        f"(expected_refusal={expected_refusal!r}, label={source_label!r})"
    )


def convert(path: Path, source_prefix: str) -> list[dict]:
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
            intent_label = derive_intent_label(raw, path, position + 2)
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

    rows = convert(args.safe_csv, "nfw01_policy") + convert(args.refusal_csv, "nfw01_adversarial")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Input CSVs generated duplicate IDs")
    if not rows:
        raise ValueError("No rows found in source CSVs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "records": len(rows),
        "intent_label_counts": {str(label): sum(r["intent_label"] == label for r in rows) for label in (0, 1)},
        "purpose": "development_only",
    }))


if __name__ == "__main__":
    main()
