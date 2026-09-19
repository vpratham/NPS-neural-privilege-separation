#!/usr/bin/env python3
"""Build a bounded, provenance-preserving NFW-002R dataset from Necent.

The output is compatible with NFW-002's JSONL contract. It is intentionally a
dataset-preparation step, not a benchmark claim: Qwen outputs must be generated
and independently labeled later. Necent access is gated; accept its terms and
authenticate with ``huggingface-cli login`` before running this script.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "Necent/llm-jailbreak-prompt-injection-dataset"
REQUIRED_COLUMNS = {
    "prompt", "prompt_harmful", "prompt_adversarial", "prompt_type",
    "category", "source", "language", "attack_technique",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_prompt(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def truth(value: Any, name: str) -> int:
    if value in (0, 1, False, True):
        return int(value)
    raise ValueError(f"{name} must be 0 or 1, got {value!r}")


def safe_component(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value).strip())[:80] or "unknown"


def stable_rank(row: dict[str, Any]) -> int:
    # Selection is deterministic across streaming order but not presented as a
    # random sample. The manifest records the exact policy and cap.
    return int(sha256_text(canonical_json(row))[:16], 16)


def load_rows(dataset_id: str, revision: str, split: str) -> tuple[Iterable[dict[str, Any]], str]:
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install datasets and huggingface_hub before running this script") from exc

    api = HfApi()
    resolved_revision = revision or api.dataset_info(dataset_id, revision="main").sha
    if not resolved_revision:
        raise RuntimeError("Could not resolve an immutable Hugging Face dataset revision")
    dataset = load_dataset(dataset_id, revision=resolved_revision, split=split, streaming=True)
    return dataset, resolved_revision


def candidate_from_row(row: dict[str, Any], language: str) -> dict[str, Any] | None:
    missing = REQUIRED_COLUMNS - set(row)
    if missing:
        raise ValueError(f"Necent row is missing required fields: {sorted(missing)}")
    if str(row["language"]).lower() != language.lower():
        return None
    prompt = str(row["prompt"] or "").strip()
    if not prompt:
        return None
    harmful = truth(row["prompt_harmful"], "prompt_harmful")
    adversarial = truth(row["prompt_adversarial"], "prompt_adversarial")
    source = safe_component(row["source"])
    prompt_type = safe_component(row["prompt_type"])
    category = safe_component(row["category"])
    technique = safe_component(row["attack_technique"] or "none")
    # The NFW-002R policy treats either harmful content or an instruction attack
    # as blockable. Raw labels stay present so the two mechanisms remain
    # separately reportable.
    intent_label = int(bool(harmful or adversarial))
    prompt_hash = sha256_text(normalize_prompt(prompt))
    return {
        "id": f"necent-{source}-{prompt_hash[:16]}",
        "messages": [{"role": "user", "content": prompt}],
        "intent_label": intent_label,
        # Source-disjoint group: an entire upstream source cannot cross the
        # NFW-002 train/development/calibration/final partitions.
        "group_id": f"source-{source}",
        "source": f"necent:{source}",
        "nfw002r_policy": "block_prompt_harmful_or_prompt_adversarial_v1",
        "prompt_harmful": harmful,
        "prompt_adversarial": adversarial,
        "prompt_type": prompt_type,
        "category": category,
        "attack_technique": technique,
        "language": str(row["language"]),
        "normalized_prompt_sha256": prompt_hash,
    }


def select_candidates(rows: Iterable[dict[str, Any]], language: str, cap: int) -> tuple[list[dict[str, Any]], Counter]:
    # Keep the lowest stable hashes in each stratum without materializing the
    # 1M+ gated source dataset in Colab RAM.
    heaps: dict[tuple[str, int, str], list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    counters = Counter()
    for raw in rows:
        counters["rows_seen"] += 1
        candidate = candidate_from_row(dict(raw), language)
        if candidate is None:
            counters["rows_language_or_empty_excluded"] += 1
            continue
        counters["rows_eligible"] += 1
        stratum = (candidate["source"], candidate["intent_label"], candidate["prompt_type"])
        rank = stable_rank(candidate)
        heap = heaps[stratum]
        # Include ID as a deterministic tie-breaker; heapq must never compare
        # dictionaries if two stable hashes happened to match.
        item = (-rank, candidate["id"], candidate)
        if len(heap) < cap:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = [candidate for heap in heaps.values() for _, _, candidate in heap]
    return sorted(selected, key=lambda row: row["id"]), counters


def deduplicate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["normalized_prompt_sha256"]].append(row)
    kept, audit = [], []
    for prompt_hash, duplicates in by_prompt.items():
        labels = {row["intent_label"] for row in duplicates}
        if len(labels) != 1:
            raise RuntimeError(
                f"Conflicting policy labels for exact normalized prompt {prompt_hash}; "
                "resolve source provenance manually before building a benchmark."
            )
        canonical = min(duplicates, key=lambda row: row["id"])
        kept.append(canonical)
        if len(duplicates) > 1:
            audit.append({
                "normalized_prompt_sha256": prompt_hash,
                "kept_id": canonical["id"],
                "dropped_ids": sorted(row["id"] for row in duplicates if row is not canonical),
                "reason": "exact_normalized_prompt_duplicate",
            })
    return sorted(kept, key=lambda row: row["id"]), audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="NFW-002R foundation_prompts.jsonl path")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", default=None, help="Optional immutable Hub commit; resolved and recorded when omitted")
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-per-stratum", type=int, default=100)
    args = parser.parse_args()
    if args.max_per_stratum < 1:
        raise ValueError("--max-per-stratum must be positive")

    rows, resolved_revision = load_rows(args.dataset_id, args.revision, args.split)
    selected, counters = select_candidates(rows, args.language, args.max_per_stratum)
    records, dedup_audit = deduplicate(selected)
    labels = Counter(row["intent_label"] for row in records)
    source_counts = Counter(row["source"] for row in records)
    if set(labels) != {0, 1}:
        raise RuntimeError(f"Prepared data must contain both policy classes, got {dict(labels)}")
    if len(source_counts) < 8:
        raise RuntimeError("Too few independent upstream sources for source-disjoint NFW-002R splits")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")
    manifest = {
        "artifact_type": "nfw002r_necent_preparation_v1",
        "dataset_id": args.dataset_id,
        "dataset_revision": resolved_revision,
        "split": args.split,
        "language": args.language,
        "selection": "lowest_stable_hash_per_source_intent_prompt_type_stratum",
        "max_per_stratum": args.max_per_stratum,
        "policy": "intent_label = prompt_harmful OR prompt_adversarial",
        "grouping": "source_disjoint",
        "raw_rows": dict(counters),
        "output_records": len(records),
        "intent_label_counts": {str(k): v for k, v in sorted(labels.items())},
        "upstream_sources": len(source_counts),
        "exact_normalized_duplicates_removed": len(dedup_audit),
        "output_sha256": sha256_text(args.output.read_text(encoding="utf-8")),
        "limitations": [
            "Near-duplicate and paraphrase-family detection is not automatic; review final-source overlaps before publication.",
            "Necent response labels are not labels for newly generated Qwen outputs.",
            "The prepared dataset is a candidate benchmark; freeze splits before any monitor selection.",
        ],
    }
    manifest_path = args.output.with_name("nfw002r_dataset_manifest.json")
    audit_path = args.output.with_name("nfw002r_exact_duplicate_audit.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(dedup_audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
