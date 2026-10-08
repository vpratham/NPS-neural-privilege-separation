"""Metadata-only split checks; never read or display prospective test prompts.

Host-authored source/group declarations are not proof of novelty, rights or
independent custody. A passing check is not permission to run a final test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re
from urllib.parse import urlsplit

from .artifacts import digest, read, write


SPLITS = ("training", "calibration", "development", "final_test", "adaptive_test")
GROUPS = ("source_group", "template_group", "pair_id", "document_sha256", "request_sha256")


def source_key(location):
    """Normalize known repository URLs; renamed/mirrored sources still need review."""
    parsed = urlsplit(location)
    if parsed.scheme:
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Source location must be an HTTPS URL or repository-relative path")
        parts = parsed.path.strip("/").split("/")
        if parsed.hostname.casefold() == "github.com":
            parts = parts[:2]
        elif parsed.hostname.casefold() == "huggingface.co" and parts[0] == "datasets":
            parts = parts[:3]
            parts[-1] = parts[-1].split("@")[0]
        return "https://" + parsed.hostname.casefold() + "/" + "/".join(parts).removesuffix(".git")
    path = posixpath.normpath(location)
    if path.startswith(("/", "../")) or path in (".", "..") or "\\" in path:
        raise ValueError("Source path must stay repository-relative")
    return path


def validate(manifest, exclusions):
    required = {"kind", "exclusions_sha256", "sources", "cases", "custody", "target_counts"}
    if (not isinstance(manifest, dict) or set(manifest) != required
            or manifest["kind"] != "firewall-split-provenance-v1"):
        raise ValueError("Expected strict split provenance manifest")
    if manifest["exclusions_sha256"] != digest(exclusions):
        raise ValueError("Exclusion registry changed; review and rebind the manifest")
    if (not isinstance(manifest["target_counts"], dict) or set(manifest["target_counts"]) != set(SPLITS)
            or any(type(n) is not int or n < 0 for n in manifest["target_counts"].values())):
        raise ValueError("Invalid target counts")
    if (not isinstance(manifest["custody"], dict) or set(manifest["custody"]) != {"status", "evaluator"}
            or manifest["custody"]["status"] not in ("pending", "declared_sealed")
            or not isinstance(manifest["custody"]["evaluator"], str)
            or manifest["custody"]["status"] == "declared_sealed" and not manifest["custody"]["evaluator"].strip()):
        raise ValueError("Invalid custody declaration")
    excluded = [source_key(e["path"]) for e in exclusions["entries"]]
    sources = {}
    if not isinstance(manifest["sources"], list) or not isinstance(manifest["cases"], list):
        raise ValueError("Sources and cases must be lists")
    for source in manifest["sources"]:
        keys = {"id", "location", "revision", "license", "research_use", "exposure", "near_duplicate_review"}
        if (not isinstance(source, dict) or set(source) != keys
                or any(not isinstance(v, str) or not v.strip() for v in source.values())
                or source["id"] in sources or source["research_use"] not in ("approved", "pending")
                or source["exposure"] not in ("seen", "unseen", "unknown")
                or source["near_duplicate_review"] not in ("pending", "completed")):
            raise ValueError("Invalid or duplicate source declaration")
        source_key(source["location"])
        sources[source["id"]] = source
    ids, requests, owners, counts = set(), set(), {}, dict.fromkeys(SPLITS, 0)
    for case in manifest["cases"]:
        keys = {"id", "split", "condition", "source_id", *GROUPS}
        if (not isinstance(case, dict) or set(case) != keys
                or any(not isinstance(v, str) or not v.strip() for v in case.values())
                or case["id"] in ids or case["split"] not in SPLITS
                or case["condition"] not in ("benign", "attack") or case["source_id"] not in sources):
            raise ValueError("Invalid or duplicate case declaration")
        ids.add(case["id"])
        if case["request_sha256"] in requests:
            raise ValueError("Cross-split overlap or duplicate request digest")
        requests.add(case["request_sha256"])
        split = case["split"]
        counts[split] += 1
        if split in ("training", "calibration") and case["condition"] != "benign":
            raise ValueError("Training and calibration require benign cases")
        for name in GROUPS:
            if name.endswith("sha256") and not re.fullmatch("[0-9a-f]{64}", case[name]):
                raise ValueError("Invalid content digest")
            key = (name, case[name])
            if key in owners and owners[key] != split:
                raise ValueError("Cross-split overlap: " + name)
            owners[key] = split
        if split in ("final_test", "adaptive_test"):
            source = sources[case["source_id"]]
            location = source_key(source["location"])
            if (source["exposure"] != "unseen" or source["research_use"] != "approved"
                    or source["near_duplicate_review"] != "completed"
                    or any(location == old or location.startswith(old + "/") for old in excluded)):
                raise ValueError("Final/adaptive test contains excluded or unreviewed material")
    missing = {split: max(0, manifest["target_counts"][split] - counts[split]) for split in SPLITS}
    return {"kind": "firewall-split-check-v1", "manifest_sha256": digest(manifest),
            "metadata_checks_passed": True, "counts": counts, "missing_target_cases": missing,
            "declared_assembly_complete": not any(missing.values())
                and manifest["custody"]["status"] == "declared_sealed",
            "payload_hashes_verified": False, "independent_custody_verified": False,
            "final_evaluation_authorized": False,
            "scope": "Metadata only; verify payload hashes, actual custody and the frozen runtime before final evaluation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--exclusions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = validate(read(args.manifest), read(args.exclusions))
    report["manifest_file_sha256"] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    write(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["declared_assembly_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
