"""Host-owned, pinned document retrieval for the single-realm Q&A pilot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .artifacts import digest, loads


def _identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value)


def load_workload(path):
    """Verify the host manifest and snapshot only this principal's readable data."""
    path = Path(path)
    with path.open("rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise ValueError("Workload manifest exceeds 64 KiB")
    manifest = loads(raw.decode("utf-8"))
    if (not isinstance(manifest, dict) or set(manifest) != {"principal_id", "max_results", "records"}
            or not _identifier(manifest["principal_id"])
            or type(manifest["max_results"]) is not int or not 1 <= manifest["max_results"] <= 8
            or not isinstance(manifest["records"], list) or len(manifest["records"]) > 64):
        raise ValueError("Invalid host workload manifest")
    records, seen = [], set()
    for entry in manifest["records"]:
        if (not isinstance(entry, dict)
                or set(entry) != {"source_id", "record_id", "path", "sha256", "terms", "readers"}
                or not _identifier(entry["source_id"]) or not _identifier(entry["record_id"])
                or entry["source_id"] in seen
                or not isinstance(entry["path"], str) or not entry["path"]
                or not isinstance(entry["sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
                or not isinstance(entry["terms"], list) or not 1 <= len(entry["terms"]) <= 64
                or any(not isinstance(t, str) or not re.fullmatch(r"[a-z0-9_]{1,80}", t) for t in entry["terms"])
                or len(set(entry["terms"])) != len(entry["terms"])
                or not isinstance(entry["readers"], list) or len(entry["readers"]) > 64
                or any(not _identifier(p) for p in entry["readers"])
                or len(set(entry["readers"])) != len(entry["readers"])):
            raise ValueError("Invalid or duplicate workload source")
        seen.add(entry["source_id"])
        with (path.parent / entry["path"]).open("rb") as handle:
            data = handle.read(65537)
        if len(data) > 65536 or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Source size or pinned digest mismatch")
        record = loads(data.decode("utf-8"))
        if (not isinstance(record, dict) or set(record) != {"source_id", "record_id", "value"}
                or record["source_id"] != entry["source_id"] or record["record_id"] != entry["record_id"]
                or not isinstance(record["value"], str) or not record["value"].strip()):
            raise ValueError("Invalid pinned source record")
        if manifest["principal_id"] in entry["readers"]:
            records.append({**record, "sha256": entry["sha256"], "terms": entry["terms"]})
    if sum(len(record["value"]) for record in records) > 160000:
        raise ValueError("Readable source text budget exceeded")
    return {"principal_id": manifest["principal_id"], "max_results": manifest["max_results"],
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "records": records}


def select_records(workload, task):
    """Rank only the authorized startup snapshot; task text never assigns access."""
    words = set(re.findall(r"[a-z0-9_]+", task.casefold()))
    # ponytail: bounded keyword scan; use an ACL-aware app retriever for larger corpora.
    ranked = [(len(words.intersection(record["terms"])), record) for record in workload["records"]]
    ranked.sort(key=lambda item: (-item[0], item[1]["source_id"]))
    return [record for score, record in ranked if score][:workload["max_results"]]


def source_references(records):
    return [{key: record[key] for key in ("source_id", "record_id", "sha256")} for record in records]


class DocumentQAFirewall:
    """One loaded model, fresh permission adapter/cache for each selected bundle.

    Called serially by the pilot worker. This class is not a concurrent server.
    Readable evidence can still redirect the model; references are retrieval
    provenance, not a claim that every generated assertion has been verified.
    """
    mode = "permissions"

    def __init__(self, adapter, workload, *, timeout_seconds):
        self.adapter = adapter
        self.workload = json.loads(json.dumps(workload, allow_nan=False))
        self._binding = digest(self.workload)
        self.timeout_seconds = timeout_seconds

    def run(self, task, context="", *, max_new_tokens):
        from .read_permissions import ReadPermissionAdapter
        from .runtime import Firewall

        if (not isinstance(task, str) or not task.strip() or len(task) > 16000 or context != ""
                or type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 32768):
            raise ValueError("Invalid Q&A request")
        if digest(self.workload) != self._binding:
            raise ValueError("Workload configuration changed")
        # Check the original startup binding BEFORE establishing a fresh adapter's
        # baseline, so a mutated model cannot become the next request's trusted state.
        base = self.adapter
        base._verify_runtime_binding()
        records = select_records(self.workload, task)
        if not records:
            return {"status": "allowed", "output": "No authorized source matched this question.",
                    "reason": "no_authorized_evidence", "observed_steps": 0, "sources": [],
                    "mode": self.mode, "enforced": True}
        documents = {record["source_id"]: record["value"] for record in records}
        adapter = ReadPermissionAdapter.from_components(
            base.model, base.tokenizer, documents=documents, readable_sources=list(documents),
            policy=base.policy, layers=base.layers, projection_dim=base.projection_dim,
            seed=base.seed, max_context=base.max_context, device=base._device,
            model_identifier=base.identity["model_identifier"])
        result = Firewall(adapter, mode="permissions", timeout_seconds=self.timeout_seconds).run(
            task, max_new_tokens=max_new_tokens)
        result["sources"] = source_references(records)
        return result
