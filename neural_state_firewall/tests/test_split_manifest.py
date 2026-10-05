"""Source/template leakage must not be hidden by different request hashes."""
import copy
import unittest

from neural_state_firewall.artifacts import digest
from neural_state_firewall.split_manifest import SPLITS, validate


class SplitManifestTests(unittest.TestCase):
    def manifest(self):
        exclusions = {"entries": [{"path": "https://github.com/microsoft/BIPIA/tree/old/benchmark"}]}
        source = {"id": "fresh", "location": "https://example.org/corpus", "revision": "v1",
                  "license": "test fixture", "research_use": "approved", "exposure": "unseen",
                  "near_duplicate_review": "completed"}
        cases = [{"id": split, "split": split, "condition": "benign", "source_id": "fresh",
                  **{key: split for key in ("source_group", "template_group", "pair_id")},
                  "document_sha256": digest([split, "document"]), "request_sha256": digest([split, "request"])}
                 for split in SPLITS]
        return {"kind": "firewall-split-provenance-v1", "exclusions_sha256": digest(exclusions),
                "sources": [source], "cases": cases, "target_counts": dict.fromkeys(SPLITS, 1),
                "custody": {"status": "declared_sealed", "evaluator": "fixture"}}, exclusions

    def test_complete_metadata_is_not_runtime_or_custody_authorization(self):
        manifest, exclusions = self.manifest()
        report = validate(manifest, exclusions)
        self.assertTrue(report["declared_assembly_complete"])
        self.assertFalse(report["final_evaluation_authorized"])
        self.assertFalse(report["payload_hashes_verified"])
        manifest["cases"] = []
        self.assertFalse(validate(manifest, exclusions)["declared_assembly_complete"])

    def test_each_group_and_digest_overlap_is_rejected(self):
        for key in ("source_group", "template_group", "pair_id", "document_sha256", "request_sha256"):
            with self.subTest(key=key):
                manifest, exclusions = self.manifest()
                manifest["cases"][3][key] = manifest["cases"][0][key]
                with self.assertRaisesRegex(ValueError, "Cross-split overlap"):
                    validate(manifest, exclusions)

    def test_seen_source_cannot_be_laundered_through_new_revision(self):
        manifest, exclusions = self.manifest()
        manifest["sources"][0]["location"] = "https://github.com/microsoft/BIPIA/tree/new/benchmark"
        with self.assertRaisesRegex(ValueError, "excluded or unreviewed"):
            validate(manifest, exclusions)

    def test_incomplete_rights_review_stale_exclusions_and_attack_training_rejected(self):
        for field, value in (("exposure", "seen"), ("research_use", "pending"), ("near_duplicate_review", "pending")):
            manifest, exclusions = self.manifest()
            manifest["sources"][0][field] = value
            with self.assertRaises(ValueError):
                validate(manifest, exclusions)
        manifest, exclusions = self.manifest()
        exclusions["entries"].append({"path": "another/seen/source"})
        with self.assertRaisesRegex(ValueError, "registry changed"):
            validate(manifest, exclusions)
        manifest, exclusions = self.manifest()
        manifest["cases"][0]["condition"] = "attack"
        with self.assertRaisesRegex(ValueError, "benign cases"):
            validate(manifest, exclusions)

    def test_ambiguous_schema_and_duplicate_ids_rejected(self):
        original, exclusions = self.manifest()
        for mutate in (lambda m: m.update(extra=True),
                       lambda m: m["cases"].append(copy.deepcopy(m["cases"][0])),
                       lambda m: m["cases"][0].update(document_sha256="not-a-hash"),
                       lambda m: m["cases"][1].update(split="training", request_sha256=m["cases"][0]["request_sha256"]),
                       lambda m: m["sources"][0].update(location="../outside")):
            manifest = copy.deepcopy(original)
            mutate(manifest)
            with self.assertRaises(ValueError):
                validate(manifest, exclusions)
