import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from neural_state_firewall import bipia_analysis


def output(*, eos=True, error=None, text="answer"):
    return {"output": text, "ended_with_eos": eos, "error_type": error,
            "reference_exact": False, "reference_contained": False}


class BIPIAAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.case = {"id": "email-0000-Test-0-start", "task_name": "email", "source_row": 0,
                     "source_context_sha256": "context", "request_sha256": "request", "reference_sha256": "reference",
                     "attack_name": "Test-0", "attack_sha256": "attack", "position": "start", "alternate_attack_name": "Test-1"}
        conditions = {condition: {arm: output() for arm in bipia_analysis.ARMS} for condition in bipia_analysis.CONDITIONS}
        conditions["attack_denied"]["read_permissions_denied_variant"] = {
            "alternate": output(), "same_input_shape": True, "all_step_logits_byte_equal": True,
            "generated_tokens_equal": True, "invariance_passed": True}
        self.report = {"kind": "bipia-local-read-permission-evaluation-v1", "source_sha256": {"x": "y"},
                       "source_revision": "revision", "rows": [{"id": self.case["id"], "task": "email", "source_row": 0,
                       "source_context_sha256": "context", "request_sha256": "request", "reference_sha256": "reference",
                       "attack_name": "Test-0", "attack_sha256": "attack", "position": "start", "alternate_attack_name": "Test-1",
                       "conditions": conditions, "denied_content_invariance": True}]}

    def validate(self, report):
        with patch.object(bipia_analysis, "_expected_cases", return_value={self.case["id"]: self.case}), \
             patch.object(bipia_analysis, "_source_hashes", return_value={"x": "y"}), \
             patch.object(bipia_analysis.subprocess, "check_output", return_value="revision\n"):
            return bipia_analysis.validate_report(report, "/unused")

    def test_rejects_tampered_missing_and_incomplete_records(self):
        self.assertEqual([], self.validate(self.report))
        tampered = copy.deepcopy(self.report)
        tampered["rows"][0]["attack_sha256"] = "changed"
        self.assertTrue(self.validate(tampered))
        missing = copy.deepcopy(self.report)
        del missing["rows"][0]["conditions"]["clean"]["ordinary"]
        self.assertTrue(self.validate(missing))
        incomplete = copy.deepcopy(self.report)
        incomplete["rows"][0]["conditions"]["clean"]["ordinary"]["ended_with_eos"] = False
        summary = bipia_analysis.summarize(incomplete)
        row = next(item for item in summary["attempts"] if item["task"] == "email" and item["condition"] == "clean" and item["arm"] == "ordinary")
        self.assertEqual((row["attempted"], row["incomplete"], row["errors"]), (1, 1, 0))

    def test_regeneration_preserves_existing_review_work(self):
        case = {**self.case, "attack": "attack goal", "request": "question", "reference": "answer"}
        with tempfile.TemporaryDirectory() as directory, patch.object(
                bipia_analysis, "_expected_cases", return_value={case["id"]: case}):
            paths = [Path(directory) / name for name in ("packet.json", "mapping.json", "labels.json")]
            bipia_analysis.create_review_packet(self.report, "/unused", *paths, "raw-hash")
            paths[2].write_text('{"reviewer_notes": "completed review"}\n')
            before = [p.read_bytes() for p in paths]
            with self.assertRaises(FileExistsError):
                bipia_analysis.create_review_packet(self.report, "/unused", *paths, "raw-hash")
            self.assertEqual([p.read_bytes() for p in paths], before)

if __name__ == "__main__":
    unittest.main()
