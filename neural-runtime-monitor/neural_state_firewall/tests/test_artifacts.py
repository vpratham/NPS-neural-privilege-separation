"""Provenance and strict-artifact tests for state-space calibration."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from neural_state_firewall import artifacts
from neural_state_firewall.runtime import Step


class CaptureAdapter:
    identity = {"model": "capture-fixture", "feature_dim": 2}
    policy = "trusted capture policy"

    def iter_steps(self, task, context, max_new_tokens):
        offset = sum((index + 1) * ord(char) for index, char in enumerate(task + context)) / 100_000.0
        yield Step([offset, 1.0 + offset], 9, False)
        yield Step([0.1 + offset, 1.1 + offset], 2, True)


class BadCaptureAdapter(CaptureAdapter):
    def __init__(self, frames):
        self.frames = frames
        self.closed = False

    def iter_steps(self, task, context, max_new_tokens):
        try:
            yield from self.frames
        finally:
            self.closed = True


def capture_for(items):
    return artifacts.capture(CaptureAdapter(), items, max_new_tokens=4)


TRAIN_ITEMS = [
    {"task": "summarize alpha", "context": "one", "label": "benign"},
    {"task": "summarize beta", "context": "two", "label": "benign"},
]
CALIBRATION_ITEMS = [
    {"task": "summarize gamma", "context": "three", "label": "benign"},
    {"task": "summarize delta", "context": "four", "label": "benign"},
]


class ArtifactTests(unittest.TestCase):
    def test_capture_and_fit_preserve_disjoint_provenance(self):
        training = capture_for(TRAIN_ITEMS)
        calibration = capture_for(CALIBRATION_ITEMS)
        fitted = artifacts.fit(training, calibration)
        self.assertEqual(fitted["kind"], "neural-state-profile-v1")
        self.assertEqual(fitted["profile"]["identity"], CaptureAdapter.identity)
        self.assertEqual(fitted["profile"]["calibration"]["training_trajectory_count"], 2)
        self.assertEqual(fitted["profile"]["calibration"]["trajectory_count"], 2)
        self.assertEqual(artifacts.validate_profile(fitted), fitted)

    def test_fit_rejects_shared_request_or_trajectory(self):
        training = capture_for(TRAIN_ITEMS)
        calibration = capture_for(CALIBRATION_ITEMS)
        calibration["traces"][0]["request_sha256"] = training["traces"][0]["request_sha256"]
        with self.assertRaisesRegex(ValueError, "Duplicate request/trajectory"):
            artifacts.fit(training, calibration)

        calibration = capture_for(CALIBRATION_ITEMS)
        calibration["traces"][0]["vectors"] = copy.deepcopy(training["traces"][0]["vectors"])
        with self.assertRaisesRegex(ValueError, "Duplicate request/trajectory"):
            artifacts.fit(training, calibration)

    def test_fit_rejects_model_policy_and_horizon_mismatch(self):
        training = capture_for(TRAIN_ITEMS)
        calibration = capture_for(CALIBRATION_ITEMS)
        for key, changed in (
            ("identity", {"model": "other", "feature_dim": 2}),
            ("policy_sha256", "f" * 64),
            ("max_new_tokens", 5),
        ):
            broken = copy.deepcopy(calibration)
            broken[key] = changed
            with self.assertRaisesRegex(ValueError, "Capture mismatch"):
                artifacts.fit(training, broken)

    def test_fit_rejects_malformed_trace_and_capture_schema(self):
        training = capture_for(TRAIN_ITEMS)
        calibration = capture_for(CALIBRATION_ITEMS)
        for mutate in (
            lambda item: item["traces"][0].__setitem__("ended_with_eos", False),
            lambda item: item["traces"][0].pop("ended_with_eos"),
            lambda item: item["traces"][0].__setitem__("vectors", [[0.0], [0.1]]),
            lambda item: item.__setitem__("unexpected", "field"),
            lambda item: item.__setitem__("policy_sha256", "not-a-hash"),
        ):
            broken = copy.deepcopy(training)
            mutate(broken)
            with self.assertRaises(ValueError):
                artifacts.fit(broken, calibration)

    def test_capture_rejects_incomplete_or_nonterminal_eos_and_closes_stream(self):
        for frames in (
            [Step([0.0, 1.0], 9, False), Step([0.1, 1.1], 10, False)],
            [Step([0.0, 1.0], 2, True), Step([0.1, 1.1], 10, False)],
            [Step([0.0, 1.0], 9, False), object()],
        ):
            adapter = BadCaptureAdapter(frames)
            with self.assertRaises(ValueError):
                artifacts.capture(adapter, TRAIN_ITEMS[:1], max_new_tokens=4)
            self.assertTrue(adapter.closed)

    def test_validate_profile_requires_exact_schema_and_distinct_hashes(self):
        fitted = artifacts.fit(capture_for(TRAIN_ITEMS), capture_for(CALIBRATION_ITEMS))
        for mutate in (
            lambda item: item.__setitem__("unexpected", "field"),
            lambda item: item.__setitem__("training_sha256", "bad"),
            lambda item: item.__setitem__("calibration_sha256", item["training_sha256"]),
            lambda item: item.pop("profile"),
        ):
            broken = copy.deepcopy(fitted)
            mutate(broken)
            with self.assertRaises(ValueError):
                artifacts.validate_profile(broken)

    def test_strict_json_rejects_nan_and_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "Invalid JSON number"):
            artifacts.loads('{"value": NaN}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            artifacts.loads('{"value": 1, "value": 2}')

    def test_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "profile.json"
            artifacts.write(target, {"one": 1})
            self.assertEqual(artifacts.read(target), {"one": 1})
            with self.assertRaises(FileExistsError):
                artifacts.write(target, {"two": 2})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
