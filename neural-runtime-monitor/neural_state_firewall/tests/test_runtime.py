"""Fail-closed release-gate tests using a deterministic sensor adapter."""

from __future__ import annotations

import copy
import unittest

from neural_state_firewall.observer import fit_profile
from neural_state_firewall.runtime import Firewall, Step, policy_digest


TRAIN = [
    [[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]],
    [[0.2, 0.9], [0.3, 1.0], [0.4, 1.1]],
    [[-0.1, 1.1], [0.0, 1.2], [0.1, 1.3]],
]
CALIBRATION = [
    [[0.05, 1.0], [0.15, 1.1], [0.25, 1.2]],
    [[0.15, 0.95], [0.25, 1.05], [0.35, 1.15]],
]


class StubAdapter:
    def __init__(self, frames=(), *, identity=None, policy="trusted host policy", fail_after=None,
                 decode_failure=False):
        self.identity = identity or {"model": "runtime-fixture", "feature_dim": 2}
        self.policy = policy
        self.frames = list(frames)
        self.fail_after = fail_after
        self.decode_failure = decode_failure
        self.decode_calls = []
        self.closed = False

    def iter_steps(self, task, context, max_new_tokens):
        try:
            for index, frame in enumerate(self.frames):
                if self.fail_after == index:
                    raise RuntimeError("provider includes secret prompt text")
                yield frame
            if self.fail_after == len(self.frames):
                raise RuntimeError("provider failed after tokens")
        finally:
            self.closed = True

    def decode(self, token_ids):
        self.decode_calls.append(list(token_ids))
        if self.decode_failure:
            raise RuntimeError("decode failure")
        return "/".join(str(token) for token in token_ids)


class CloseFailureAdapter(StubAdapter):
    def iter_steps(self, task, context, max_new_tokens):
        adapter = self

        class Stream:
            def __init__(self):
                self._frames = iter(adapter.frames)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._frames)

            def close(self):
                adapter.closed = True
                raise RuntimeError("provider cleanup failure")

        return Stream()


class RuntimeTests(unittest.TestCase):
    def test_deployment_artifact_locks_generation_horizon(self):
        adapter = StubAdapter([Step([0.05, 1.0], 2, True)])
        artifact = {"kind": "neural-state-profile-v1", "profile": self.profile(),
                    "max_new_tokens": 3, "training_sha256": "a" * 64,
                    "calibration_sha256": "b" * 64}
        firewall = Firewall.from_artifact(adapter, artifact)
        self.assertEqual(firewall.mode, "monitor")
        self.assertEqual(firewall.run("task")["status"], "monitored")
        result = firewall.run("task", max_new_tokens=4)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["output"])

    def profile(self):
        identity = {"model": "runtime-fixture", "feature_dim": 2}
        return fit_profile(TRAIN, CALIBRATION, identity=identity,
                           policy_sha256=policy_digest("trusted host policy"),
                           drift=1.0, variance_floor=0.01)

    def firewall(self, frames=(), **adapter_kwargs):
        return Firewall(StubAdapter(frames, **adapter_kwargs), self.profile(), mode="enforce")

    def test_default_trajectory_monitor_does_not_enforce_anomaly_alarms(self):
        adapter = StubAdapter([Step([50.0, -50.0], 9, False), Step([0.25, 1.2], 2, True)])
        result = Firewall(adapter, self.profile()).run("safe task")
        self.assertEqual(result["status"], "monitored")
        self.assertTrue(result["alarm_observed"])
        self.assertEqual(result["output"], "9")

    def test_midstream_alarm_releases_no_partial_output(self):
        firewall = self.firewall([
            Step([0.05, 1.0], 9, False),
            Step([50.0, -50.0], 10, False),
            Step([0.25, 1.2], 2, True),
        ])
        result = firewall.run("safe task")
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "state_anomaly")
        self.assertIsNone(result["output"])
        self.assertEqual(firewall.adapter.decode_calls, [])
        self.assertTrue(firewall.adapter.closed)
        self.assertEqual(result["observed_steps"], 2)

    def test_invalid_telemetry_fails_closed_even_in_monitor_mode(self):
        adapter = StubAdapter([Step([float("nan"), 1.0], 2, True)])
        result = Firewall(adapter, self.profile(), mode="monitor").run("safe task")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], "invalid_telemetry")
        self.assertIsNone(result["output"])
        self.assertEqual(adapter.decode_calls, [])
        self.assertTrue(adapter.closed)
        self.assertTrue(result["events"][0]["sensor_error"])
        self.assertIsNone(result["events"][0]["cusum"])

    def test_monitor_mode_records_anomaly_but_releases_completed_buffer(self):
        adapter = StubAdapter([
            Step([50.0, -50.0], 9, False),
            Step([0.25, 1.2], 2, True),
        ])
        result = Firewall(adapter, self.profile(), mode="monitor").run("safe task")
        self.assertEqual(result["status"], "monitored")
        self.assertEqual(result["output"], "9")
        self.assertTrue(result["alarm_observed"])
        self.assertEqual(adapter.decode_calls, [[9]])

    def test_provider_exception_after_tokens_releases_nothing(self):
        adapter = StubAdapter([Step([0.05, 1.0], 9, False)], fail_after=1)
        result = Firewall(adapter, self.profile()).run("safe task")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], "runtime_error")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertIsNone(result["output"])
        self.assertEqual(adapter.decode_calls, [])
        self.assertTrue(adapter.closed)

    def test_no_eos_and_budget_overrun_release_nothing(self):
        no_eos = self.firewall([Step([0.05, 1.0], 9, False)])
        result = no_eos.run("safe task")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reason"], "generation_ended_without_eos")
        self.assertIsNone(result["output"])
        self.assertEqual(no_eos.adapter.decode_calls, [])

        over_budget = self.firewall([Step([0.05, 1.0], 9, False), Step([0.15, 1.1], 2, True)])
        result = over_budget.run("safe task", max_new_tokens=1)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertIsNone(result["output"])
        self.assertEqual(over_budget.adapter.decode_calls, [])

    def test_decode_and_cleanup_failure_fail_closed(self):
        decode_failure = StubAdapter([Step([0.05, 1.0], 2, True)], decode_failure=True)
        result = Firewall(decode_failure, self.profile()).run("safe task")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertIsNone(result["output"])
        self.assertTrue(decode_failure.closed)

        cleanup_failure = CloseFailureAdapter([Step([0.05, 1.0], 2, True)])
        result = Firewall(cleanup_failure, self.profile()).run("safe task")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["reason"], "cleanup_error")
        self.assertEqual(result["error_type"], "RuntimeError")
        self.assertIsNone(result["output"])
        self.assertTrue(cleanup_failure.closed)

    def test_profile_binding_and_mutation_are_checked_before_generation(self):
        profile = self.profile()
        with self.assertRaises(ValueError):
            Firewall(StubAdapter(identity={"model": "other", "feature_dim": 2}), profile)
        with self.assertRaises(ValueError):
            Firewall(StubAdapter(policy="other host policy"), profile)

        original = self.profile()
        firewall = Firewall(StubAdapter([Step([0.05, 1.0], 2, True)]), original, mode="monitor")
        original["identity"]["model"] = "mutated-by-caller"
        result = firewall.run("safe task")
        self.assertEqual(result["status"], "monitored")
        self.assertEqual(result["output"], "")
        self.assertTrue(firewall.adapter.closed)

    def test_adapter_identity_mutation_is_detected_before_generation(self):
        firewall = self.firewall([Step([0.05, 1.0], 2, True)])
        firewall.adapter.identity = copy.deepcopy(firewall.adapter.identity)
        firewall.adapter.identity["model"] = "changed weights"
        result = firewall.run("safe task")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "ValueError")
        self.assertFalse(firewall.adapter.closed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
