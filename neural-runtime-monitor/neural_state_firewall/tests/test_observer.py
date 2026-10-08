import copy
import json
import math
import unittest

from neural_state_firewall.observer import Observer, fit_profile


TRAIN = [
    [[0.0, 1.0], [0.1, 1.1], [0.2, 1.2], [0.3, 1.3]],
    [[0.2, 0.9], [0.3, 1.0], [0.4, 1.1], [0.5, 1.2]],
    [[-0.1, 1.1], [0.0, 1.2], [0.1, 1.3], [0.2, 1.4]],
]
CALIBRATION = [
    [[0.05, 1.0], [0.15, 1.1], [0.25, 1.2], [0.35, 1.3]],
    [[0.15, 0.95], [0.25, 1.05], [0.35, 1.15], [0.45, 1.25]],
]


class ObserverTests(unittest.TestCase):
    def profile(self):
        return fit_profile(TRAIN, CALIBRATION, identity={"model": "fixture"}, policy_sha256="a" * 64,
                           drift=1.0, variance_floor=0.01)

    def test_fit_is_deterministic_and_calibration_is_held_out(self):
        first = self.profile()
        second = self.profile()
        self.assertEqual(first, second)
        self.assertEqual(first["calibration"]["training_trajectory_count"], 3)
        self.assertEqual(first["calibration"]["trajectory_count"], 2)
        self.assertGreaterEqual(first["threshold"], first["calibration"]["base_threshold"])
        json.dumps(first, allow_nan=False)

    def test_transition_does_not_cross_trace_boundary(self):
        observer = Observer(self.profile())
        observer.step([100.0, 100.0])
        observer.reset()
        result = observer.step(CALIBRATION[0][0])
        self.assertEqual(result["phase"], "initial")
        self.assertEqual(result["step"], 1)

    def test_sustained_anomaly_triggers_and_reset_clears_cusum(self):
        observer = Observer(self.profile())
        outcomes = [observer.step(vector) for vector in [[5.0, 6.0]] * 4]
        self.assertTrue(outcomes[-1]["alarm"])
        self.assertFalse(outcomes[-1]["sensor_error"])
        self.assertTrue(observer.step([0.0, 1.0])["alarm"])
        observer.reset()
        clean = observer.step(CALIBRATION[0][0])
        self.assertEqual(clean["step"], 1)
        self.assertEqual(clean["phase"], "initial")
        self.assertLess(clean["cusum"], outcomes[-1]["cusum"])
        self.assertFalse(clean["sensor_error"])

    def test_bad_features_fail_closed(self):
        observer = Observer(self.profile())
        for bad in ([], [1.0], [1.0, float("nan")], [1.0, float("inf")], "not-a-vector"):
            observer.reset()
            result = observer.step(bad)
            self.assertTrue(result["alarm"])
            self.assertIsNone(result["cusum"])
            self.assertTrue(result["sensor_error"])
            json.dumps(result, allow_nan=False)

    def test_huge_finite_vector_fails_closed_without_nan_cusum(self):
        observer = Observer(self.profile())
        result = observer.step([1e308, -1e308])
        self.assertTrue(result["alarm"])
        self.assertTrue(result["sensor_error"])
        self.assertIsNone(result["innovation_energy"])

    def test_malformed_profiles_are_rejected(self):
        profile = self.profile()
        for mutate in (
            lambda item: item.pop("ar"),
            lambda item: item.__setitem__("initial_scale", [0.0, 1.0]),
            lambda item: item.__setitem__("residual_scale", [float("nan"), 1.0]),
            lambda item: item.__setitem__("drift", -1.0),
            lambda item: item.__setitem__("threshold", 0.0),
            lambda item: item.__setitem__("unknown", "field"),
            lambda item: item.__setitem__("schema_version", True),
            lambda item: item.__setitem__("calibration", {}),
        ):
            broken = copy.deepcopy(profile)
            mutate(broken)
            with self.assertRaises(ValueError):
                Observer(broken)

    def test_fit_rejects_malformed_identity_hash_and_overflow(self):
        with self.assertRaises(ValueError):
            fit_profile(TRAIN, CALIBRATION, identity={}, policy_sha256="a" * 64)
        with self.assertRaises(ValueError):
            fit_profile(TRAIN, CALIBRATION, identity={"model": float("nan")}, policy_sha256="a" * 64)
        with self.assertRaises(ValueError):
            fit_profile(TRAIN, CALIBRATION, identity={"model": "fixture"}, policy_sha256="A" * 64)
        huge = [
            [[1e308], [-1e308]],
            [[-1e308], [1e308]],
        ]
        with self.assertRaises(ValueError):
            fit_profile(huge, huge, identity={"model": "fixture"}, policy_sha256="a" * 64)


if __name__ == "__main__":
    unittest.main()
