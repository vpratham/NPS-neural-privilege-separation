"""Deterministic state-space telemetry for neural trajectories.

This module detects deviations from a benign calibration distribution.  An
alarm is an anomaly indicator only; it does not assign intent or attack type.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any


SCHEMA_VERSION = 1


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _positive_number(value: Any, name: str) -> float:
    value = _finite_number(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_trace(trace: Any, feature_dim: int | None = None) -> list[list[float]]:
    if not isinstance(trace, list) or len(trace) < 2:
        raise ValueError("each trajectory must contain at least two vectors")
    result: list[list[float]] = []
    for row_index, vector in enumerate(trace):
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"trajectory vector {row_index} must be a non-empty list")
        if feature_dim is not None and len(vector) != feature_dim:
            raise ValueError("all feature vectors must have the same dimension")
        result.append([_finite_number(value, "feature") for value in vector])
    if feature_dim is None:
        feature_dim = len(result[0])
        if any(len(vector) != feature_dim for vector in result):
            raise ValueError("all feature vectors must have the same dimension")
    return result


def _quantile(values: list[float], quantile: float) -> float:
    """Linear interpolation quantile with deterministic endpoints."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _profile_values(profile: Any) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise ValueError("profile must be a dictionary")
    required = {
        "schema_version", "identity", "policy_sha256", "feature_dim",
        "initial_mean", "initial_scale", "ar", "intercept",
        "residual_scale", "drift", "threshold", "calibration",
    }
    missing = required.difference(profile)
    if missing:
        raise ValueError(f"profile is missing fields: {', '.join(sorted(missing))}")
    unknown = set(profile).difference(required)
    if unknown:
        raise ValueError(f"profile has unknown fields: {', '.join(sorted(unknown))}")
    if (isinstance(profile["schema_version"], bool)
            or not isinstance(profile["schema_version"], int)
            or profile["schema_version"] != SCHEMA_VERSION):
        raise ValueError("unsupported profile schema version")
    _validate_identity(profile["identity"])
    if (not isinstance(profile["policy_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", profile["policy_sha256"])):
        raise ValueError("policy_sha256 must be a lowercase SHA-256 hex digest")
    dim = profile["feature_dim"]
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("feature_dim must be a positive integer")
    for name in ("initial_mean", "initial_scale", "ar", "intercept", "residual_scale"):
        values = profile[name]
        if not isinstance(values, list) or len(values) != dim:
            raise ValueError(f"{name} must have feature_dim entries")
        profile[name] = [_finite_number(value, name) for value in values]
    for name in ("initial_scale", "residual_scale"):
        if any(value <= 0.0 for value in profile[name]):
            raise ValueError(f"{name} entries must be positive")
    profile["drift"] = _finite_number(profile["drift"], "drift")
    if profile["drift"] < 0.0:
        raise ValueError("drift must be non-negative")
    profile["threshold"] = _positive_number(profile["threshold"], "threshold")
    _validate_calibration(profile["calibration"])
    return profile


def _validate_identity(identity: Any) -> None:
    if not isinstance(identity, dict) or not identity:
        raise ValueError("identity must be a non-empty dictionary")
    try:
        json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("identity must be finite JSON data") from error


def _validate_calibration(calibration: Any) -> None:
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be a dictionary")
    required = {
        "training_trajectory_count", "training_vector_count", "trajectory_count",
        "vector_count", "quantile", "base_threshold", "conservative_margin",
        "observed_false_alarm_count",
    }
    if set(calibration) != required:
        raise ValueError("calibration fields do not match schema")
    for name in ("training_trajectory_count", "training_vector_count", "trajectory_count", "vector_count", "observed_false_alarm_count"):
        value = calibration[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"calibration {name} must be a non-negative integer")
    if calibration["training_trajectory_count"] < 2 or calibration["trajectory_count"] < 2:
        raise ValueError("calibration requires at least two trajectories per split")
    if calibration["training_vector_count"] < 2 * calibration["training_trajectory_count"]:
        raise ValueError("training vector count is inconsistent")
    if calibration["vector_count"] < 2 * calibration["trajectory_count"]:
        raise ValueError("calibration vector count is inconsistent")
    if calibration["observed_false_alarm_count"] > calibration["trajectory_count"]:
        raise ValueError("false alarm count is inconsistent")
    quantile = _finite_number(calibration["quantile"], "calibration quantile")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("calibration quantile must be between zero and one")
    if _finite_number(calibration["base_threshold"], "calibration base_threshold") < 0.0:
        raise ValueError("calibration base_threshold must be non-negative")
    _positive_number(calibration["conservative_margin"], "calibration conservative_margin")


class Observer:
    """Online diagonal linear observer with a one-sided CUSUM statistic."""

    def __init__(self, profile: dict[str, Any]) -> None:
        # Copy only the values used at inference so a caller cannot mutate them.
        checked = _profile_values(dict(profile))
        self.profile = checked
        self.feature_dim = checked["feature_dim"]
        self.reset()

    def reset(self) -> None:
        """Start a new trajectory; observations never cross trace boundaries."""
        self._previous: list[float] | None = None
        self._cusum = 0.0
        self._step = 0
        self._alarm = False

    def _failed_step(self, phase: str) -> dict[str, Any]:
        self._step += 1
        self._alarm = True
        return {
            "step": self._step,
            "innovation_energy": None,
            "max_standardized_residual": None,
            "cusum": None,
            "threshold": self.profile["threshold"],
            "alarm": True,
            "phase": phase,
            "sensor_error": True,
        }

    def step(self, features: list[float]) -> dict[str, Any]:
        """Observe one projected feature vector and fail closed on bad input."""
        phase = "initial" if self._previous is None else "transition"
        try:
            if not isinstance(features, list) or len(features) != self.feature_dim:
                raise ValueError("feature vector has wrong shape")
            current = [_finite_number(value, "feature") for value in features]
        except ValueError:
            return self._failed_step(phase)

        if self._previous is None:
            center = self.profile["initial_mean"]
            scale = self.profile["initial_scale"]
        else:
            center = [
                self.profile["ar"][index] * self._previous[index]
                + self.profile["intercept"][index]
                for index in range(self.feature_dim)
            ]
            scale = self.profile["residual_scale"]
        try:
            standardized = [(current[index] - center[index]) / scale[index]
                            for index in range(self.feature_dim)]
            squared = [value * value for value in standardized]
            energy = sum(squared) / self.feature_dim
            maximum = max(abs(value) for value in standardized)
            next_cusum = max(0.0, self._cusum + energy - self.profile["drift"])
            if not all(math.isfinite(value) for value in standardized) or not all(
                    math.isfinite(value) for value in (energy, maximum, next_cusum)):
                raise ValueError("non-finite observer arithmetic")
        except (ArithmeticError, ValueError):
            return self._failed_step(phase)
        self._cusum = next_cusum
        self._previous = current
        self._step += 1
        self._alarm = self._alarm or self._cusum > self.profile["threshold"]
        return {
            "step": self._step,
            "innovation_energy": energy,
            "max_standardized_residual": maximum,
            "cusum": self._cusum,
            "threshold": self.profile["threshold"],
            "alarm": self._alarm,
            "phase": phase,
            "sensor_error": False,
        }


def _fit_profile(
    training: list[list[list[float]]],
    calibration: list[list[list[float]]],
    *,
    identity: dict,
    policy_sha256: str,
    drift: float = 1.0,
    quantile: float = 1.0,
    variance_floor: float = 1e-6,
) -> dict[str, Any]:
    """Fit a benign-only diagonal observer profile.

    Training estimates dynamics.  Held-out calibration alone sets the CUSUM
    threshold, preventing trajectories from being accidentally joined.
    """
    _validate_identity(identity)
    if (not isinstance(policy_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", policy_sha256)):
        raise ValueError("policy_sha256 must be a lowercase SHA-256 hex digest")
    drift = _finite_number(drift, "drift")
    if drift < 0.0:
        raise ValueError("drift must be non-negative")
    quantile = _finite_number(quantile, "quantile")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    variance_floor = _positive_number(variance_floor, "variance_floor")
    scale_floor = math.sqrt(variance_floor)
    if not isinstance(training, list) or len(training) < 2:
        raise ValueError("at least two training trajectories are required")
    if not isinstance(calibration, list) or len(calibration) < 2:
        raise ValueError("at least two calibration trajectories are required")

    train = [_validate_trace(trace) for trace in training]
    feature_dim = len(train[0][0])
    if any(len(trace[0]) != feature_dim for trace in train):
        raise ValueError("all feature vectors must have the same dimension")
    train = [_validate_trace(trace, feature_dim) for trace in train]
    calibrated = [_validate_trace(trace, feature_dim) for trace in calibration]

    starts = [[trace[0][index] for trace in train] for index in range(feature_dim)]
    initial_mean = [sum(values) / len(values) for values in starts]
    initial_scale = [max(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)), scale_floor)
                     for values, mean in zip(starts, initial_mean)]

    ar: list[float] = []
    intercept: list[float] = []
    residual_scale: list[float] = []
    for index in range(feature_dim):
        xs = [trace[position][index] for trace in train for position in range(len(trace) - 1)]
        ys = [trace[position + 1][index] for trace in train for position in range(len(trace) - 1)]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denominator = sum((value - x_mean) ** 2 for value in xs)
        slope = (sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
                 if denominator > variance_floor * len(xs) else 0.0)
        bias = y_mean - slope * x_mean
        residuals = [y - (slope * x + bias) for x, y in zip(xs, ys)]
        ar.append(slope)
        intercept.append(bias)
        residual_scale.append(max(math.sqrt(sum(value * value for value in residuals) / len(residuals)), scale_floor))

    seed_profile: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "identity": dict(identity),
        "policy_sha256": policy_sha256,
        "feature_dim": feature_dim,
        "initial_mean": initial_mean,
        "initial_scale": initial_scale,
        "ar": ar,
        "intercept": intercept,
        "residual_scale": residual_scale,
        "drift": drift,
        "threshold": variance_floor,
        "calibration": {
            "training_trajectory_count": len(train),
            "training_vector_count": sum(len(trace) for trace in train),
            "trajectory_count": len(calibrated),
            "vector_count": sum(len(trace) for trace in calibrated),
            "quantile": quantile,
            "base_threshold": 0.0,
            "conservative_margin": variance_floor,
            "observed_false_alarm_count": 0,
        },
    }
    observer = Observer(seed_profile)
    maxima: list[float] = []
    for trace in calibrated:
        observer.reset()
        scores = [observer.step(vector) for vector in trace]
        if any(score["sensor_error"] for score in scores):
            raise ValueError("non-finite arithmetic while calibrating profile")
        maxima.append(max(score["cusum"] for score in scores))
    base_threshold = _quantile(maxima, quantile)
    margin = max(variance_floor, abs(base_threshold) * 1e-12)
    threshold = base_threshold + margin
    seed_profile["threshold"] = threshold
    observer = Observer(seed_profile)
    false_alarms = 0
    for trace in calibrated:
        observer.reset()
        if any(observer.step(vector)["alarm"] for vector in trace):
            false_alarms += 1
    seed_profile["calibration"] = {
        "training_trajectory_count": len(train),
        "training_vector_count": sum(len(trace) for trace in train),
        "trajectory_count": len(calibrated),
        "vector_count": sum(len(trace) for trace in calibrated),
        "quantile": quantile,
        "base_threshold": base_threshold,
        "conservative_margin": margin,
        "observed_false_alarm_count": false_alarms,
    }
    return seed_profile


def fit_profile(
    training: list[list[list[float]]],
    calibration: list[list[list[float]]],
    *,
    identity: dict,
    policy_sha256: str,
    drift: float = 1.0,
    quantile: float = 1.0,
    variance_floor: float = 1e-6,
) -> dict[str, Any]:
    """Fit a profile, converting numerical overflow into a safe failure."""
    try:
        return _fit_profile(
            training,
            calibration,
            identity=identity,
            policy_sha256=policy_sha256,
            drift=drift,
            quantile=quantile,
            variance_floor=variance_floor,
        )
    except ArithmeticError as error:
        raise ValueError("non-finite arithmetic while fitting profile") from error
