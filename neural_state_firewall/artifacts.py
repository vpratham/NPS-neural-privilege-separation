"""Strict capture/profile artifacts and disjoint calibration checks.

Artifacts are host-owned calibration inputs, not signed attestations. Their
schema and hashes catch accidental mixing or corruption; filesystem access
control remains the authority boundary.
"""
import hashlib
import json
import math
import re
from pathlib import Path

from .observer import Observer, fit_profile
from .runtime import Step, policy_digest


CAPTURE_KIND = "neural-state-capture-v1"
PROFILE_KIND = "neural-state-profile-v1"
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_CAPTURE_KEYS = {"kind", "identity", "policy_sha256", "max_new_tokens", "traces"}
_TRACE_KEYS = {"request_sha256", "vectors", "ended_with_eos"}
_PROFILE_ARTIFACT_KEYS = {"kind", "profile", "max_new_tokens", "training_sha256", "calibration_sha256"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def loads(text):
    def invalid(value):
        raise ValueError(f"Invalid JSON number: {value}")
    return json.loads(text, object_pairs_hook=_unique, parse_constant=invalid)


def read(path):
    return loads(Path(path).read_text())


def write(path, value):
    # Refuse overwrite to preserve fitted profile/capture provenance.
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _hash(value, name):
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _identity(identity):
    if not isinstance(identity, dict) or not identity:
        raise ValueError("capture identity must be a non-empty dictionary")
    try:
        json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("capture identity must be finite JSON data") from error
    dim = identity.get("feature_dim")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError("capture identity feature_dim must be a positive integer")
    return dim


def _horizon(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= 32768:
        raise ValueError(f"{label} must be an integer between 2 and 32768")
    return value


def _vectors(vectors, feature_dim, horizon):
    if not isinstance(vectors, list) or not 2 <= len(vectors) <= horizon:
        raise ValueError("trace vectors must contain between two and max_new_tokens observations")
    result = []
    for index, vector in enumerate(vectors):
        if not isinstance(vector, list) or len(vector) != feature_dim:
            raise ValueError(f"trace vector {index} does not match identity feature_dim")
        result.append([_finite_number(item, "trace feature") for item in vector])
    return result


def _validate_capture(value, *, require_two_traces=False):
    if not isinstance(value, dict) or set(value) != _CAPTURE_KEYS or value.get("kind") != CAPTURE_KIND:
        raise ValueError("Expected strict neural-state capture artifact")
    feature_dim = _identity(value["identity"])
    _hash(value["policy_sha256"], "policy_sha256")
    horizon = _horizon(value["max_new_tokens"], "capture horizon")
    traces = value["traces"]
    if not isinstance(traces, list) or (require_two_traces and len(traces) < 2):
        raise ValueError("At least two traces per split required" if require_two_traces else "capture traces must be a list")
    for trace in traces:
        if not isinstance(trace, dict) or set(trace) != _TRACE_KEYS:
            raise ValueError("trace fields do not match capture schema")
        _hash(trace["request_sha256"], "request_sha256")
        _vectors(trace["vectors"], feature_dim, horizon)
        if trace["ended_with_eos"] is not True:
            raise ValueError("calibration traces must end with EOS")
    return value


def requests(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        item = loads(line)
        if not isinstance(item, dict) or set(item) - {"task", "context", "label"}:
            raise ValueError("Each request allows only task, context and label")
        if not isinstance(item.get("task"), str) or not item["task"].strip() or not isinstance(item.get("context", ""), str):
            raise ValueError("Invalid task/context")
        result.append(item)
    if not result:
        raise ValueError("Request file is empty")
    return result


def capture(adapter, items, max_new_tokens):
    """Capture complete bounded benign trajectories, closing every source stream."""
    identity = getattr(adapter, "identity", None)
    feature_dim = _identity(identity)
    horizon = _horizon(max_new_tokens, "capture horizon")
    policy = getattr(adapter, "policy", None)
    if not isinstance(policy, str):
        raise ValueError("adapter policy must be text")
    if not isinstance(items, list):
        raise ValueError("capture items must be a list")
    traces = []
    for item in items:
        if (not isinstance(item, dict) or set(item) - {"task", "context", "label"}
                or item.get("label") != "benign" or not isinstance(item.get("task"), str)
                or not item["task"].strip() or not isinstance(item.get("context", ""), str)):
            raise ValueError("Calibration capture requires explicitly labelled benign requests")
        prompt = {"task": item["task"], "context": item.get("context", "")}
        stream = None
        frames = []
        try:
            stream = iter(adapter.iter_steps(**prompt, max_new_tokens=horizon))
            for index, frame in enumerate(stream):
                if index >= horizon:
                    raise ValueError("adapter exceeded capture horizon")
                if (not isinstance(frame, Step) or type(frame.token_id) is not int or frame.token_id < 0
                        or type(frame.is_eos) is not bool):
                    raise ValueError("invalid adapter frame")
                _vectors([frame.features, frame.features], feature_dim, horizon)
                if frames and frames[-1].is_eos:
                    raise ValueError("EOS must be the terminal capture frame")
                frames.append(frame)
        finally:
            if stream is not None and callable(getattr(stream, "close", None)):
                stream.close()
        if not 2 <= len(frames) <= horizon or not frames[-1].is_eos:
            raise ValueError("Each calibration trace must contain two or more frames and terminate with EOS")
        traces.append({"request_sha256": digest(prompt), "vectors": [list(frame.features) for frame in frames],
                       "ended_with_eos": True})
    result = {"kind": CAPTURE_KIND, "identity": identity,
              "policy_sha256": policy_digest(policy), "max_new_tokens": horizon, "traces": traces}
    return _validate_capture(result)


def fit(training, calibration, *, quantile=1.0):
    _validate_capture(training, require_two_traces=True)
    _validate_capture(calibration, require_two_traces=True)
    for key in ("identity", "policy_sha256", "max_new_tokens"):
        if training[key] != calibration[key]:
            raise ValueError(f"Capture mismatch: {key}")
    seen_requests, seen_vectors = set(), set()
    for value in (training, calibration):
        for trace in value["traces"]:
            request_hash = trace["request_sha256"]
            vector_hash = digest(trace["vectors"])
            if request_hash in seen_requests or vector_hash in seen_vectors:
                raise ValueError("Duplicate request/trajectory: training and calibration must be disjoint")
            seen_requests.add(request_hash)
            seen_vectors.add(vector_hash)
    profile = fit_profile([trace["vectors"] for trace in training["traces"]],
                          [trace["vectors"] for trace in calibration["traces"]],
                          identity=training["identity"], policy_sha256=training["policy_sha256"], quantile=quantile)
    return {"kind": PROFILE_KIND, "profile": profile,
            "max_new_tokens": training["max_new_tokens"],
            "training_sha256": digest(training), "calibration_sha256": digest(calibration)}


def validate_profile(value):
    if not isinstance(value, dict) or set(value) != _PROFILE_ARTIFACT_KEYS or value.get("kind") != PROFILE_KIND:
        raise ValueError("Expected strict neural-state profile artifact")
    _horizon(value["max_new_tokens"], "fitted horizon")
    _hash(value["training_sha256"], "training_sha256")
    _hash(value["calibration_sha256"], "calibration_sha256")
    if value["training_sha256"] == value["calibration_sha256"]:
        raise ValueError("training and calibration digests must differ")
    Observer(value["profile"])
    return value
