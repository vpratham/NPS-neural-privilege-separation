"""Deterministic synthetic integration demo; not an attack benchmark."""
from .observer import fit_profile
from .runtime import Firewall, Step, policy_digest


class ReplayAdapter:
    identity = {"adapter": "synthetic-replay-v1", "feature_dim": 2}
    policy = "Demonstrate response buffering and telemetry enforcement."

    def iter_steps(self, task, context, max_new_tokens):
        del context
        frames = [[0.1, 0.2], [0.2, 0.1], [0.1, 0.2]]
        if task == "anomaly":
            frames[1] = [100.0, -100.0]
        if task == "sensor_failure":
            frames[1] = [float("nan"), 0.0]
        for index, frame in enumerate(frames[:max_new_tokens]):
            yield Step(frame, index + 1, index == 2)

    def decode(self, tokens):
        return "Buffered demo response." if tokens else ""


def demo():
    adapter = ReplayAdapter()
    profile = fit_profile(
        [[[0., 0.], [0.2, 0.1], [0.1, 0.2]], [[0.2, 0.2], [0., 0.3], [0.3, 0.]]],
        [[[0.1, 0.2], [0.2, 0.1], [0.1, 0.2]], [[0.15, 0.15], [0.1, 0.15], [0.2, 0.1]]],
        identity=adapter.identity, policy_sha256=policy_digest(adapter.policy))
    firewall = Firewall(adapter, profile, mode="enforce")
    cases = {task: firewall.run(task, max_new_tokens=3) for task in ("benign", "anomaly", "sensor_failure")}
    assert cases["benign"]["status"] == "allowed"
    assert cases["anomaly"]["status"] == "blocked" and cases["anomaly"]["output"] is None
    assert cases["sensor_failure"]["status"] == "error" and cases["sensor_failure"]["output"] is None
    return {"demo_type": "synthetic_mechanics_only", "prompt_injection_accuracy_measured": False,
            "cases": cases}
