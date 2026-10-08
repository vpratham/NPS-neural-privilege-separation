# Neural runtime policy firewall

This workstream is pivoting from activation-anomaly blocking toward a deterministic capability boundary. The model proposes a constrained output; trusted host code decides whether it can be released or sent to an action broker.

The first implementation is in [`neural_state_firewall/capability.py`](neural_state_firewall/capability.py). It accepts exact quotes from host-supplied evidence, fixed host-generated refusals, or allowlisted typed action proposals. Arbitrary prose is rejected. Action proposals are never executed here; the calling application must authorize them through its trusted broker. The existing NPS broker is documented in [`../docs/WORKING_SYSTEM.md`](../docs/WORKING_SYSTEM.md).

Run the offline check:

```sh
python -m neural_state_firewall capability-demo
```

`CapabilityFirewall` uses a small `ModelProvider` protocol, so integrations can supply different model APIs without changing the deterministic gate. This folder currently has no live provider adapter; the demo uses scripted responses and makes no model-efficacy claim. The exact-quote answer mode is intentionally limited and does not certify that the model understood an instruction or that quoted evidence is benign.

The older activation-trajectory monitor remains available for research under [`neural_state_firewall/runtime.py`](neural_state_firewall/runtime.py). Its anomaly scores are not an authorization boundary and should remain in observation mode. The permission boundary that controls which documents may be read is maintained separately in [`../permission-boundary/README.md`](../permission-boundary/README.md).
