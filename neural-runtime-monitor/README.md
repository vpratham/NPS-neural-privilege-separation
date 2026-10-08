# Neural runtime policy firewall

The primary deterministic capability boundary now lives in [`../permission-boundary/`](../permission-boundary/), where read grants, disclosure grants and action proposals are attached to a host-owned per-instance profile. The implementation here is the earlier prototype; use it for its historical tests, not as the canonical application integration.

The prototype is in [`neural_state_firewall/capability.py`](neural_state_firewall/capability.py). It accepts exact quotes from host-supplied evidence, fixed host-generated refusals, or allowlisted typed action proposals. Arbitrary prose is rejected. Action proposals are never executed here; the calling application must authorize them through its trusted broker. See the permission-boundary [capability contract](../permission-boundary/neural_state_firewall/CAPABILITY_BOUNDARY.md) for the current design.

Run the offline check:

```sh
python -m neural_state_firewall capability-demo
```

`CapabilityFirewall` uses a small `ModelProvider` protocol, so integrations can supply different model APIs without changing the deterministic gate. This folder currently has no live provider adapter; the demo uses scripted responses and makes no model-efficacy claim. The exact-quote answer mode is intentionally limited and does not certify that the model understood an instruction or that quoted evidence is benign.

The older activation-trajectory monitor remains available for research under [`neural_state_firewall/runtime.py`](neural_state_firewall/runtime.py). Its anomaly scores are not an authorization boundary and should remain in observation mode. The permission boundary that controls which documents may be read is maintained separately in [`../permission-boundary/README.md`](../permission-boundary/README.md).
