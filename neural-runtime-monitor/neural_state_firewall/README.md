# Runtime policy firewall

`CapabilityFirewall` is the deterministic enforcement path. The host supplies trusted evidence and an allowlist of action names. A model provider returns one strict JSON proposal:

- `quote`: release only an exact substring of a host-supplied evidence source;
- `action`: return an allowlisted proposal to the host for separate broker authorization; or
- `refusal`: return a fixed host-controlled refusal.

Malformed JSON, arbitrary prose, unprovided sources, fabricated quotes, and unallowlisted actions fail closed. This proves only these structural constraints. Exact quote extraction does not establish that the model understood the task or that evidence itself is safe. The gate does not execute tools, and an allowlisted action still requires the host's action broker to validate its caller, arguments, destination, and authorization.

The provider interface is model-agnostic. No live provider adapter is included yet; use `python -m neural_state_firewall capability-demo` to exercise the gate with deterministic scripted responses.

`runtime.Firewall` is the earlier activation-trajectory experiment. Keep it in monitor mode; an anomaly score is not a reliable prompt-injection label or permission decision.
