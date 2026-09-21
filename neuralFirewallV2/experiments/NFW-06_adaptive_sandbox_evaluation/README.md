# NFW-006 — Adaptive red-team and sandboxed side-effect evaluation

NFW-006 follows NFW-005 by testing the stronger endpoint: whether an adaptive attacker can cause an unauthorized side effect. It uses deterministic, in-memory mock tools and a default-deny capability broker. No real filesystem, shell, network, API, or external service is accessed.

## Coverage

- Strict JSON wire parsing and authority-field rejection
- Resource- and action-scoped capabilities
- Expiry, replay, revocation, and signature checks
- Adaptive attacks that observe denial reasons and mutate subsequent requests
- Leaked low-privilege token scope-confusion and replay attempts
- In-memory write and message side effects
- Authorized control cases kept separate from attack outcomes
- Chained audit telemetry
- Drive-backed immutable manifests and resumable checkpoints
- Separate authorized benign control path for utility measurement

The primary security endpoint is `unauthorized_side_effects == 0`. A request being authorized is not itself a side effect-safety result; the notebook records both decisions and sandbox state changes.

This is a reference PoC, not production isolation. A production broker still requires a separately isolated service/process, protected key custody, authenticated principals, resource policy, TOCTOU defenses, and independent security review.
