# Capability-firewall roadmap

## Product direction

The primary security control is a deterministic capability boundary around model outputs and effects. The model proposes a constrained result; host code decides what may be released and whether a proposed effect is authorized. Activation-trajectory scoring is optional diagnostic telemetry, not the security decision.

This workstream is separate from [`permission-boundary`](../../permission-boundary/README.md), which controls which document sources enter the model context. It is also separate from the [NPS action broker](../../docs/WORKING_SYSTEM.md), which authorizes and executes a narrow tool effect. The runtime policy firewall may produce a typed action proposal; it never executes it.

## Implemented now

- `CapabilityFirewall` is model-provider agnostic and uses only the Python standard library.
- It releases an answer only as a quote that exactly occurs in host-supplied evidence.
- It returns an action proposal only when the host has allowlisted that action name; an application must still pass it to the broker for caller, argument, destination, and grant checks.
- It emits a fixed host refusal, rejects free-form prose, and fails closed on malformed output.
- `capability-demo` and security-focused unit tests exercise these rules without downloading a model or performing side effects.

This is a useful enforcement kernel, not yet a production service. No live model API adapter is implemented. Exact quotation is intentionally narrow: it prevents fabricated text from being released through this response mode but does not establish task understanding, benign evidence, or general prompt-injection resistance.

## Next work and acceptance gates

1. **Choose one workload and contract.** Define host identity, evidence provenance, permitted response types, action names, argument schemas, and what the application does on rejection. Success: these are explicit, versioned, host-owned inputs; no request or model field can broaden them.
2. **Connect one live provider.** Implement the provider protocol for one model API and require strict structured output. Success: provider changes do not alter the policy gate; malformed, duplicate-key, oversized, or unsupported responses release no answer or effect.
3. **Integrate the existing action broker.** Pass only typed action proposals to the broker. Success: denied caller/action/argument/destination, replay, and stale grants produce zero side effects; allowed exact actions are audited.
4. **Add an answer mode only when it has a deterministic contract.** Start with exact quotation or host-computed fields. Do not release unrestricted generated prose as “verified.” Success: every released field is mechanically validated against the host contract and its authorized evidence.
5. **Evaluate with frozen cases.** Use matched benign and hostile evidence, source-separated splits, and exact hashes. Measure task usefulness, rejected-output rate, unauthorized effects, latency, and resource use. Success: zero unauthorized effects in the held-out suite and predeclared utility/latency gates pass. Any unauthorized effect is a release blocker.
6. **Qualify operations.** Test authentication, tenant isolation, limits, cancellation, timeouts, concurrency, logging failure, restart, configuration rollback, and model/provider drift. Success: no cross-request authority leakage, no partial release, and every deployed contract is versioned and reproducible.

## Optional anomaly monitor

The earlier trajectory observer remains experimental. It may produce alerts for analysis, but its score must not authorize actions or be treated as a prompt-injection label. Its recent six-case development fixture blocked one benign case and allowed one injection case; this is enough to reject current enforcement use, not to estimate population performance. Keep it in shadow mode unless a separate, fresh evaluation proves value without unacceptable false blocks or overhead.
