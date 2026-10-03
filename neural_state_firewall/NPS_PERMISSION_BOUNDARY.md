# Neural privilege separation: permission-boundary design

**Status:** protected policy KV memory and all-layer document read permissions implemented, with local API/CLI and development evaluation. General instruction-authority enforcement for readable evidence remains unestablished. See [the running guide and exact contract](READ_PERMISSIONS.md) and [the implementation plan](NPS_IMPLEMENTATION_PLAN.md).

## Goal

Keep trusted policy and permission state outside the writable token stream while allowing the model to use lower-trust text as evidence. A prompt label such as “untrusted data” is not an enforcement boundary. A trajectory alarm is not a permission decision.

The implemented read guarantee is deliberately narrow: **document text cannot change the host-issued read set, and denied document values have no attention path into the public response under the declared model/runtime assumptions.** A separate action mediator remains responsible for privileged tool effects. Whether generated natural-language answers obey semantic instructions remains an empirical model behavior.

## Trust and authority

The October pilot workload is document-grounded question answering for one host-configured permission realm. The authenticated token identifies that realm; it confers no ability to alter grants, source IDs, policy, model configuration or response budget. A model answer is untrusted text. Authentication and a process watchdog are implemented in `pilot.py`; the deployment instructions and limits are in `PILOT.md`.

Protected assets are denied document values and the integrity of host-issued grants/policy. Free-form answer correctness and resistance to instructions in readable documents are measured separately and remain unvalidated. A timeout, worker crash or integrity failure releases no response. The broker is the sole authorized tool-effect path; the response server exposes no tool endpoint. Host OS compromise, document-file access outside this service and downstream applications that execute prose are outside this contract.

| Component | Trust | May provide | May change permissions? |
|---|---|---|---|
| Host policy/configuration | Trusted | Policy and allowed capabilities | Yes, before a request starts |
| Authenticated user task | Lower than host policy | Requested task | No |
| Retrieved documents/tool results | Untrusted data | Evidence for the task | No |
| Model output | Untrusted proposal | Answer or proposed action | No |
| Host action mediator | Trusted enforcement point | Execute an authorized effect | No; it checks host-issued grants |
| Neural trajectory monitor | Advisory sensor | Deviation telemetry / veto signal | No |

User-task authority is application-specific. Do not silently equate “user supplied” with “trusted to override the application policy.”

## Architecture contract

Represent the trusted policy as a separate, immutable request state `q`, and the model's writable working state as `x_t`:

```text
q = EncodeTrustedPolicy(host_policy, host_permissions)
x_0 = EncodeRequest(user_task, untrusted_context)
x_(t+1) = DecoderStep(x_t, read_only(q))
q_(t+1) = q_t
```

The implementation must not serialize `q` and untrusted text into one writable memory and then call that separation. The policy state must be initialized only from host inputs, stored separately from the autoregressive working/KV state, and exposed to decoder steps through read-only access. No decoder transition may write, replace, or extend `q`. Any monitor/controller state is separate and cannot grant permissions.

The model may propose a typed action. It receives no grant token or permission-minting API. The host mediator checks the proposal against the host-issued capability and exact resource/content constraints immediately before the effect. No other component executes the action.

This architecture prevents a lower-trust token from mutating the protected policy memory by construction. It does **not** prove that the model follows the policy when producing ordinary text, nor does it prevent semantic influence from untrusted text on `x_t`. Do not claim a general prompt-injection solution from read-only policy state alone.

## Required invariants

1. **Policy-state immutability:** for a fixed trusted policy/configuration, the canonical policy-state bytes and digest are identical before and after every decode step, regardless of request/context content.
2. **Permission non-escalation:** the model cannot create, widen, or delegate host capabilities. A missing, malformed, expired, mismatched, or replayed grant authorizes no effect.
3. **Effect mediation:** each privileged side effect is checked against the exact host grant at the point of effect; a monitor alarm may veto but cannot authorize.
4. **Fail-closed boundary errors:** state-integrity, parser, mediator, and runtime failures produce no privileged effect. Do not infer answer correctness from a blocked action.
5. **Behavioral policy compliance is measured separately:** task utility, policy violation, data disclosure, and unauthorized action outcomes remain separate labels.

The threat model trusts the host process, policy source, model-loading/runtime code, and mediator. It does not cover compromised host memory, malicious model weights, physical side channels, or arbitrary downstream use of displayed prose.

## Prototype scope

The first increment uses the existing Qwen weights to prefill the trusted system prefix alone, then seals its per-layer K/V tensors separately from the writable request cache. `PolicyMemoryAdapter` in `policy_memory.py` uses those tensors at every attention step through newly allocated attention views. It rejects attempted cache overwrites, checks policy digests before releasing each generation step, and uses a distinct decoder identity. It supports batch-one full-attention Qwen2 with Transformers 4.57.6; it does not add a learned encoder, change attention permissions, or train weights. The existing anomaly monitor remains a separate optional control; the equivalence check runs without it.

**Causal baseline correction:** ordinary causal attention already prevents later tokens from changing earlier prefix representations ([Transformers documentation](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation)). The first increment adds explicit memory ownership and integrity checks, not a new semantic defense. The tests include ordinary-prefix invariance and ordinary-versus-isolated decoder equivalence to prevent a false security claim. A constant or unread policy state would also be immutable; immutability alone is not sufficient evidence of policy enforcement.

The read-permission increment is implemented in `read_permissions.py`: host-assigned evidence visibility, fixed-capacity denied slots, logical public positions, reserved-token-safe data encoding, and verified masks at every layer/cached step. Tests cover Qwen2 and Llama under eager and SDPA attention. This is a specific information-flow guarantee; injections in permitted evidence may still influence natural-language answers. Tool effects remain on the separate mediator path.

The first checks are structural, not efficacy benchmarks:

- inspect the computation graph and state ownership to establish that only trusted initialization writes `q`;
- hash `q` before and after every step under clean, conflicting, and role-spoofed contexts;
- verify policy swaps change `q` while holding request/context constant;
- verify lower-trust context swaps leave `q` and the host capability set unchanged;
- verify allowed actions pass and altered scope/content, forged grants, and replayed grants cause zero effects;
- verify model/API failure and monitor failure do not create an effect.

These checks demonstrate the implementation invariant, not that responses resist attacks. Behavioral evaluation comes after the architecture and supported guarantee are explicit.

## Existing code and gap

The default `neural_state_firewall/hf_adapter.py` sends policy as a system message and task/context in one user message; its own docstring states the context label is not a security boundary. The opt-in `policy_memory.py` path verifies that the system-only encoding is an exact token prefix before storing it separately; it rejects incompatible tokenizers. `neural_state_firewall/runtime.py` buffers and conditionally releases text based on an anomaly monitor; it does not authorize actions. `nps_gateway/` already has a separate host-owned, source-bound action mediator, but it is not an internal policy-state architecture and remains on the tool/action path only.

This design follows the long-term Neural Privilege Separation objective in `docs/NPS_Charter.md` and the protected policy-state formulation in `docs/NPS_IMPLEMENTATION_AUDIT.md` (Milestone D). The audit's runtime and evaluation milestones remain prerequisites for making behavioral security claims.
