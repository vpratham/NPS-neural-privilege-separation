# NPS implementation plan

**Progress:** Stage 1 implemented. Stages 2–5 pending. See `validation/policy_memory_smoke.json` for the local pretrained comparison and `tests/test_policy_memory.py` for structural checks.

## Requirements and evidence

Build deterministic, host-defined permissions in the model execution path. Preserve the user's separate tool-action broker. Do not substitute anomaly or text classification for authority.

Current evidence: `neural_state_firewall/hf_adapter.py:374` serializes system policy and task/context; `:442` owns cached inference; `runtime.py:66` gates response release; `nps_gateway/store.py:149` mediates a narrow source-copy effect. The boundary contract is `neural_state_firewall/NPS_PERMISSION_BOUNDARY.md`. This plan implements its structural prerequisite and adds an explicit causal-attention baseline so unchanged policy memory cannot be mistaken for a new defense.

## Stages and acceptance criteria

1. **Protected policy-memory foundation (implement now).** Encode the trusted system prefix alone; isolate its per-layer K/V tensors from the writable request cache. Read a fresh combined attention view; reject overwrite positions and unsupported cache mutations. Reuse the current adapter and profile identity checks. Tests must show real tiny-Qwen greedy/logit equivalence to ordinary causal decoding, ordinary causal-prefix invariance, independent working memory across requests, policy-change sensitivity, and fail-closed corruption handling. A local pretrained smoke comparison must record what actually ran. This stage changes memory ownership, not permission semantics.
2. **Deterministic read permissions inside attention (next).** Define host-assigned policy/data/output compartments and the complete allowed information-flow graph. Implement denied read edges at every decoder layer and cached step, including indirect paths through other tokens. Start with two permissions: permitted evidence and denied evidence. Define observable metadata (including length/positions) explicitly. Reject any design that only masks the final layer. Acceptance: changing denied values at fixed public metadata leaves released logits/tokens unchanged; permitted facts can still affect answers; input role spoofing cannot change host-assigned permissions. A model cannot derive its own permissions from text. The policy-memory module is a prerequisite, not evidence that these read permissions exist.
3. **Controlled behavioral evaluation.** Freeze model, serializer, graph and decoding. Compare ordinary model, isolated-memory-only, and permission-enforced paths on paired policy swaps, context swaps, benign tasks and adversarial documents. Report task utility separately from the structural noninterference property. First use seen development fixtures; author and quarantine final evaluation only after design freeze. An injection in permitted evidence can still influence answers; report it as a residual limitation.
4. **Runtime integration and portability.** Carry verified compartment labels through request ingestion, caching, cancellation and response release. Keep tool-effect grants only at the existing action boundary. Test cross-request/tenant reuse, errors and concurrency before adding a second local model adapter. Closed APIs without attention/cache control are unsupported for the internal boundary.
5. **Reproducible release decision.** Publish the exact enforceable property and trusted computing base, measured utility/latency, failed cases and regression suite. Expand scope only after a new permission mechanism has a testable invariant. No claim of arbitrary natural-language instruction immunity follows from any of the above.

## Risks and mitigations

- Ordinary causal decoders already keep earlier prefix representations independent of future tokens (Transformers 4.57 caching documentation). Include this control and expect memory-only output equivalence; do not train new weights simply to recreate it.
- Mutable PyTorch tensors are not an OS isolation boundary. Clone ownership, avoid exposing protected aliases, check digests before release, and trust host/runtime code explicitly.
- Attention restrictions can destroy usefulness or leak through indirect token paths. Specify the full graph and check all layers/cache steps before behavioral evaluation.
- Cache APIs vary by version. Support installed Transformers 4.57.6, full-attention Qwen2, batch-one greedy float32 only initially; fail explicitly on unsupported layouts.
- Reusing a previous monitor profile under new inference mechanics is invalid. Bind a distinct decoder identity and require fresh capture/profile for the optional anomaly monitor.

## Verification and stop condition for this work slice

Run focused structural tests, existing firewall regressions, a cached local Qwen2.5-0.5B comparison, CLI help and diff checks. Complete stage 1 and record stages 2–5 as pending. No GPU sweep, dataset import, model training, or production deployment is needed for stage 1.

Primary reference: https://huggingface.co/docs/transformers/v4.57.1/cache_explanation (causal prefix behavior and cache API); implementation source inspected in the pinned local environment.

## Stage 1 completion evidence

- Full firewall suite: 53 tests passed, including nine new policy-memory checks.
- Pinned local Qwen2.5-0.5B comparison: all three seen development requests produced identical token sequences and completed at EOS. Feature differences stayed within the declared tolerance.
- The ordinary causal-prefix control also passed. The memory-only change is not a measured security improvement.
- CLI comparison command, source hashes and sanitized report are committed with the implementation. No model weights or new dataset rows were added.
- Next: Stage 2 information-flow specification and all-layer attention read-permission implementation.
