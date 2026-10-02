# NPS implementation plan

**Progress:** Stages 1–2 implemented. Stage 3 has a reproducible seven-case development comparison; held-out/adaptive evaluation remains pending. Stage 4 has a working local request/response API, failure handling and two-family structural tests; authenticated production deployment and pretrained Llama validation remain pending. Stage 5 has a documented boundary and reproducible development evidence, with production eligibility explicitly false. Start with `READ_PERMISSIONS.md`.

## Requirements and evidence

Build deterministic, host-defined permissions in the model execution path. Preserve the user's separate tool-action broker. Do not substitute anomaly or text classification for authority.

Current implementation: `read_permissions.py` assigns host source permissions and enforces every-layer/cached attention masks; `policy_memory.py` seals the trusted prefix; `runtime.py` buffers response release; `permission_evaluation.py` compares ordinary, isolated-memory-only and read-enforced decoding. The separate `nps_gateway` retains tool-effect mediation. The boundary contract is `NPS_PERMISSION_BOUNDARY.md`, with the exact implemented graph and limits in `READ_PERMISSIONS.md`.

## Stages and acceptance criteria

1. **Protected policy-memory foundation (implemented).** Encode the trusted system prefix alone; isolate its per-layer K/V tensors from the writable request cache. Read a fresh combined attention view; reject overwrite positions and unsupported cache mutations. Reuse the current adapter and profile identity checks. Tests must show real tiny-Qwen greedy/logit equivalence to ordinary causal decoding, ordinary causal-prefix invariance, independent working memory across requests, policy-change sensitivity, and fail-closed corruption handling. A local pretrained smoke comparison must record what actually ran. This stage changes memory ownership, not permission semantics.
2. **Deterministic read permissions inside attention (implemented).** Define host-assigned policy/data/output compartments and the complete allowed information-flow graph. Implement denied read edges at every decoder layer and cached step, including indirect paths through other tokens. Start with two permissions: permitted evidence and denied evidence. Define observable metadata (including length/positions) explicitly. Reject any design that only masks the final layer. Acceptance: changing denied values at fixed public metadata leaves released logits/tokens unchanged; permitted facts can still affect answers; input role spoofing cannot change host-assigned permissions. A model cannot derive its own permissions from text. The policy-memory module is a prerequisite, not evidence that these read permissions exist.
3. **Controlled behavioral evaluation.** Freeze model, serializer, graph and decoding. Compare ordinary model, isolated-memory-only, and permission-enforced paths on paired policy swaps, context swaps, benign tasks and adversarial documents. Report task utility separately from the structural noninterference property. First use seen development fixtures; author and quarantine final evaluation only after design freeze. An injection in permitted evidence can still influence answers; report it as a residual limitation.
4. **Runtime integration and portability.** Carry verified compartment labels through request ingestion, caching, cancellation and response release. Keep tool-effect grants only at the existing action boundary. Test cross-request/tenant reuse, errors and concurrency before adding a second local model adapter. Closed APIs without attention/cache control are unsupported for the internal boundary.
5. **Reproducible release decision.** Publish the exact enforceable property and trusted computing base, measured utility/latency, failed cases and regression suite. Expand scope only after a new permission mechanism has a testable invariant. No claim of arbitrary natural-language instruction immunity follows from any of the above.

## Risks and mitigations

- Ordinary causal decoders already keep earlier prefix representations independent of future tokens (Transformers 4.57 caching documentation). Include this control and expect memory-only output equivalence; do not train new weights simply to recreate it.
- Mutable PyTorch tensors are not an OS isolation boundary. Clone ownership, avoid exposing protected aliases, check digests before release, and trust host/runtime code explicitly.
- Attention restrictions can destroy usefulness or leak through indirect token paths. Specify the full graph and check all layers/cache steps before behavioral evaluation.
- Cache APIs vary by version. Support pinned Transformers 4.57.6, full-attention Qwen2 and Llama with ordinary RoPE, eager/SDPA, batch-one greedy float32; fail explicitly on unsupported layouts. Pretrained evidence currently covers Qwen2.5-0.5B only.
- Reusing a previous monitor profile under new inference mechanics is invalid. Bind a distinct decoder identity and require fresh capture/profile for the optional anomaly monitor.

## Verification and stop condition for this work slice

Run structural and failure-path regressions, the pinned Qwen2.5-0.5B three-arm development comparison, and live local HTTP checks. Record actual results and reproduction instructions. This implementation slice stops at a verified local permission boundary; held-out/adaptive security claims, multi-tenant deployment and broad semantic instruction immunity remain outside its acceptance criteria.

Primary reference: https://huggingface.co/docs/transformers/v4.57.1/cache_explanation (causal prefix behavior and cache API); implementation source inspected in the pinned local environment.

## Stage 1 completion evidence

- Full firewall suite: 53 tests passed, including nine new policy-memory checks.
- Pinned local Qwen2.5-0.5B comparison: all three seen development requests produced identical token sequences and completed at EOS. Feature differences stayed within the declared tolerance.
- The ordinary causal-prefix control also passed. The memory-only change is not a measured security improvement.
- CLI comparison command, source hashes and sanitized report are committed with the implementation. No model weights or new dataset rows were added.
- Stage 2 is now implemented; see the completion evidence below.


## Read-permission implementation evidence

- The graph denies every denied-to-public attention edge in every decoder layer and cached step. Denied queries read only themselves; public position IDs ignore denied slots. Fixed 256-token denied-source capacity prevents content-length-dependent shape changes within the allowed capacity.
- Structural tests cover real tiny Qwen2 and Llama under eager and SDPA, denied-content/length swaps, readable-source and host-grant sensitivity, missing masks, role-token encoding, request reuse, concurrency, cancellation and failure cleanup.
- `validation/read_permission_development.json` records the seven seen pretrained cases, three controls, per-case source/grant hashes and counterfactual logit checks. All three arms answered all seven exact-answer tasks; no baseline attack succeeded. This cannot support an attack-success reduction claim.
- `validation/read_permission_http_smoke.json` records actual pretrained HTTP responses: permitted answer `Friday`, unavailable private answer `UNKNOWN`, and HTTP 400 for client-supplied grants/context. These outcomes are smoke evidence, not a guarantee that all private-answer requests return UNKNOWN.
- CLI commands `run-permissions`, `serve-permissions` and `evaluate-permissions` require host-loaded sources/grants and no anomaly profile. The endpoint does not run tools.
- Evaluator timings exclude adapter construction, including repeated weight hashing. They are diagnostics, not a performance benchmark; optimize startup only when profiling justifies it.

## Next acceptance gates

1. Freeze the model, serializer, permissions and scoring protocol; collect source-separated representative held-out tasks and benign utility tasks. Exclude all seen fixtures and derivatives.
2. Run a defense-aware adaptive attacker with a recorded budget; separately score denied-data disclosure, readable-evidence instruction takeover, utility, incomplete generations and errors. Use the disclosed single-review protocol if a second reviewer is unavailable.
3. Measure generation and end-to-end latency, including model/configuration initialization, under realistic load. Test pretrained second-family models before claiming pretrained portability.
4. Bind principals, retrieval provenance and grants in the intended application; add process-level deadlines/isolation and deployment authentication before exposing the service beyond loopback.
5. Make the release decision from those results. Production and publication robustness claims remain unsupported until these gates pass. No classifier threshold or successful unit suite can substitute for them.
