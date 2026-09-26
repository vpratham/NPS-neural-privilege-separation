# Neural Firewall experiments

Each NFW experiment should answer one falsifiable question, use a frozen configuration, and write a manifest with model revision, tokenizer/template, activation site, data IDs, split IDs, code revision, seeds, and artifact hashes.

| Series | Purpose | Status |
|---|---|---|
| NFW-00 | Legacy artifact integration and reproducibility | Historical artifacts available |
| NFW-01 | First runtime firewall prototype | Historical; known limitations documented in the audit |
| [NFW-02](NFW-02_foundation_firewall/) | Target-model monitor-and-block baseline | Current foundational experiment |
| [NFW-03](NFW-03_behavioral_proxy/) | Historical-response proxy evaluation, baselines and buffered-gate laboratory | Standalone Colab notebook; full GPU integration run pending |
| [NFW-04](NFW-04_veto_capability_firewall/) | Veto-gated neural attestation with external capability broker | Colab-first PoC implemented; local security tests pass |
| [NFW-05](NFW-05_adversarial_capability_firewall/) | Reproducible adversarial capability-boundary evaluation | Colab-first notebook implemented; local tests pass |
| [NFW-06](NFW-06_adaptive_sandbox_evaluation/) | Adaptive red-team attacks and sandboxed side-effect evaluation | Colab-first notebook implemented; local tests pass |
| [NFW-07](NFW-07_integrated_boundary_evaluation/) | Integrated broker effects, utility, adaptive episodes, and telemetry | Colab-first notebook implemented; local tests pass |
| [NFW-08](NFW-08_live_model_broker_evaluation/) | Pinned live-model proposals, paired broker-policy effects, and benign utility | Colab GPU notebook implemented; local CPU contract tests pass |
| [NFW-09](NFW-09_tool_result_injection/) | Actual tool-result injection, paired model utility/proposals, and capability-broker effects | Completed Colab run archived; post-hoc forensic audit identifies native tool-format confounding |
| [NFW-10](NFW-10_protocol_corrected_injection/) | Protocol-corrected native tool-call injection study with development gate and frozen held-out matrix | Colab GPU notebook implemented; local CPU contract tests pass |
| [NFW-11](NFW-11_authorization_provenance/) | Replays NFW-010 proposals to compare scope-only, trusted-source-bound, and expected-answer-oracle capabilities | CPU-only replay complete; reference broker module and contract tests |

Do not use the final evaluation partition to choose layer, threshold, persistence, intervention strength, or judge. Every expected example needs one terminal status: `ok`, `blocked`, or `error`; errors remain in the report.

NFW-03 now establishes behavioral-proxy evaluation before policy-isolation work. The latter remains a future research milestone; the new notebook does not claim it has been achieved.

NFW-04 is the recommended product direction: model-generated requests are untrusted, and the external broker—not an activation score—enforces authority.

NFW-11 is a post-hoc, deterministic replay of NFW-010 outputs. Its source-bound arm derives the permitted write value from a host-verified source receipt without supplying the expected-answer label to the issuer. The experiment remains limited to exact-copy synthetic tasks and does not validate production trust roots or arbitrary natural-language authorization.
