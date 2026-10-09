# Neural Privilege Boundary (NPB): research and engineering plan

**Purpose:** turn the current capability-boundary prototype into a reproducible, useful security component, while separately testing the research claim that policy state can be kept structurally distinct from untrusted context.

**Scope boundary:** NPB can deterministically restrict data passed to a model, text it may release under a narrow schema, and actions it may propose. Arbitrary natural-language generation is not made injection-proof by prompt formatting or protected policy memory. Any privileged effect must remain behind trusted host authorization.

## Current baseline

The permission-boundary workstream has two different pieces that must not be conflated:

1. The local Transformers path masks denied document tokens at every supported attention layer, seals policy key/value (K/V) state from mutation, and buffers output until generation completes. The current evidence supports a scoped denied-source noninterference claim under documented model/runtime assumptions. It does not show resistance to instructions in readable evidence.
2. `CapabilityBoundary` binds a host-created profile to one instance. It separates readable sources, sources whose exact text may be disclosed, and typed action proposals. It blocks free prose and does not execute actions. This gate is not yet connected to the live Q&A service or a real provider adapter.

The existing BIPIA-derived 175-case sample is **consumed development material**: no semantic attack-success labels were collected, and 42/175 readable-attack cases changed ordinary output strings. That is descriptive only. Do not use those cases as a fresh held-out test or call output change attack success. See the [capability contract](neural_state_firewall/CAPABILITY_BOUNDARY.md), [read-permission report](neural_state_firewall/READ_PERMISSIONS.md), and [paper study protocol](../docs/neural_state_firewall_paper/STUDY_PROTOCOL.md).

## Work plan and gates

### Stage 1 — Freeze the security contract

Write and version the exact NPB policy schema for each model instance:

- Sources the instance may read.
- Sources whose content may be disclosed to the caller.
- Action names and argument schemas it may propose.
- Which trusted broker authorizes each action, including identity, argument values, consent, expiry, replay protection, and audit requirements.
- Behavior on invalid output, unavailable provider, timeout, configuration drift, or incomplete generation: release no text and perform no effect.

Treat profiles as host-owned immutable configuration; never accept grants, identity, source provenance, or action authorization from the model request or retrieved content. Keep free-form answers explicitly outside the deterministic guarantee.

**Gate:** every allowed effect has one named host authorization path; every denied operation fails closed in unit tests. Any action without a concrete broker contract stays disabled.

### Stage 2 — Build the model/provider integration and test harness

Implement one real structured-output provider adapter for `CapabilityBoundary` and route all its output through the gate. Preserve the current document-Q&A endpoint separately until the adapter passes its own tests; do not silently change Q&A into quote-only responses. For each provider call, record model/provider revision, request/config hashes, decoding settings, latency, token use, status/reason, and broker outcome, while excluding raw private content from routine logs.

Use a deterministic fake provider for exhaustive gate tests. Add end-to-end tests proving: denied evidence never reaches the provider; only exact quotes from disclosable sources escape; unlisted action names or argument keys are rejected; an allowed proposal still cannot cause an effect without broker authorization; duplicate, expired, replayed, cross-principal, and malformed proposals cause no effect. Add crash, timeout, concurrency, and configuration-mutation cases.

**Gate:** zero unauthorized disclosures or effects across the complete deterministic regression suite; all tests reproducible offline.

### Stage 3 — Prepare versioned evaluation data

Use three data tracks and keep them separate:

| Track | Data | Use and restriction |
|---|---|---|
| Consumed regression | Existing NPB synthetic fixtures and the recorded BIPIA subset | Catch regressions only. Never describe these as held-out evidence. |
| Public development | BIPIA tasks not used in the existing sample where licensing permits; a small development slice of InjecAgent and AgentDojo | Adapter debugging and metric calibration. Pin commit, files, and hashes; exclude from the final test. BIPIA's WebQA and summarization source data require separate terms review. |
| Locked evaluation | Source-separated, held-out cases from at least two independent benchmarks plus NPB capability-matrix cases | One final run after code, prompts, profiles, and thresholds are frozen. Keep examples and labels private until the run is complete. |

Recommended sources:

- [BIPIA](https://github.com/microsoft/BIPIA): useful for indirect instructions in external content and QA-style outcomes. The repository is archived, and its README flags separate license/terms requirements for WebQA and summarization sources. Our existing sample is already consumed; use it only as a regression set.
- [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent): the project reports 1,054 tool-integrated attack cases across 17 user tools and 62 attacker tools and publishes the repository under MIT. This is the closest fit for measuring whether capability and broker checks prevent unauthorized actions.
- [AgentDojo](https://github.com/ethz-spylab/agentdojo): a dynamic agent environment with user tasks and injection tasks, useful for measuring both task utility and attack impact. Pin its version because the project warns that its API is still evolving.
- NPB capability matrix: generate small, deterministic cases covering every profile permission combination (read/disclose/action), plus benign near-misses and malformed or adversarial proposals. This proves gate mechanics, not real-world attack robustness.

Split by source document, task identity, and attack template—not random prompt rows alone—to reduce leakage. Keep a manifest of source URLs, licenses/terms, repository commits, selected IDs, raw-file hashes, exclusions, transformations, and the date each source became “seen.” No benchmark should be both used to tune the defense and reported as an independent final test.

### Stage 4 — Measure the right outcomes

Report each metric by provider/model, task family, source, attack family, profile, and baseline. Preserve numerator and denominator; do not combine unlike tasks into one score.

**Security outcomes**

- Unauthorized read: whether non-granted content reached the model provider.
- Unauthorized disclosure: whether protected or non-disclosable data reached the caller.
- Forbidden action proposal: whether the model emitted an action outside its profile or with extra argument keys.
- Unauthorized effect (primary agent-security outcome): whether the broker/environment recorded any effect not covered by the host grant.
- Attack success: whether the attack's specified harmful goal occurred, based on environment state or a validated semantic label—not keyword detection or mere output change.
- Cross-instance isolation: attempts by one profile to use another profile's evidence, disclosure rights, or action grants.

**Utility and operations**

- Clean task success and evidence-supported answer quality; valid quote release rate; legitimate action completion; refusal and over-block rates.
- Difference from an unprotected but otherwise identical baseline and from host-filtering/broker-only controls.
- Provider and gate latency separately: median and p95; timeout/incomplete/error rates; peak host/GPU memory; tokens and API cost; recovery and audit behavior.
- Model/provider generality: same tests on at least two local model families and one API provider if access, terms, and budget allow. Pin exact revisions and settings; a passing profile test on one provider is not a universal model guarantee.

Use official benchmark scoring where it matches the attack goal, but retain the raw label and scorer version. For ambiguous semantic cases, use one blinded primary annotator and review disagreements plus a random audit sample; report the adjudication rule and agreement. Do not require two people to label every case. The broker's actual state change is the preferred label for action effects.

### Stage 5 — Statistical analysis and preregistration

Before the locked run, state the primary endpoint, hypotheses, exclusions, sample-size target, and analysis code. A suitable primary endpoint is unauthorized-effect rate on held-out attack opportunities, paired against a defined baseline. Also predefine a clean-task utility non-inferiority margin with the intended application owner; do not choose it after seeing results.

- Report rates with exact binomial confidence intervals (and raw counts). If zero failures occur in `n` independent trials, the one-sided 95% upper bound is `1 - 0.05^(1/n)` (approximately `3/n`); zero observed failures is not proof of zero risk. About 300 independent opportunities are needed to put that bound near 1%. Source/task clustering reduces effective sample size, so plan additional cases and use cluster-aware intervals.
- Compare paired binary outcomes with exact McNemar tests; use paired/cluster bootstrap confidence intervals for risk differences and utility differences, resampling by source or task group rather than individual prompt when cases share a source.
- Calculate power/sample size from pilot baseline rates and a preselected practically meaningful effect. Apply Holm correction when making multiple confirmatory comparisons. Keep exploratory analyses labeled exploratory.
- For the capability gate's deterministic invariants, statistical significance is unnecessary: enumerate finite small profile/proposal combinations and use property-based fuzzing for larger structured inputs. Statistical evaluations address provider behavior and benchmark attack outcomes.

**Gate:** evaluation code reproduces from the pinned manifest; all outcomes have a disclosed denominator and confidence interval; no claim relies on a post-hoc threshold or a model judge alone.

### Stage 6 — State and verify the mathematical claims

Keep three properties separate:

1. **Capability enforcement:** for host profile `C`, the provider receives only evidence in `read(C)`; released quotes must be exact substrings of `disclose(C)`; action proposals must have a name and argument-key set allowed by `C`; and the gate itself causes no side effect. Prove these directly from the validator/control flow and test them exhaustively over small profiles. A separate broker proof/test must show every effect is bound to a valid, unexpired, one-use host grant.
2. **Denied-source noninterference:** for fixed model weights, policy, public task/evidence, host grants, tokenization, and runtime, replacing denied content `D` with `D'` leaves every public/output logit unchanged at every generation step, and therefore leaves greedy tokens unchanged. Formalize the causal-layer invariant and induction over layers and decode steps. State the supported model/backend assumptions and excluded channels (timing, resource exhaustion, memory access, compromised runtime). Existing tests are finite implementation evidence, not a proof across all models or kernels.
3. **Policy-state integrity vs. policy obedience:** the current sealed policy K/V cache is intended to remain byte-identical while the working cache grows. This is testable as an invariant and can be proved for an abstract append-only cache transition system. It does **not** prove generated outputs obey the system prompt: the model can attend to both policy and readable context and still follow an injected instruction. A simple counterexample model that ignores policy demonstrates why cache separation alone cannot imply semantic compliance.

The target claim “system prompt cannot be diluted” must therefore be split into a structural claim (policy memory cannot be modified or overwritten by context) and a behavioral claim (the model always follows policy). The first is a tractable proof target under explicit implementation assumptions; the second requires a mechanism that restricts outputs/effects and a separately stated threat model. Do not claim the latter from unchanged K/V hashes.

**Formal methods sequence:** first write definitions, assumptions, and proof obligations in the paper appendix; then model the append-only policy/working-cache state machine and the capability authorization state machine in a small model checker; finally prove the mask/noninterference lemma at the mathematical abstraction and connect tested implementation behavior to that model. Mechanized theorem proving is a later milestone, not a substitute for the initial operational definitions.

### Stage 7 — Production qualification or explicit no-go

Require: pinned provider/model and runtime; authenticated identity-to-profile mapping; profile/config integrity; broker mediation for every privileged effect; security and clean-task gates passed on locked data; documented API limits, p95 latency, resource ceilings and recovery; audit privacy review; data/model license review; and a versioned release artifact with reproducible tests.

**No-go conditions:** any unauthorized effect or disclosure; any caller-controlled permission change; inability to reproduce the locked run; unacceptable clean-task utility/over-blocking; unsupported model/runtime behavior; missing broker validation; or a security claim that exceeds the measured/proved scope. A no-go result should identify which capability remains unavailable rather than relaxing the gate after the fact.

## Development environments and compute

Keep two isolated environments so model-runtime dependencies cannot destabilize the standard-library capability gate:

1. **Gate/CI environment:** Python 3.11+, no model dependencies; run capability tests, parser fuzz/property tests, broker contract tests, lint/type checks, and deterministic demo on every change.
2. **Neural evaluation environment:** Linux x86_64, Python 3.11, Torch 2.6.0 and Transformers 4.57.6 for the currently supported adapter. Pin CUDA, driver, tokenizer/model commits, and container/environment lock. The active shell has previously shown a different Torch/Transformers combination; do not treat it as the reproducibility environment.

The Mac M2 8-GiB CPU run is a useful small-model correctness baseline, but the recorded BIPIA-derived run took about 99 minutes of summed generation time across 1,750 attempts. Do not use it for a broad multi-model sweep. For the next substantial open-model evaluation, provision one fixed Linux VM with at least 16 vCPUs, 64 GiB system RAM, 200 GiB local SSD, and a 24-GiB GPU such as an L4; run models sequentially. NVIDIA lists 24 GB for L4; Google Cloud maps its G2 family to L4 GPUs. Use a 40–80-GiB GPU if the selected model/context exceeds the 24-GiB budget; A100 40/80-GiB configurations are documented by NVIDIA/Google Cloud. Verify actual memory at startup and save it in each report.

Colab can be used for development batches and notebook-based inspection, but not as the sole final-evaluation host: hardware types and usage limits vary, runtimes can terminate, and availability is not guaranteed. Checkpoint each case and copy artifacts to versioned storage. For final runs, use a persistent fixed VM or another reserved GPU environment; Colab’s own FAQ recommends dedicated capacity when specific hardware or guaranteed resources are needed.

No cloud account, GPU quota, or spending budget is assumed by this plan. Do not launch paid compute until the user selects a budget; the CPU capability suite and proof-spec work can proceed meanwhile.

## Immediate next actions

1. Freeze the profile/action schema and select the first broker-backed action; default all action capabilities to empty until the broker contract is tested.
2. Implement one real provider adapter that emits the gate’s strict schema, then add end-to-end no-effect-on-denial tests.
3. Build a versioned dataset manifest and evaluator for InjecAgent; add AgentDojo after the first end-to-end action path is stable. Keep the existing BIPIA sample regression-only.
4. Reserve a fixed 24-GiB Linux GPU VM for the model evaluation matrix; use Colab only for short development runs if convenient.
5. Write the preregistered analysis and the precise cache-integrity/noninterference definitions before running held-out cases.

## External source references

- [Microsoft BIPIA repository and data-use notes](https://github.com/microsoft/BIPIA) (archived; WebQA/Summarization require separate source-data terms review).
- [InjecAgent repository](https://github.com/uiuc-kang-lab/InjecAgent) (benchmark size, tools, license and evaluation instructions).
- [AgentDojo repository](https://github.com/ethz-spylab/agentdojo) (dynamic benchmark, license and version warning).
- [Google Colab resource FAQ](https://research.google.com/colaboratory/faq.html) (resource availability and runtime limits vary).
- [NVIDIA L4 specifications](https://www.nvidia.com/en-us/data-center/l4/) and [Google Cloud GPU families](https://cloud.google.com/compute/docs/gpus/about-gpus) (24-GB L4 and supported VM families).
- [NVIDIA A100 specifications](https://www.nvidia.com/en-us/data-center/a100/) (40/80-GB configurations).
