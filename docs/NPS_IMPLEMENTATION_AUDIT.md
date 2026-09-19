# Neural Privilege Separation: implementation and research audit

Audit date: 2026-09-19.

**Assessment.** NPS has a useful activation-monitoring prototype and a substantial exploratory experiment history. The checked-in evidence does not yet demonstrate a robust neural firewall, a causally isolated policy representation, or neural privilege separation. The highest-value next step is to consolidate and validate one implementation, generate correctly labeled target-model outputs, and evaluate a narrowly defined security claim. More direction searches or larger notebooks will not resolve the present measurement problems.

This audit reviewed the charter and mathematical framework, both implementation branches, experiment notebook source, archived reports and metric tables, and the latest NFW-001 Parquet outputs. It ran the existing synthetic smoke test and targeted CPU reproductions of package defects. It did not rerun the pretrained Qwen experiments, download model weights, or independently judge generated responses. Historical numbers below are saved results, not newly replicated model benchmarks. No runtime implementation was changed by this audit.

**1. What the project is trying to establish**

The charter proposes separating task computation from the internal state that governs authorized behavior. Its intended progression is:

```text
Decode a candidate representation
    -> establish a selective causal effect on behavior
    -> restrict unauthorized changes to policy state
    -> maintain enforcement across computation
    -> withstand an explicit adaptive attacker while retaining utility
```

That is a meaningful research program. However, the repository currently contains several different experimental targets under the word “policy”:

| Target | Suitable ground truth | What success establishes |
|---|---|---|
| Harmful request detection | Request intent under a specified policy | The monitor recognizes a request category |
| Refusal prediction | What the target model actually does | The monitor predicts a response style/decision |
| Harmful output prediction | Independently judged target-model continuations | The monitor predicts a behavioral violation |
| Instruction authority | Trusted policy, request, source roles, authorized outcome | The system distinguishes authorized instructions from untrusted data |
| Privilege separation | An enforced information-flow boundary and behavioral evaluation | A stated restriction on attacker influence actually holds |

These labels are not interchangeable. A model can recognize harmful intent and safely answer with a refusal; a harmless request can trigger an excessive refusal; an injection can request an innocuous but unauthorized action. Most current probes investigate the first row. The original NPS objective is closer to the last two rows.

The v2 README already recognizes these distinctions. The problem is carrying them consistently into dataset schemas, runtime APIs, experiment selection, and public claims.

**2. What the saved evidence supports**

| Evidence | Recorded result | Defensible interpretation |
|---|---|---|
| [Early experiment log](../EXPERIMENT_LOG.md) | Exp002 near-perfect AUC, with zero shared topics across labels; category-held-out Exp003 best AUC 0.561 | Topic confounding was identified; broad policy generalization remained unresolved |
| [Exp011 archive](../results/nps_exp011_policy_subspace_results-3.zip) | Near-perfect probes on 400 prompts; PCA/subspace comparisons | Geometry of the sampled labeled prompts; not identification of a privileged policy variable |
| [Exp012 archive](../results/NPS_Experiment_012_Results-2.zip) | Offline subspace removal changes probe performance | A feature ablation study; no generation-time security claim. Some strongly below-chance refit results need further diagnosis |
| [Exp013 archive](../results/NPS_Experiment_013_Results.zip) | Policy-memory classifier reaches 100% validation accuracy on a small templated dataset | A trainable readout of frozen features; the report explicitly says no generated output was modified |
| [Exp014 archive](../results/exp014_outputs_f.zip) | Baseline refusal 0.554, peak reinforcement 0.562 | Weak aggregate refusal change; regex evaluation and an approximate inverse of the encoder prevent a strong selective-control claim |
| [Exp017 summary](../results/exp017/exp017_summary.json) | Clean recall 48/64 = 75%; adversarial recall 44/100 = 44%; 0 false positives among 71 clean negatives | A useful prompt detector with a substantial adversarial transfer gap |
| [Local XSTest evaluation](../neural_firewall/phase1_real/xstest/heldout_evaluation.json) | TP=46, FN=14, TN=75, FP=0 | Another small prompt-detection result; this is a different run/split from the preceding row |
| [Exp019 archive](../experiments/exp019/exp019_run_20260803_110726.zip) | XSTest-only probe recall: SorryBench 19.3%, StrongREJECT 50.5%; improves after adding training benchmarks | Dataset coverage matters. Most non-XSTest evaluation sets are positive-only, so their precision/F1 cannot establish specificity |
| [Exp021B pilot](../neural_firewall/phase1_real/xstest/exp021B_head_causal_4bit/sweep_summary.json) | 80 training examples, 8 evaluation examples, 128 head/strength combinations | Pilot screening, as the artifact itself says; four unsafe examples make a 0.25 refusal gain a one-example change |
| [Earlier NFW-001 archive](../neuralFirewallV2/experiments/NFW-01_firewall_v0/v1to3/succesful_run/outputs_outputs.zip) | Unsafe interruption 18%, jailbreak interruption 24%, benign false block 7.94%; measured overhead 8.61% | Runtime interruption works, but reported interruption is not independently verified harm prevention |
| [Latest NFW-001 report](../neuralFirewallV2/results/NFW-01_firewall_v0/drive-download-20260915T104150Z-1-001/FORENSIC_REPORT.md) | Prompt harmful detection 25%; jailbreak-positive detection 5.57%; OOD detection 10.93%, OOD AUROC 0.5687 | Frozen legacy probes have weak performance at this selected operating point on the new distribution |

The latest report's external behavioral-label recall of 8.65% and FPR of 1.91% are **not Qwen-output safety metrics**. The report correctly discloses that its labels describe responses from other models. Its corrections to activation indexing, calibration/test separation, and metric names are worth retaining.

Zero observed false positives in 75 benign examples does not establish a 2% population FPR. Even assuming independent representative examples, the exact one-sided 95% upper bound is approximately 3.92%. Grouped/template dependence can make effective evidence weaker still.

**3. Immediate blockers in evaluation**

**P0: behavioral labels refer to the wrong generated object.** V5 generates Qwen continuations but calibrates and evaluates their activation trajectories using Necent `response_harmful` labels attached to other models' responses. The provenance check records a warning and continues. Recalibration on those labels cannot repair the mismatch.

Replace this with a response table keyed by `(example_id, model_revision, defense_id, decoding_config, seed)`. Generate baseline and defended responses under the same serialization and comparable decoding settings. Independently label each actual response for harmful assistance and task success. Keep original dataset labels as separate metadata. Use a frozen evaluator and a blinded, stratified human audit of disagreements and apparent defense successes. Refusal phrases are useful diagnostics, not the primary safety evaluator.

**P0: the frozen evaluation is incomplete and the missingness is class-dependent.** Direct comparison of `behavioral_labels.parquet` against `frozen_behavioral_eval_behavioral_test.parquet` gives:

```text
Expected: 1,000 rows = 215 external positives + 785 negatives
Evaluated: 970 rows = 185 external positives + 785 negatives
Missing: 30 rows, all external positives
Missing source: llama3-jailbreaks
Missing attack_technique: circuit_breaker
```

The firewall crosscheck also has 970 rows. The available release does not establish why those rows disappeared. Do not infer intentional exclusion or treat missing cases as safe. V5's `summarize_trajectory` returns `None` for missing/empty trajectories, and subsequent evaluation loops skip missing summaries. Require exactly one terminal status per expected example, assert complete joins, rerun failures, and report unresolved failures and bounds separately. Source-specific missingness must be visible in the main result table.

**P0: layer selection consumes the nominal test set.** [nps_36layer_probe_sweep.py](../neural_firewall/nps_36layer_probe_sweep.py) evaluates `T, test_y`, computes `selected_by_rule` using those results, then exports the earliest qualifying layer. That set is a development set once used for selection. Use train/development/calibration/final-test partitions or nested grouped validation. Selection of layer, rank, controller strength, persistence, and aggregation must not use final-test outcomes.

The checked-in Exp011 `compute_subspace_for_layer` also fits PCA on the pooled full activation dataset. Those components are acceptable for descriptive full-dataset geometry, but cannot then be reused as training-only preprocessing in a purportedly untouched Exp012 evaluation. Refit the subspace and all centering/scaling on training rows. PCA of pooled activations measures total variance, which can include topic and formatting; its explained-variance dimension is not by itself the dimension of policy. Subtracting a common benign centroid before ordinary centered PCA does not isolate a policy factor.

**P1: label/source confounding persists beyond the early topic audit.** Exp018/019 retain both classes in XSTest but load several other benchmarks as harmful-only. Holding out XSTest leaves a single-class training pool, which the saved LOBO results explicitly skip. Permuted-label and random-direction controls are helpful, but do not rule out a benchmark-source classifier when source correlates with the true label. Include safe and unsafe examples from multiple sources and matched domains; split by underlying behavior, template/paraphrase family, and source. Deduplicate across all benchmark imports before splitting, including legacy probe training data.

V5 checks disjoint row indices and held-out attack techniques. Those are useful checks, but distinct row indices do not establish distinct prompt content or distinct base behaviors. The inspected labeled behavioral calibration/test subsets had zero exact prompt-string overlap; that limited check does not establish global semantic deduplication.

**P1: prompt-injection labels and task outcomes are mismatched.** All 1,000 latest `test_injection` examples have `prompt_harmful=0` and lack behavioral labels. Its 1.3% flag rate cannot measure injection defense effectiveness. Evaluate whether untrusted content caused deviation from a legitimate task or policy, including cases whose requested output is harmless in isolation. Preserve system/user/tool/document roles in the data schema.

**P1: refusal, noninterruption, output similarity, and utility have been used as substitutes for each other.** Exp014 uses approximate back-projection through a nonlinear encoder and a refusal heuristic. Exp015 evaluates baseline-text similarity. Earlier NFW-001 explicitly defines capability preservation as `1 - false_block_rate`. None of these independently establishes answer correctness or usefulness. Record refusal, harmful assistance, benign task success, and semantic utility separately. A safe rejection of a discriminatory premise can be useful without matching a refusal regex.

**4. Reproduced implementation defects**

The following CPU checks execute the existing package with a tiny deterministic adapter; they do not require pretrained weights. The raw outcomes are saved in [NPS_AUDIT_CHECKS.json](NPS_AUDIT_CHECKS.json).

| Priority | Location | Observed defect | Required correction |
|---|---|---|---|
| P0 | `neural_firewall/neural_firewall/firewall.py:147` | `_combine_votes` returns a decision, but `score_pooled` discards it and thresholds the vote fraction at 0.5. ANY with 1/4 hits allows; ALL with 2/4 hits blocks | Preserve each policy's boolean decision; define cross-policy aggregation separately from a display score |
| P0 | `firewall.py:137` | Missing layers are silently skipped. Empty and all-NaN inputs both produce `exceeded_threshold=False` | Validate all required layers, shapes, finite scores, and nonempty policies. Surface an explicit error status; enforce fail-closed behavior at the request/output boundary |
| P0 | `firewall.py:248` | `intervene()` installs hooks, executes `pass`, removes hooks, and returns `intervened=True` | Make generation occur inside a context manager, or expose a single `generate()` API that owns the entire monitored execution |
| P0 | `activation_extractor.py:70`, `model_interface.py:220` | Offline extraction reads decoder-layer inputs; streaming returns `hidden_states[i+1]` under key `i` | Use one explicit activation-site specification for extraction, streaming, intervention, and artifacts. Verify it against the real reference model |
| P1 | `activation_extractor.py:49` | `mask.sum()-1` assumes right padding. With mask `[0,1,1]`, it selects the middle token | Locate the final nonmasked position, or enforce and validate a documented padding convention |
| P1 | `firewall.py:143` | A batch-shaped activation is accepted, but only row 0 is scored | Implement per-example decisions or reject batch sizes above one explicitly |
| P1 | `firewall.py:115` | `AVERAGE_PROJECTION` ignores stored thresholds and applies sigmoid to unit-normalized margins | Preserve raw logits and their score transform; calibrate the actual ensemble decision rather than silently changing operating points |

The no-op finding applies to `NeuralFirewall.intervene()/decide()`; the older streaming path does keep hooks installed during forward passes. That streaming path, however, applies intervention across the loop whenever INTERVENE mode is selected, rather than implementing a validated risk-gated controller.

V5 separately fixes the layer-offset problem for layers 19–22 by comparing `hidden_states[L]` with decoder-layer `L-1` output. Do not attribute the old package's offset defect to that corrected path. Port the verified convention into the common adapter. At final layers, account explicitly for any final normalization; do not assume every architecture's returned hidden-state tuple is identical to raw block outputs.

The existing `test_smoke.py` passed. It tests the intervention return flag rather than an actual intervened forward pass, does not exercise ANY/ALL voting truth tables, and its mock generation does not implement a persistent KV cache. Passing it is useful plumbing evidence, but insufficient regression coverage for the deployed paths.

**5. Serialization, artifacts, and release reproducibility**

V5's prompt extraction and generation tokenize raw strings. It does not apply Qwen's chat template in those paths. Its batch prompt extraction truncates at 2,048 tokens, while generation tokenizes the full prompt. The older extraction path uses raw strings with a 512-token limit, while `nps_causal_intervention.py` applies a chat template during generation. Thus some experiments use materially different model inputs despite referring to the same prompt.

Raw-text scoring is valid as an explicitly labeled legacy reproduction. For a deployed chat firewall, define one canonical message serializer and train/calibrate on the actual deployment serialization. Treat foreign model chat delimiters in attack data as untrusted content with recorded provenance. Reject or explicitly account for overlength requests; never silently score only a prefix while generating from a different full input.

Artifact manifests must bind:

```text
model ID and revision; tokenizer revision and template hash;
dtype/quantization; activation site and indexing convention;
pooling and token position; truncation and padding;
training/development/calibration IDs and dataset revisions;
raw probe weights, bias, preprocessing and score transform;
threshold/aggregation/persistence configuration;
code commit, seed, dependency versions and artifact hashes.
```

Validate these fields before inference, not only hidden dimension. The v2 raw-LogisticRegression artifacts and older unit-normalized `PolicyDirection` artifacts need explicit import conversion rather than interchangeable loading. Keep detector weights and normalized actuator directions as distinct artifact types.

For a non-unit vector, an exact projection onto the probe half-space requires division by `||w||²`:

```text
h_new = h - max(w·h + b - threshold, 0) * w / ||w||²
```

This only establishes the probe inequality at that site. It does not establish safe behavior. Some existing intervention functions assume unit directions; raw legacy coefficients must not enter those operators unchecked.

The current V5 resume logic skips files by row ID/existence. It does not bind those cached trajectories to changed hook code, probe hashes, serialization, or thresholds. A corrected function can therefore coexist with stale cached outputs. Make cache keys depend on the full relevant experiment identity, use atomic writes, and reject incompatible resumes. The latest report prints `None` for probe hashes, and the exported manifest lacks those probe-hash entries. The locked split file referenced by V5 is not available as a tracked standalone file in this checkout; provide it or an immutable downloadable artifact and a verified reconstruction command.

`neuralFirewallV2/src/` currently consists of eight empty `__init__.py` files. Its reference YAML and requirements file are empty. The actual latest implementation is duplicated across notebooks and roughly 4,000-line Python exports. The v2 README's proposed reusable architecture has not been implemented there yet. Root docs and experiment numbering also disagree: for example, the detailed log's Exp003 is a category-held-out negative result, while `results/exp003.md` describes policy-vector discovery.

For a GitHub release, supply one installable package, pinned tested environments, a runnable CPU example, a reference-model command, an experiment index, artifact checksums, and a code license plus citation metadata. Replace blanket `*.csv` ignores with targeted generated-output rules so new split tables and small metric tables are not silently omitted. Keep heavyweight results in versioned releases/object storage with download manifests. Preserve historical experiments as an archive rather than using them as multiple competing production implementations.

**6. The central conceptual correction**

The current implementation often treats “less detectable unsafe intent” as a desirable intervention target. That is not necessarily a safety improvement. Removing information used by the detector can lower its score while preserving harmful behavior, or suppress the model's ability to recognize that refusal is warranted. The known distinction between a readable feature and a causal actuator is central here. Refusal-direction work shows that erasing a direction can remove refusals, illustrating why intervention sign must be established experimentally. [Arditi et al.](https://arxiv.org/abs/2406.11717)

Use separate objects:

```text
Trusted policy configuration: what behavior is authorized
Contextual risk state: evidence about this request/continuation
Actuator: a tested causal modification of model behavior
Output gate: which tokens/actions are actually released
```

The mathematical framework asks for both `P ≈ g(policy, relevant context)` and `I(P; U | policy) ≈ 0`. Those conditions need clarification. Contextual risk must depend on user content. Protect the policy specification from unauthorized writes, while allowing its application to depend on the request. A constant detector satisfies a naive invariance objective without doing useful safety work. Also, deleting the direct `U -> P` edge does not prevent influence through `U -> C -> P` in the proposed `P'=f_P(C,P)` update.

A clearer formulation is:

```text
q = E(trusted_policy, trusted_configuration)
s_t = Monitor(q, request, lower-trust context, model_state_t)
a_t = Controller(q, s_t)
```

Here `q` has an enforceable write restriction. `s_t` is explicitly input-dependent. Freezing `q` is an engineering property, not proof that the model obeys it. The research question becomes whether policy-conditioned monitoring/control improves authorized behavior under attack compared with ordinary prompt hierarchy training, an external guard, and an unconditioned activation probe.

Run controlled policy experiments on harmless tasks: change tenant permissions, allowed fields, output constraints, or access to synthetic secret canaries. Hold request text constant while changing trusted authorization; then hold authorization constant while inserting a conflicting instruction in a lower-trust document. Test authorized changes and unauthorized overrides separately. Such experiments can identify authority sensitivity much more cleanly than another safe/unsafe topic classifier.

**7. Recommended implementation sequence**

**Milestone A — one correct runtime, before further GPU sweeps.** Consolidate the older module boundaries and V5's validated extraction/calibration code into `neuralFirewallV2/src/`. Add `pyproject.toml` and expose a package such as `nps_firewall` with these components:

| Component | Contract |
|---|---|
| `serialization.py` | Canonical role-preserving messages and tokenization |
| `activation_sites.py` | Explicit site metadata and shared offline/runtime capture |
| `artifacts.py` | Validated immutable detector/controller manifests |
| `monitor.py` | GPU-resident prompt or continuation risk scoring |
| `controller.py` | Separate causal intervention with logged pre/post state |
| `runtime.py` | Owns generation, hooks, cache lifecycle, output release, and terminal errors |
| `data.py`, `splits.py` | Stable IDs, source/behavior groups, deduplication, frozen partitions |
| `evaluation.py` | Actual output judgments, utility, failure accounting, confidence intervals |

Required checks: zero-strength intervention matches baseline logits/tokens; DETECT mode preserves deterministic output; offline and runtime activations agree at the same site and prefix; cached and full-prefix decoding agree within documented tolerance; left/right padded batch extraction agrees with single-example extraction; hooks are removed after exceptions; voting truth tables and missing/NaN states behave correctly; no token is released after a block decision. These tests should use a tiny real transformer as well as a mock, with a small reference-model integration check before the expensive run.

Store probe tensors on the same device as monitored activations. V5 currently copies layer vectors to CPU on every decoding step; the older package also round-trips through NumPy. Score on-device and move only needed decisions/telemetry. Benchmark this optimization rather than assuming a target overhead.

**Milestone B — a valid monitor-plus-block baseline.** Freeze the reference model, serializer, splits, evaluator, and runtime. Generate actual baseline continuations from diverse safe and unsafe requests. Fit a prompt detector for intent and a separate continuation detector for harmful assistance; retain prompt-only and continuation-only ablations. Include safe refusals of harmful requests and benign discussions of risky topics as negatives for the continuation target.

Train continuation features on positions and prefixes that will exist at inference. A state used to predict token `t` has not yet seen that token; align labels accordingly. Full-response labels can supervise response-level prediction but should not be indiscriminately assigned to every token. Use annotated prefixes/spans or an explicitly response-level objective, and report the limits of any weak token supervision.

Calibrate the complete deployed decision rule on held-out trajectories: all layers, aggregation, persistence, prompt gate, maximum generation length, and stopping behavior. Per-layer or per-token FPR is not an end-to-end false-block guarantee. Lock this policy before final evaluation. Report raw margins/ranking and operating-point recall separately; sigmoid scores are not automatically calibrated probabilities on shifted distributions.

Start with blocking as the simplest enforcement baseline. For offline research runs, buffer generated output until its release decision so the exact released content is auditable. For streaming claims, assess the already-released prefix; a late stop cannot retract previous tokens. A short buffer or persistence rule is a latency/security tradeoff, not a proof of prevention. Define tool-call authorization at the action boundary if evaluating agents.

**Milestone C — establish an actuator independently of its detector.** Test a refusal-reinforcing direction or a small trained residual adapter against baseline, random norm-matched intervention, opposite-sign intervention, unconditional intervention, risk-gated intervention, and block-only monitoring. Select layer/rank/strength on development data using actual harmful-assistance and benign-utility outcomes. Keep pre-intervention monitoring separate from post-intervention diagnostics to avoid “success” from blinding the reader.

If training an adapter, use safe target responses for unsafe training requests and a benign retention objective, such as task loss plus KL to baseline on benign contexts. Test whether effects survive later layers and decoding steps. A paired activation-patching study can help localize causally useful sites before a broad head sweep. Circuit-breaker representation training is an essential comparator for this claim. [Zou et al.](https://arxiv.org/abs/2406.04313)

If a controller changes an already-cached prefix or regenerates earlier tokens, rebuild/restore the affected KV cache and sampling state. Otherwise the resumed computation does not correspond to the intervention being claimed. Record interventions across the trajectory rather than the final overwritten `captured["score"]` and `captured["intervention"]` values used in the local causal pilot.

**Milestone D — an explicit NPS experiment.** Only after A–C, implement a trusted policy encoder/read-only policy stream and compare it with the same monitor/controller without protected policy conditioning. If using policy tokens or recurrent policy memory, enforce their write restriction at every relevant transition; initialization alone is insufficient. Untrusted evidence can feed a separate decision stream. Evaluate authority counterfactuals, policy swaps, lower-trust role spoofing, and ordinary task utility. State which noninterference property is enforced by code and which behavioral robustness remains empirical.

**Milestone E — adaptive evaluation and release.** Give the evaluator/attacker the deployed defense specification appropriate to the declared threat model. Evaluate attack families that optimize both behavioral success and detector evasion, transfer attacks, held-out domains, long contexts, multilingual inputs, and multi-turn inputs if supported. Keep continuous embedding/activation attacks as a distinct stress-test class when deployment only permits text input. Include failures and attack budgets. Fixed jailbreak collections alone cannot support a robust-boundary claim: adaptive work has demonstrated that harmful behavior can persist while latent monitors are evaded. [Bailey et al.](https://arxiv.org/abs/2412.09565)

Defer RMU/capability hardening until the runtime and evaluation are stable. It is a different intervention with its own retention and recovery questions. Adding it now would make it harder to identify whether any improvement came from detection, refusal, incapacity, or generic degradation.

**8. Minimum credible experimental matrix**

| Axis | Required comparisons |
|---|---|
| Defense | Base model; input/text guard; output guard; activation block-only; unconditional steering; gated controller; protected-policy variant when implemented |
| Security outcome | Harmful assistance in the actual released output; unauthorized task/action success for injection |
| Utility | Ordinary task correctness; difficult benign safety-adjacent requests; benign refusal and false blocks |
| Generalization | In-distribution, held-out domain/source, held-out attack family, and an independent final test |
| Adaptive robustness | Defense-aware attacks with explicit access/query/optimization budgets and repeated trials |
| Mechanism | Random/sign/site controls; reader versus actuator ablations; post-intervention independent evaluator |
| Compute | Time to first token, fixed-length ms/token, request latency, memory, GPU-hours and environment |

HarmBench and StrongREJECT provide relevant behavioral evaluation infrastructure; XSTest targets excessive safety behavior. Keep XSTest as an external test in the new protocol if possible; the existing XSTest-trained runs must retain their documented training use rather than be relabeled benchmark-generalization results. For injection, use role-aware task evaluation such as AgentDojo once the reference model can perform the underlying tasks. [HarmBench](https://arxiv.org/abs/2402.04249), [StrongREJECT](https://arxiv.org/abs/2402.10260), [XSTest](https://arxiv.org/abs/2308.01263), [AgentDojo](https://arxiv.org/abs/2406.13352).

Do not run the complete Cartesian product first. Use one reference model and a small development run to verify coverage, base-model vulnerability, judge validity, and utility. For a broader mechanism claim, replicate the selected protocol on a second model family using separately fitted artifacts. Bootstrap by underlying behavior/prompt group, not correlated tokens; include seed/attack variability. Report source/class denominators and evaluation failures. Choose sample sizes from the precision needed for the claimed FPR/ASR rather than convenience.

Proposed engineering gates, to choose and freeze before final evaluation: at least 50% relative reduction in independently measured ASR versus baseline, at most 2% added benign false blocks, at most two percentage points lost task accuracy, and at most 15% monitoring overhead under matched fixed-length generation. These are suggested decision criteria, not established research standards or predicted results. Include uncertainty; a dataset on which baseline ASR is near zero cannot meaningfully support the relative-reduction gate.

The latest 20-sample latency result of negative overhead is compatible with measurement noise and is not evidence of a free defense. Warm up both paths, synchronize device timing, randomize measurement order, match generated lengths, and report distributions. End-to-end savings from early blocking should be reported separately from monitoring cost.

**9. A publishable scope**

For GitHub, a defensible first release is **“Activation-based runtime monitor for Qwen, with reproducible evaluation and documented failure modes.”** It can be useful without claiming universal safety or privilege separation.

For a paper, the strongest routes are:

1. **A rigorous empirical study:** how harmful-intent probe quality transfers—or fails to transfer—to prediction and prevention of actual harmful continuations, under matched utility and adaptive attacks. The existing history motivates this, but the corrected controlled evaluation must supply the contribution.
2. **A narrower NPS method:** a policy-conditioned runtime controller with an enforced trusted-state write restriction that improves instruction-authority preservation on counterfactual policy tasks and held-out injections. The comparison must isolate the value of the protected state.

“We found a safety direction and stopped generation above a threshold” is unlikely to establish sufficient novelty by itself. Position against refusal directions, circuit breakers, instruction-hierarchy training, and prompt-injection defenses such as SecAlign, not only RepE/ITI/RMU. [Instruction Hierarchy](https://arxiv.org/abs/2404.13208), [SecAlign](https://arxiv.org/abs/2410.05451).

A current literature check also surfaced closely adjacent 2026 work on activation disentanglement and cross-layer trajectory detection. Inspect their full methods and compare before claiming novelty from adding a nonlinear representation or trajectory monitor. This audit is not an exhaustive novelty review. [FrameShield/ReDAct](https://arxiv.org/abs/2602.19396), [Manifold Trajectory Kinetics](https://arxiv.org/abs/2606.07335).

The original charter can remain the long-term objective. The next concrete deliverable should be a correct shared runtime plus a frozen, complete, target-model behavioral evaluation. Its results should determine whether the following paper is about an effective controller, an authority-preserving architecture, or a carefully measured limitation of latent monitoring.
