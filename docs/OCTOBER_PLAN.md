# October 2026: firewall product and paper submission plans

**Plan date:** 2026-10-02
**Target:** an implementable, evidence-bounded firewall and a journal submission by 2026-10-31. Journal publication or acceptance by that date is outside the authors' control; the achievable deadline is submission.

## Current position and fixed boundaries

- The runtime enforces host-assigned document read permissions inside Qwen2/Qwen2.5 and Llama attention (`neural_state_firewall/read_permissions.py:58-60,110-181`). It has one pretrained Qwen2.5 evaluation; pretrained Llama behavior is untested (`neural_state_firewall/READ_PERMISSIONS.md:54-60`).
- The 175-case BIPIA-derived run found identical shapes, per-step logits and outputs across denied-payload swaps (175/175); it did **not** score semantic attack success. Readable attacks changed ordinary outputs in 42/175 cases, which is not an attack-success measure (`neural_state_firewall/READ_PERMISSIONS.md:78-99`).
- The separate state-trajectory monitor can buffer and suppress responses, but its current evidence includes both a benign false alarm and an allowed injection-marked case; it has no independent behavioral labels (`neural_state_firewall/PRODUCTION_ROADMAP.md:5-17`, `docs/neural_state_firewall_paper/manuscript.tex:23-25`).
- The NPS broker remains on the tool/action path. The model-API firewall does not execute or authorize tools (`neural_state_firewall/READ_PERMISSIONS.md:31-33`).
- Existing BIPIA data/results and all prior fixtures are now seen development material; none can be reused as a fresh locked test (`docs/neural_state_firewall_paper/data/seen_material_exclusions.json`).

The central product question is whether a firewall can stop untrusted evidence from exercising authority, rather than merely detect unusual text or prevent access to a host-denied document. Readable evidence can currently influence the model. Until an enforcement mechanism and held-out behavioral evidence cover that path, describe the system as a permission-boundary firewall with an experimental response monitor, not a generally prompt-injection-resistant model.

## Plan A — build the firewall product

### Outcome

By October 31, produce a pilotable model-API firewall with an explicit trust model, host-controlled source permissions, response release checks, and a separately mediated action path. Make a release decision from predeclared evidence. If readable-evidence injection remains unblocked, ship only the narrow permission-boundary capability in monitor-only/closed pilot and state the uncovered behavior plainly.

### Steps and dates

1. **Oct 2–4 — Freeze product security contract.** Name one initial workload (document-grounded Q&A), principals and source roles, protected assets, attacker controls, and allowed outcomes. Define three separate properties: (a) denied-source noninterference; (b) readable-source instruction-takeover resistance; (c) unauthorized action prevention by host/broker authorization. Define what the firewall promises when its monitor times out or fails. Do not call property (a) proof of (b). Update `NPS_PERMISSION_BOUNDARY.md` and the release section of `READ_PERMISSIONS.md`.

   **Accept when:** each source, policy, secret and effect has an explicit authority owner; request data cannot grant itself a permission; tests can distinguish each of the three properties; unsupported model/runtime combinations fail closed.

2. **Oct 3–7 — Interpret the BIPIA run as development evidence.** Review the paired ordinary, isolated-memory and read-permission outputs for all readable-attack cases using an explicit rubric: task maintained, unauthorized instruction followed, protected data exposed, ambiguous, or incomplete. Review paired clean outputs too if making a utility claim. Use the documented single-review protocol because two independent human reviewers are unavailable; blind condition/order where possible, preserve ambiguous cases, and disclose that this is not independent annotation. If an official or model judge is used, report it as a separate automated sensitivity analysis, never as independent human confirmation. Record results in a sanitized table and retain raw outputs locally.

   **Accept when:** all 175 paired readable-attack cases are reviewed across the three arms, or any sampling rule is predeclared; any utility claim also has paired clean-output judgments; results include denominator, omissions, EOS/truncation, and label uncertainty; no response-change count is called attack success.

3. **Oct 5–11 — Specify and implement the missing authority boundary.** Keep host-assigned document permissions for data confidentiality. For any operation that can reveal a secret or cause an effect, require a host-controlled capability decision outside natural-language model instructions; tool proposals continue through the NPS broker. Define which response violations can be checked deterministically (schema, protected-value disclosure, allow-listed claims) and which remain semantic/monitor-only. Do not add another classifier unless a measured gap requires it. If the target is ordinary free-form answers, document that no deterministic semantic authorization rule currently guarantees resistance to readable prompt injection.

   **Accept when:** tests demonstrate denied-source isolation, rejection of caller-supplied provenance/grants, no partial release on errors/timeouts, and broker authorization at the effect point; readable attacks cannot directly add grants or invoke tools; limitations are explicit in CLI/API docs.

4. **Oct 8–14 — Build a fresh development red-team set and freeze gates.** Source-separate new attack/benign pairs (not BIPIA or any listed seen material); include quoted attacks, benign security content, instruction conflicts, requests for protected facts, long/noisy evidence, and adaptive low-anomaly attempts. Freeze model revision, prompt serialization, policy, decoding, monitor/profile, label rubric, thresholds and analysis before the test. Define minimum baseline attack susceptibility, task utility, false-block, incomplete-output and latency gates; use cluster-aware uncertainty. Keep this set development-only if implementers see results.

   **Accept when:** provenance/licenses, split hashes, source/template grouping and attack budget are recorded; baseline failures prove the benchmark can measure the target; gates and test partition are frozen before final labels/results are accessed. If no fresh test can be protected by the date, do not call the development set held-out.

5. **Oct 15–20 — Run paired evaluation and adaptive challenge.** Compare ordinary model, deterministic permissions, optional anomaly monitor, and full firewall on paired requests. Score semantic attack success, denied disclosure, task success, benign false blocks, no-partial-release failures, EOS/incomplete outputs, resource use, and end-to-end latency separately. Run a disclosed-defense adaptive set with a fixed budget. Re-evaluate the denied-content invariant under each supported configuration.

   **Accept when:** the run is reproducible from pinned code/model/data manifests; every denominator and missing case is reported; adaptive and standard attacks remain separate; no threshold is tuned on the locked test.

6. **Oct 21–25 — Harden the deployable path.** Exercise cancellation, deadlines, malformed requests, long/oversize documents, concurrency, reload/restart, identity/profile mismatch, memory pressure, logging failure and cross-request state leakage. Add only the minimum deployment controls required for the named pilot: authenticated host ingress, principal-bound retrieval/grants, bounded request/worker resources, privacy-safe audit, health/readiness and monitor-only rollback. Keep loopback demo separate from a public service. Mark model support adapter-specific; adding an abstraction is not evidence of portability.

   **Accept when:** integration tests prove no unauthorized text/action releases under injected runtime failures, tenant/principal isolation is tested, resource ceilings and rollback work, and deployment documentation names supported exact revisions.

7. **Oct 26–31 — Release gate and pilot decision.** Freeze exact code, model, tokenizer, adapter, policy, permissions, monitor/profile and analysis hashes. Compare results against the frozen gates. If all security, benign-utility, operational and adaptive gates pass, make a limited internal canary with sampled blinded review and immediate rollback. If any gate fails or evidence is underpowered, keep enforcement disabled for general traffic and release only the narrow host-permission boundary in a closed pilot.

   **Accept when:** signed/versioned release identity, reproduction commands, failure behavior, rollback and a written pass/fail/insufficient-evidence decision exist. No general “prompt-injection-proof” claim is allowed.

### Product risks and stop rules

- **Readable evidence still controls behavior.** If no deterministic authority boundary or evaluated monitor prevents this in the chosen workload, stop broad-resistance claims; constrain effects and protected data, or keep the semantic response path monitor-only.
- **The trajectory monitor may not predict harmful behavior.** Treat it as an IoC, not the security policy. If it misses attacks or blocks benign work beyond the frozen limits, disable enforcement for that monitor rather than tuning on test failures.
- **Deadline pressure may compromise evaluation.** Reduce product scope to one workload and one pinned model; do not lower data/label standards or relabel development cases as held out.
- **Model-agnostic operation is limited.** Maintain a provider/adaptor boundary, but declare only individually tested model/runtime combinations supported.

## Plan B — submit the research paper by October 31

### Outcome

Submit a reproducible, accurately scoped manuscript by October 31. The current manuscript is a six-case study of a temporal activation monitor; the newer BIPIA evidence is about a distinct deterministic read-permission mechanism. Do not combine their claims without rewriting the research question and methods. Preferred paper scope: a systems/measurement paper on host-enforced document read permissions and its precise noninterference property, with readable-injection behavior reported as unresolved or as a separately labeled exploratory result. A stronger efficacy paper requires fresh semantic, preferably adaptive evaluation and may not be supportable by October 31.

### Steps and dates

1. **Oct 2–4 — Choose the paper's claim and venue.** Decide whether the manuscript is (a) a deterministic permission-boundary systems paper, (b) an empirical temporal-monitor paper, or (c) a carefully separated comparison of both. Do not leave the present monitor title/abstract attached to permission-boundary results. Select one primary journal and one fallback based on scope and current instructions. Treat October 31 as submission, not publication; editorial review and acceptance cannot be scheduled by the authors.

   **Accept when:** title, abstract, research question, threat model, primary contribution and target readership all describe the same implemented mechanism and evidence.

2. **Oct 3–8 — Amend and freeze study protocol.** Resolve the reviewer-resource constraint transparently: one blinded human reviewer may provide case labels, with an explicitly disclosed protocol deviation and, if available, a separate automated judge sensitivity analysis. Do not claim inter-rater reliability or independent human adjudication. Define label handling for disagreements/ambiguity, missing responses, output caps and failed cases. Lock outcomes, exclusions, statistical analysis and all permitted BIPIA-derived analyses before additional scoring.

   **Accept when:** protocol version, claim-to-measure map, data partitions and analysis code are dated and hashed; BIPIA is identified as seen development material; single-review limitations are visible in Methods/Limitations.

3. **Oct 5–12 — Complete analysis and figures.** Validate all 175 result rows and source/model identities; produce a compact result table for denied-content invariance and separate descriptive tables for readable attacks, task completion and token-limit cases. If semantic judgments are obtained, show counts and uncertainty with the disclosed reviewer method; do not infer efficacy from response changes or keyword matches. Keep raw model outputs out of the paper repository unless licensing/privacy permits them; release sanitized aggregates, hashes and code.

   **Accept when:** every plotted number reproduces from the raw local artifact and a committed script; labels, outcomes and denominators are visibly distinct; no raw corpus/completion text is accidentally included.

4. **Oct 9–18 — Run only the fresh study needed for the chosen claim.** For the deterministic-boundary paper, expand structural checks across supported model/runtime conditions and document the mechanism assumptions; include only completed, verified pretrained claims. For an empirical prompt-injection-resistance paper, create a fresh source-separated test and adaptive evaluation, freeze thresholds before access, and collect behavioral outcomes. If fresh data, review, or sample size cannot be completed, narrow the article to a system characterization and report that limitation rather than claim efficacy.

   **Accept when:** protocol gates pass before the final run; all results can be reproduced; unsupported comparisons and incomplete model configurations are omitted or explicitly marked.

5. **Oct 16–23 — Rewrite the manuscript around the evidence.** Revise `docs/neural_state_firewall_paper/manuscript.tex`, tables, figures, bibliography and README. Update literature review with primary sources and distinguish this work from attention monitoring, instruction hierarchy, provenance marking and capability security. Describe the formal/structural assumptions and counterexamples; state that readable evidence may still influence the model. Report negative results and all protocol deviations. Avoid asserting novelty until a documented literature search supports it.

   **Accept when:** abstract claims are traceable to a result table; Methods includes exact revisions, hardware/software, selection, labels and analysis; limitations name single-review and sampling constraints; no invented or unsupported result appears.

6. **Oct 24–27 — Reproducibility and author checks.** Run the full test suite, analysis from clean artifacts, LaTeX build, reference/URL audit, figure/source validation and manuscript consistency review. Complete author identities/order/affiliations, funding, conflict, ethics, data/software availability and AI-use disclosures as applicable. The target journal's current instructions govern formatting and disclosure; the [OUP Journal of Cybersecurity guidelines](https://academic.oup.com/cybersecurity/pages/general_instructions), if selected, request source files and data/software availability where ethically feasible and require disclosure of AI use.

   **Accept when:** final PDF compiles without missing references/figures; every result is regenerated; code/data access statements match what can legally be shared; all author and ethics declarations are complete.

7. **Oct 28–31 — Submit and archive.** Apply final journal template and metadata; prepare cover letter, abstract, keywords, figures, supplementary methods/code and disclosure statements. Submit no later than October 31; archive the submitted source bundle, PDF, code/data hashes and confirmation. If the preferred journal's rules or scope do not fit, use the preselected fallback rather than miss the deadline.

   **Accept when:** submission receipt is saved and the exact submitted bundle is reproducible. Acceptance/publication by Oct 31 is not an acceptance criterion.

### Paper risks and stop rules

- **Evidence does not support a high-efficacy prompt-injection paper.** Submit a narrow system/security characterization or wait for stronger data; do not convert an engineering invariant into semantic attack resistance.
- **Single reviewer weakens outcome validity.** Disclose it and avoid reliability claims. If this makes the target journal's evidentiary standard unattainable, choose a venue/article type that permits a transparent preliminary systems contribution, or submit after additional review becomes available.
- **Fresh held-out data cannot be obtained before the deadline.** Use the BIPIA run only as seen development evidence and make the paper about the verified structural property; no new holdout performance claim.
- **Journal decision timing is external.** Commit to complete submission by Oct 31; do not promise publication by then.

## Shared decision gates

- **Oct 4:** freeze project claim and manuscript scope; separate deterministic source permissions from semantic behavior monitoring.
- **Oct 14:** stop/go on whether a valid fresh behavioral study and single-review protocol can be completed; if not, narrow the product and paper claims.
- **Oct 20:** decide monitor enforcement from frozen results; negative or inconclusive results mean monitor-only/closed pilot.
- **Oct 27:** final paper evidence and formatting audit; no new analyses after this without updating every table and claim.
- **Oct 31:** firewall limited-release decision and journal submission receipt.

## Verification evidence already available

- BIPIA result: 175 cases, 175/175 denied-payload invariance checks, no semantic attack-success score; local raw artifact `neural_state_firewall/artifacts/bipia_read_permission_20261002.json`.
- Regression suite: 67 tests passed when loopback binding was permitted; suite is an engineering check, not a prompt-injection efficacy result.
- Current paper draft and protocol: `docs/neural_state_firewall_paper/manuscript.tex`, `STUDY_PROTOCOL.md`, `README.md`.

## October 3 execution checkpoint

This dated plan records the starting point. The current implementation, evidence and remaining work are in [OCTOBER_DELIVERY.md](OCTOBER_DELIVERY.md). The local pilot is implemented; the broader readable-injection objective remains unresolved. The paper package is a draft and has not been submitted.
