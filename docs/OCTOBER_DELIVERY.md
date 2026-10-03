# October firewall and paper delivery

Recorded on 2026-10-03. This is the current status of both October plans; older roadmap documents describe their earlier checkpoints.

## Working deliverables

- **Authenticated local model API:** [start the pilot](../neural_state_firewall/PILOT.md). Host-owned policy, documents and grants; a separate inference worker; response buffering; a parent-enforced deadline; no response text on runtime failure. The caller cannot grant access to a source. This serves one permission realm on loopback. Tool effects still require the separate NPS broker.
- **Deterministic model boundary:** host-denied document tokens cannot influence public output states under the supported attention-mask, position, cache and runtime assumptions. This is implemented at every layer and cached step. Readable documents can still influence instructions and answers.
- **Evidence package:** reproducible BIPIA analysis, two figures, a simpler host-filtering baseline, randomized structural checks, a second pretrained architecture check, and real-model HTTP smoke results. Raw BIPIA records remain local and ignored.
- **Research package:** [PDF](neural_state_firewall_paper/manuscript.pdf), [LaTeX](neural_state_firewall_paper/manuscript.tex), bibliography, protocol, reproducibility commands, cover letter, author/venue metadata stubs and [journal shortlist](neural_state_firewall_paper/JOURNAL_SHORTLIST.md). Author: Pratham Vasa. Working target: SoftwareX; alternatives include Software Impacts, PeerJ Computer Science, Cybersecurity, JISA and IEEE Access. TIFS is a stretch option. Fit is not an acceptance prediction.

## Results and their limits

| Check | Recorded result | What it establishes |
|---|---|---|
| Firewall regression suite | 76 tests passed | Tested runtime, permissions, API, worker and evidence-validation behavior, including preservation of existing reviewer labels |
| BIPIA-derived denied-content swaps | 175/175 invariant | Same input shapes, per-step logits and greedy tokens when denied payloads change |
| BIPIA generation accounting | 1,750 attempts; 1,657 EOS; 93 capped; zero errors | Complete denominators, including alternate denied payloads |
| Readable BIPIA attacks | Four inspected counterexamples in the permission arm | Readable attacks can redirect output; no aggregate semantic attack-success rate was scored |
| Remove denied documents before ordinary decoding | 175/175 clean token sequences and release outcomes matched | A simpler host filter suffices for this measured property; no utility advantage from retaining masked denied inputs was demonstrated |
| Tiny Qwen2/Llama, eager/SDPA | 100/100 denied swaps invariant; 100/100 public perturbations changed logits | Cross-backend structural engineering coverage, not pretrained behavioral efficacy |
| Pretrained SmolLM2-135M (Llama) | 7/7 structural checks; 28 EOS; zero errors; exact task score 0/7 per arm | A second architecture executes the boundary, but the frozen task-output metric fails |
| Authenticated real-model API | 6/6 smoke checks passed | Authentication, readiness, allowed/private answers, rejection of forged grants and caller context |
| New readable-injection development diagnostic | 56 retained attempts; zero exact attack markers in either arm; 51 EOS and five timeout errors | No measured baseline susceptibility. Earlier concurrent launches confound timing and total attempt accounting; no efficacy claim |

The challenge executor discovered and stopped three leftover duplicate processes from earlier launches. Those launches have no retained per-attempt records, so the total number of attempts is unknown. The saved 56-attempt artifact is unchanged, but timing and timeout attribution are confounded. No templates or thresholds were retuned. This is an operational deviation, not a clean fixed-total-budget study; it is excluded from positive security evidence.

The JSON/CSV analysis and both figures were regenerated from the private raw artifact and matched the saved files byte for byte. [Verification metadata](../neural_state_firewall/validation/verification_20261003.json) records commands, environment and source hashes. Syntax compilation and whitespace checks passed; no lint/typecheck configuration or installed Ruff/Mypy was available for this module. The environment used Python 3.11.12, Torch 2.6.0 and Transformers 4.57.6 on an Apple M2 with 8 GiB RAM, CPU float32 inference.

## Product plan status

| Planned stage | Status | Remaining work |
|---|---|---|
| 1. Freeze security contract | Complete for the local pilot | Extend the contract when introducing real application principals, retrieval or remote deployment |
| 2. Interpret BIPIA | Structural analysis complete; semantic review partial | The blinded 525-candidate packet has blank labels. Complete a disclosed single-review study before making semantic-rate or utility claims |
| 3. Add authority boundary | Denied-source boundary and host-owned request boundary implemented | Free-form readable-evidence instruction takeover remains unresolved. Tools must still obtain broker authorization at the effect point |
| 4. Fresh challenge and release gates | Development challenge supplied; locked final test pending | Protect genuinely unseen, source-separated data and freeze measurable utility/security gates before accessing results |
| 5. Paired/adaptive evaluation | Structural controls and host-filter baseline complete | Representative behavioral evaluation and a disclosed-defense adaptive test with a fixed attack budget |
| 6. Harden deployable path | Single-realm local pilot complete | Remote ingress/TLS, deployment-specific identities, load/memory/logging-failure tests and operational ownership |
| 7. Release decision | Recorded: local engineering pilot only | General production approval is false; no canary or external deployment was made |

The broad firewall objective is not complete. Readable evidence is the remaining security problem, and structural noninterference cannot substitute for solving it. The optional neural anomaly monitor has no validated permission-path release gate and is disabled in the pilot.

## Paper plan status

| Planned stage | Status | Remaining work |
|---|---|---|
| 1. Claim and venue | Complete as a working decision | Verify SoftwareX's current full author guide/template; official retrieval was blocked |
| 2. Study protocol | Revised and dated | Human labels remain absent; single-review limitations are explicit |
| 3. Analysis and figures | Complete for recorded structural results | Add semantic results only after actual review |
| 4. Additional study | Structural stress, pretrained Llama and host-filter baseline complete | Broader behavioral/adaptive study is required for an efficacy paper, not claimed in this characterization |
| 5. Manuscript rewrite | Complete draft with negative results and close prior art | Author scientific review and final venue adaptation |
| 6. Reproducibility and author checks | Local reproduction/build checks complete | Clean-checkout reproduction; author affiliation/email/ORCID, funding, conflicts, ethics, contributions, license and public archive details |
| 7. Submission | Package prepared locally; not submitted | Final template, author attestations, permitted code/data release and actual submission receipt |

The paper does not claim novelty over CIV/DeTAM, does not equate changed answers with successful attacks, and does not claim independently annotated outcomes. October 31 remains a submission target; journal acceptance and publication dates cannot be guaranteed. The remaining author details are visible stubs as requested, not invented metadata.

## Resources still needed

- A representative Q&A workload and actual host-owned access policy. The synthetic pilot establishes mechanics, not business utility.
- Fresh source-separated evaluation data with usable licenses and untouched test custody. All consumed BIPIA data and the synthetic fixtures are excluded from future held-out claims.
- One real reviewer under the disclosed protocol; a second reviewer is not assumed available. A model judge would be a separately disclosed automated analysis, not human confirmation.
- Adequate memory/compute for larger pretrained models and adaptive attack budgets. Current evidence uses small CPU models; no paid API service or GPU platform was provisioned.
- Author metadata and declarations, a chosen software license/public repository or archival release, and the selected journal's current submission requirements and any applicable publication budget.

See the [machine-readable release decision](../neural_state_firewall/validation/release_decision_20261003.json) for the exact scope and evidence hashes. The source bundle is a local review package; it contains no submission receipt and is not an attested journal submission.
