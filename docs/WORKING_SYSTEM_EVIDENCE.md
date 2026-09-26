# From research to a working system

The implementation question was: **what can this repository support as an
implementable system today, without another experiment series?**

The repo contains two distinct prototypes. The **prompt-injection firewall** is
the request/response path in `nps_gateway.firewall` and `nps_gateway.proxy`; the
**action broker** is a separate host-authorized tool-effect control. The broker is
supporting infrastructure and does not inspect ordinary generated responses. The
firewall targets the original injection-defense request by labelling retrieved
context as untrusted and holding the model response for a policy judge. The review covered the
historical experiment log and implementation audit, earlier prototype, NFW-001–011
reports/results, current reference broker, and its contract tests. Archived runs
were inspected rather than retrained or rerun on a GPU.

## Ranked conclusions

| Rank | Conclusion | Confidence and basis |
|---|---|---|
| 1 | Resource scope alone cannot protect written content. | High: NFW-010/011 directly record seven wrong-content writes under scope-only permissions. |
| 2 | Exact source-bound permission is useful for a literal-copy workflow. | High within that task: NFW-011 removes those seven effects while preserving clean task success. |
| 3 | Neural activation scores should not grant authority. | High: historical transfer/recall weaknesses and the independent NFW-004/005 broker contracts. |
| 4 | Strict provider protocol handling is part of a usable system. | High: NFW-009's confounding and NFW-010's failed 0.5B format gate. Invalid outputs are failures, not evidence of safety. |
| 5 | Durable transactional enforcement protects tool effects when the application offers tools. | Engineering inference from the identified restart, provenance and effect gaps; it does not stop prompt injection in ordinary text responses. |

The new firewall live check passed one clean task and one injected-context case
with Qwen 3B: it released the ordinary answer and withheld the injected case.
The same model family generated and judged the answer. This is direct evidence for
one task-level gate run, not evidence of broad resistance. The endpoint is
text-only today and rejects model tool calls; broker integration into an agent's
tool path remains to be built.

## Evidence and resulting implementation choices

| Work reviewed | Direct evidence | What the system retains |
|---|---|---|
| [Historical audit](NPS_IMPLEMENTATION_AUDIT.md), [experiment log](../EXPERIMENT_LOG.md) | Early nearly perfect probing was topic-confounded; category-held-out Exp003 best AUC 0.561. Exp017 clean recall 48/64, adversarial recall 44/100. The old runtime had no-op intervention and fail-open missing-state paths. | No universal detector, no unvalidated steering, explicit denied/error/executed states and real side-effect checks. |
| [NFW-001 forensic report](../neuralFirewallV2/results/NFW-01_firewall_v0/drive-download-20260915T104150Z-1-001/FORENSIC_REPORT.md) | Prompt harmful detection 25%, jailbreak detection 5.57%, OOD AUROC 0.5687. External behavioral labels did not label the target model's generated responses. | Authorization depends on trusted policy, not probe success or external response labels. |
| [NFW-002 v2 report](../neuralFirewallV2/experiments/NFW-02_foundation_firewall/nfw002_v2_necent_001/final_report.json) | 113 generated examples; intent AUROC 0.637; harmful responses 16 baseline versus 13 under blocking; benign success 11/12 in both. Marked provisional with AI-assisted review. | Preserve utility and terminal failure accounting; do not promote the monitor into a deployed safety guarantee. |
| [NFW-003 report](../neuralFirewallV2/experiments/NFW-03_behavioral_proxy/nfw003_necent_002/REPORT.md) | Activation AUROC 0.6254, harmful-response recall 0.1765, safe-response block 0.2105. Historical-response proxy; generation audit not run. | Optional monitoring is veto-only. No pretrained monitor is enabled by default. |
| [NFW-004](../neuralFirewallV2/experiments/NFW-04_veto_capability_firewall/README.md), [saved archive](../neuralFirewallV2/experiments/NFW-04_veto_capability_firewall/nfw-04-results.zip) | Archived JSON records 11/11 broker cases passed and `model_can_mint_capability=false`; no external actions. Some Markdown invariant wording is inverted, so JSON was used. | Model output is a proposal, never permission. Host-only grant issuance is not a model tool. |
| [NFW-005 results](../neuralFirewallV2/experiments/NFW-05_adversarial_capability_firewall/NFW_005_RUN_RESULTS.md) | 13/13 security checks and no authorization without a trusted token among 2,010 deterministic cases; no external actions. | Caller binding, expiry, revocation, one-use permissions, exact schemas, duplicate-key and nonfinite rejection. |
| [NFW-006](../neuralFirewallV2/experiments/NFW-06_adaptive_sandbox_evaluation/README.md), [NFW-007](../neuralFirewallV2/experiments/NFW-07_integrated_boundary_evaluation/README.md), [NFW-008](../neuralFirewallV2/experiments/NFW-08_live_model_broker_evaluation/README.md) | Plans/notebooks separate decisions, effects, utility and telemetry. No saved executed outputs found locally for these stages. | Measure committed writes and successful tasks separately. Do not describe planned workloads as achieved evidence. |
| [NFW-009 audit](../neuralFirewallV2/experiments/NFW-09_tool_result_injection/NFW_009_RUN_AUDIT.md) | Host JSON/native model-format mismatch confounds success rates; malformed multi-action responses occur. | Native transport adapters; reject ambiguous, partial or multi-call responses without first-JSON recovery. |
| [NFW-010 report](../neuralFirewallV2/experiments/NFW-10_protocol_corrected_injection/nfw-10-results/REPORT.md) | Qwen 3B passed format gate 18/18; 0.5B failed 0/18 and was skipped. 3B clean/benign tasks 24/24 each; injected success 15/24. Scope-only wrong-content effects 7, exact-argument effects 0. | Protocol errors remain errors. Model usability is separate from broker enforcement. Bind both destination and content. |
| [NFW-011 report](../neuralFirewallV2/experiments/NFW-11_authorization_provenance/REPORT.md), [reference broker](../neuralFirewallV2/src/policy/capability_broker.py) | Frozen NFW-010 replay: source-bound permissions yield 0 wrong-content effects versus 7 scope-only, with clean/benign success 24/24. Trust was constructed; broker used ephemeral memory and mock writes. | Import approved immutable source snapshots, bind their exact values, persist grants, and atomically commit actual notes with audit records. |

## Engineering inference

NFW-011 supplies a feasible permission source for literal copies: the host already
has the approved source record. It does **not** justify deriving permissions from
the model's proposed action or from an expected-answer label. The runtime issues
the grant before generation, keeps it outside the model context, and validates the
proposal against it afterward.

For tool effects, the new opaque grants use a host-owned SQLite lookup rather than exporting signed
claims to the model. This removes a need to provision demonstration signing keys
while preserving one-use authorization. Grant consumption, note mutation and audit
are one transaction; source provenance and destination version are bound as well.
This architecture is independent of model internals and can therefore support
multiple providers through a small proposal adapter. Provider compatibility and
task reliability still require explicit validation.

The original experiment implementation stays intact. The application is a separate
installable package with no research-stack runtime dependencies. Its scope is
documented in the [working system guide](WORKING_SYSTEM.md).

## Unknowns and claim boundary

- Archived numbers are scoped observations on their original task sets, not
  new benchmarks of this gateway or population attack-rate estimates.
- Source-bound matching equaled an oracle only because these tasks copy the
  trusted record literally. Arbitrary semantic transformations are not covered.
- A locally approved digest pins bytes; authenticating the upstream author remains
  the integrating application's responsibility.
- The host process, OS account and database remain trusted. Remote model inference
  separates the model protocol from authority but does not create an OS sandbox.
- No broad harmful-text filter, protected neural state, or universally portable
  neural monitor is established. The optional veto callback cannot grant rights.
- Durable audit records are not tamper-proof against an administrator, and restoring
  an old database requires retiring outstanding grants to avoid restoring authority.

These are the boundaries of the implemented product, not prerequisites for another
experiment before using the demonstrated workflow.
