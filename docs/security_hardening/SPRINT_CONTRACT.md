# October 4–10 security and evaluation contract

**Specification v1, October 4, 2026.** Baseline `3ebbb4c6ad22e26a614b3591cf11864f97ba6f3a`. This freezes the intended scope and gates, not the final code/profile/test data. The final runtime freeze remains Day 3. [Plan](ONE_WEEK_PLAN.md); [numeric specification](release_gates_v1.json); [execution status](DAY1_STATUS.md).

The target is a request/response firewall with internal source permissions and a separately authorized tool-action path. It is not a prompt-text classifier. A model answer and a neural anomaly score are untrusted signals, never grants of authority.

## Scope and owners

- **Host administrator:** model/tokenizer revision, runtime/backend, policy, principal, source IDs and digests, grants, generation/resource limits, monitor profile, supported operations and permitted recipients.
- **Authenticated caller:** the task, within the host's permissions. A bearer token currently identifies one realm; it is not a multi-user identity provider.
- **Attacker:** readable or denied evidence content, including forged instructions, role delimiters, grants and tool proposals. Compromise of the host, model implementation or operating system is outside the claim.
- **Model:** may emit prose or a proposal, but cannot modify host policy or execute actions. The broker authorizes only its implemented exact source-copy operation. New tool types remain unsupported until authorized at their effect point.
- **Evaluator:** outcome labels and held-back data custody. One genuine human reviewer is permitted and disclosed. No invented reviewer IDs or claims of inter-rater agreement.

## Threat and verification matrix

| Threat | Required boundary and check | Current evidence / remaining work |
|---|---|---|
| Denied source influences output | Mask denied-to-public attention at every layer/cache step; preserve positions/shapes; swap denied contents and compare every-step logits/tokens | Existing CPU tests; CUDA qualification is a separate Colab run |
| Readable evidence redirects prose or code | Evaluate actual instruction takeover and task utility, including four known BIPIA failures and fresh adaptive attacks | Unresolved; structural masking alone does not settle this |
| Forged role, principal, grant or source | Request schema rejects host-field overrides; host pins source identity and provenance | Existing API/workload tests |
| Unauthorized tool or recipient | Model response is a proposal; broker checks its supported operation, exact content, subject and destination before effect | Existing gateway tests; no arbitrary tools or generated-code execution |
| State crosses requests | Fresh request adapter/cache; immutable shared weights; tampering and alternating-request tests | Existing structural/workload tests; repeat in deployment configuration |
| Partial text escapes on failure | Buffer until EOS; cap/error/deadline/cleanup failure releases no model text | Existing runtime and killable-worker tests |
| Resource or logging failure | Bounded admission/body/context/output, parent deadline, explicit recovery and privacy-safe audit | Existing controls; deployment soak, memory-pressure/log-sink tests pending |
| Detector/profile mismatch | Profile binds exact adapter identity, policy and horizon; scores cannot weaken permissions | Existing observer binding; combined permission/shadow integration pending |
| Evaluation leakage | Group/digest separation plus source exclusions; actual custody and payload-hash verification before freeze | Metadata validator added; fresh corpus and custody not yet available |

Relevant implementation: `neural_state_firewall/runtime.py`, `read_permissions.py`, `pilot.py`, `document_workload.py`, `observer.py`; action authority: `nps_gateway/store.py` and `runtime.py`. See [known failures](README.md) before interpreting a passing structural test.

## One bounded candidate: input/output and trust contract

1. A **host-owned task envelope** binds request ID, realm, policy digest, authorized source references and permitted response/operation class. Natural-language task content cannot add operations or recipients. No public API for supplying grants is introduced.
2. An **evidence reader** can propose source/span references for the original task. Each reference contains source ID, source digest and integer start/end offsets. The host checks access, bounds and identity and recovers the text from its pinned source; the reader cannot manufacture trusted facts, paths or origin labels. Missing/invalid references are failures, not silent full-context fallbacks.
3. **Answer generation** may consume selected readable spans to answer the task. Those spans remain untrusted: valid provenance and JSON do not make their instructions safe. Free-form output remains subject to behavioral evaluation. This candidate must retain useful free-form responses; replacing them with canned answers would change the scope and cannot count as success.
4. **Action proposals**, if present in a supported integration, go through the existing host-issued capability and broker. The reader and final generator never execute tools or turn evidence text into host configuration. No new general tool-execution layer is planned for this week.
5. **Release** retains current permission/runtime checks and buffering. Shadow observer telemetry cannot change tokens or grants. Detector enforcement is a separate decision requiring its own gate.

This is a candidate to implement and measure, not a completed or proven defense. It may still pass an injected instruction through a valid span; that possibility is explicitly in the red-team matrix. Stop candidate feature search by Day 3 if the known failures or utility controls fail. The previous task-prefix-sealing candidate remains rejected.

## Model and compute binding

Baseline model: `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`; Torch 2.6.0, Transformers 4.57.6, float32, greedy decoding, eager/SDPA under the current adapter constraints. CPU historical results remain CPU results. CUDA requires fresh checks and a distinct profile; do not reuse the CPU profile or imply cross-device bit equality.

**User decision:** model evaluation runs on Google Colab, not the Mac. A standard T4 is the initial target. No plan/credit purchase is authorized. The Mac hosts editing, CLI control and lightweight tests only. The remote preflight refuses local execution and CPU fallback, uses a pinned source archive, saves logs/results and is bounded by subprocess deadlines. Actual GPU availability is a prerequisite, not an assumed resource.

The current pilot constructor fixes its adapter to CPU. CUDA adapter preflight does not certify a GPU-backed authenticated API; an explicit host-owned device binding and its API tests are required before that deployment claim. Likewise `Firewall(mode="permissions")` currently forbids profiles, so shadow monitoring needs a tested integration change rather than a CLI flag flip.

## Dataset and outcome contract

- Targets: 200 benign fit, 100 benign calibration, 200 final benign cases, 100 final attacks with clean controls, and 10 separately held adaptive goals with at most five candidate queries per goal.
- Existing BIPIA revision, fixtures, historical XSTest, NeurAlchemy snapshot and their related/near-duplicate material remain development-only. Preserve those failures as regressions; never count them as unseen efficacy evidence.
- A provenance sidecar records source/revision/license, exposure, source/template/pair grouping and normalized request/document digests. `split_manifest.py` rejects declared overlap, duplicate requests and excluded/unreviewed final-test sources. It only validates metadata: payload binding, actual novelty, permissions and evaluator custody require separate evidence.
- Final payloads and labels stay unopened until the code/model/policy/profile/analysis freeze. Do not acquire a nominal holdout and then tune on it. Public benchmark data cannot establish that a pretrained model has never seen it.
- One reviewer labels task success, instruction takeover, protected-value disclosure, ambiguity and availability from blinded outputs. Model-assisted analysis is identified separately. The old evaluator's multi-review schema and output-equivalence constraint must not silently govern the new candidate study.
- Intentional blocks, caps, errors and timeouts remain separate. A timeout is not semantic rejection; a blocked benign task loses utility. Record every attempt and internal model call. No retries are silently omitted.

No final-test corpus is currently certified. `data_manifest_v1.json` deliberately records this absence. It does not fabricate 600 representative requests to meet a schedule.

## Gate interpretation

The numerical gates are frozen in `release_gates_v1.json`. A gate can pass, fail or have insufficient evidence. A structural pass never substitutes for semantic efficacy, utility or operations. Known unauthorized release blocks the corresponding release claim even if average metrics improve. Changing a gate after viewing final outcomes requires a disclosed protocol amendment and makes that run exploratory.

Until all applicable gates pass, `production_approved`, `semantic_security_approved` and `monitor_enforcement_approved` remain false. Missing author information or a journal receipt cannot be filled by inference. These are release decisions, not detector scores.
