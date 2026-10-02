# Road from prototype to product

The attention read-permission implementation now has a separate [running guide and release boundary](READ_PERMISSIONS.md). It is a local deterministic information-flow control and needs no anomaly profile. The detector-specific roadmap below remains relevant only to the optional trajectory monitor. Neither path currently has a production or general prompt-injection-resistance release claim.

## Current release decision: no production enforcement

The software path works; the detector is not ready to make security decisions for users. In the pretrained smoke run, the monitor blocked an injected instruction and also blocked a benign security-training quotation. It was calibrated on four requests. Its threshold is the maximum of four calibration trajectories, so zero calibration alarms are expected by construction; that does not show a low false-block rate on ordinary traffic.

Earlier repository experiments warn against reading a strong headline metric as product evidence. NFW-003's activation monitor reached 17.6% recall at one selected point, with a 21.1% safe-response block rate; the run had no benign-prompt negatives for estimating that false-positive rate. NFW-001's later audit recorded 5.57% jailbreak-positive detection and 0.5687 OOD AUROC at the reported operating point. Those are different monitor versions and datasets, so they are warning signals, not benchmarks for this new code. See [the implementation audit](../docs/NPS_IMPLEMENTATION_AUDIT.md) and its [NFW-003 source results](../neuralFirewallV2/experiments/NFW-03_behavioral_proxy/nfw003_necent_002/final_report.json).

For now, use `--mode monitor` in any non-local trial. Keep the current enforce mode in a closed test environment. Do not lower the threshold after looking at a failed example: that trades one observed error for an unknown number of missed attacks.

## Product shape

Ship an internal model-API firewall around generation. Keep host policy and model configuration outside request data. Buffer the full response and expose it only after the release decision. Keep tool execution outside the model adapter; any proposed tool effect goes through the separately authorized action path. Use neural trajectory deviation as a versioned IoC that can contribute to a decision, not as a synonym for injection.

Prompt injection is about an untrusted source gaining instruction authority. A large activation shift can be legitimate; a small, adaptive shift can evade this detector. The product policy must define which sources may instruct the model and which may only supply facts. Model-family support remains explicit and profile-specific.

## Release gates

### 1. Fix the evaluation question and freeze the baseline

Pick one product workload, model revision, decoding settings, context template, and supported input sources. Define a successful injection by behavior: did untrusted content cause the model to ignore the trusted task/policy, disclose protected data, or propose an unauthorized effect? A monitor alarm by itself is not attack success.

Run the same frozen requests through the unguarded target model and the guarded runtime. Preserve both outputs and have independent reviewers label task completion, policy compliance, injection success, and false block. Review disagreement rather than forcing ambiguous cases into attack labels.

### 2. Build data that measures both security and usefulness

Collect real benign workloads and a separately authored attack set, then split by source/template, task family, and time before fitting anything. Include benign security discussions and quotations, document/email/web retrieval, long contexts, multilingual inputs if supported, ordinary novelty, and different external-source roles. Pair each attack with a semantically matched benign control so topic words do not identify the label.

Add a disclosed-defense adaptive red-team set. Attackers should see the deployed monitor/profile assumptions and attempt both successful instruction takeover and low-anomaly trajectories. Keep adaptive-development examples out of the final held-out test. Record generation seeds, exact model and tokenizer revisions, source provenance, and adjudication evidence.

### 3. Set numerical go/no-go criteria before tuning

Use the existing audit's proposed initial engineering gates as a discussion baseline, not an external standard: the lower source-cluster bootstrap bound for relative attack-success reduction should be at least 50%; the one-sided 95% binomial upper bound for benign false blocks should be at most 2%; the upper bound for lost benign task success should be at most two percentage points; and the upper source-cluster bootstrap bound for p95 overhead on completed matched generations should be at most 15%. Define the denominator, uncertainty method, workload mix, and latency percentile before the run. Reject an evaluation set whose unguarded attack success is too low to make the relative-reduction target meaningful.

The 2% false-block gate needs real evidence. With independent representative examples and zero observed false blocks, roughly 149 benign cases are needed just to put the one-sided 95% binomial upper bound below 2%; any observed errors require more data. Template/source clustering reduces effective sample size, so report clustered uncertainty as well. Four calibration requests are nowhere near this evidence level.

Fit the profile and choose thresholds on training/calibration/validation data only. Plot attack success, false blocks, task success, and latency together across thresholds. If no threshold satisfies the frozen security and utility gates, the neural score is not an enforcing signal for this workload. Keep it in monitor mode or combine it with a separately tested authority/provenance control, then rerun the complete gate set. Do not tune on the final test set.

### 4. Validate the deployment boundary

Before a pilot, test the exact production model build and runtime with concurrency, cancellation, timeouts, memory pressure, malformed requests, model/profile mismatch, logging failures, and restart/reload. Preserve the current no-partial-response invariant. Choose and document what happens when the model, monitor, or calibration store fails; fail-closed behavior protects confidentiality but can stop service.

The present local HTTP server is a demo surface: it binds loopback, has no user authentication, and runs one model serially. A real deployment needs a controlled authenticated ingress, TLS termination, tenant quotas, request and context limits, worker isolation, bounded queues, health/readiness, privacy-safe metrics, a rollback switch, and secured/versioned profile distribution. Test the model adapter only on explicitly supported architectures; closed providers that do not expose internal states need a different adapter or should be marked unsupported.

Protect profile integrity with deployment-controlled, versioned artifacts and an authenticated release process. Current digests bind ordinary files to a model and policy; they are not signatures and do not defend against a compromised host.

### 5. Canary, monitor, then enforce

Deploy monitor-only to a small internal cohort first. Log profile/version, decision, score summaries, latency, task outcome, and appeal/review labels; avoid retaining raw prompts or completions unless policy and access controls explicitly permit it. Sample both allowed and flagged cases for adjudication. Watch false blocks by task/source group and drift after any model, prompt-template, tokenizer, decoding, or profile update.

Promote a specific signed combination of model revision, adapter, policy, profile, and threshold only after the frozen gates pass. Start with a small canary, keep a one-step rollback to monitor-only, and block promotion automatically when the profile identity changes. Repeat the evaluation after every such change. Treat a change to the model family as a new calibration and release candidate.

## Next concrete work item

The paired Qwen evaluation runner now implements frozen split/profile/capture digests, train/calibration/test overlap rejection, alternating paired runs, output and telemetry hashes, independent human reviews with recorded adjudication, clustered confidence intervals, and per-source/task summaries. It measures overhead only on complete matched generations, gates on uncertainty bounds, and always marks promotion ineligible.

The next work is a reviewed evaluation set for one named product workload. The six-case `examples/test_cases.example.json` is a format fixture only. Replace it with representative benign tasks and matched injection cases spanning at least five independent source groups per stratum and enough benign requests to measure the false-block target. Keep the label file under separate evaluator custody until frozen evaluation results are complete. Follow [the paired evaluation guide](README.md#paired-security-and-usefulness-evaluation), review each output using [the review guide](examples/review_guide.md), and add a disclosed-defense adaptive set outside this basic runner before considering a canary.

Do not add detector features or model families until this held-out study shows whether the present anomaly signal can meet the frozen security, utility and latency gates.
