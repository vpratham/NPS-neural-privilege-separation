# Road from prototype to product

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

Use the existing audit's proposed initial engineering gates as a discussion baseline, not an external standard: at least 50% relative reduction in independently measured attack success versus the same unguarded model; no more than 2% added benign false blocks; no more than a 2 percentage point loss in ordinary task success; and no more than 15% monitoring overhead under matched generation. Define the denominator, confidence interval, workload mix, and latency percentile before the run. Reject an evaluation set whose unguarded attack success is too low to make the relative-reduction target meaningful.

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

Build a versioned evaluation runner for the Qwen adapter that:

1. Runs each fixed example with the unguarded model and guarded firewall under the same decoding settings.
2. Captures candidate answer, gate status, activation trace reference, model/profile identity, and latency separately.
3. Accepts independent reviewed labels rather than inferring success from an alarm or keyword match.
4. Reports attack success, benign false-block rate, ordinary task success, confidence intervals, and per-source/task breakdowns.
5. Fails closed on missing labels and refuses threshold changes after the test split is frozen.

Do this before adding more detector features or supporting more architectures. Otherwise a better-looking score can hide the same false-positive problem.
