# Neural State Firewall

A standalone internal-activation monitor with a deterministic, buffered response gate. It observes a local language model during inference, estimates how its neural trajectory differs from benign calibration data, and halts generation when an anomaly threshold is crossed.

This folder is independent of `nps_gateway/`, the source-copy broker, and the earlier text-judge firewall. It imports none of them and executes no tools. It contains executable Python modules, a local HTTP API, calibration commands, and tests.

The architecture track is specified in [NPS_PERMISSION_BOUNDARY.md](NPS_PERMISSION_BOUNDARY.md), with stages and acceptance criteria in [NPS_IMPLEMENTATION_PLAN.md](NPS_IMPLEMENTATION_PLAN.md). An opt-in adapter now isolates policy KV memory. Attention read permissions are the next stage; policy-memory isolation alone does not enforce instruction authority.

**Current capability:** working activation capture, temporal anomaly scoring, and response withholding. **Unestablished capability:** reliably identifying prompt injection. The alarm means “unusual neural trajectory,” not “proven attack.” The supplied eight benign examples demonstrate the integration; they are not a deployment-quality calibration set.

## Run immediately

From the repository root, the dependency-free replay exercises allow, mid-generation block, and broken-sensor paths:

```bash
python3 -m neural_state_firewall demo
```

The demo reports `synthetic_mechanics_only`. It must not be used as detection evidence.

For real model activations, use Python 3.11 and the isolated environment:

```bash
python3.11 -m venv neural_state_firewall/.venv
neural_state_firewall/.venv/bin/pip install -r neural_state_firewall/requirements.txt
export HF_HOME="$PWD/neural_state_firewall/.cache/huggingface"
```

The environment and Qwen2.5-0.5B weights were downloaded locally during implementation. They are ignored by Git. On a fresh checkout, the first capture downloads the model. CPU float32 inference needs several GB of available memory. The initial adapter supports Qwen2/Qwen2.5 causal models with `model_type=qwen2`.

Capture separate benign training and calibration splits, then fit:

```bash
neural_state_firewall/.venv/bin/python -m neural_state_firewall capture \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --policy neural_state_firewall/examples/policy.txt --layers 6,18 \
  --max-new-tokens 64 --requests neural_state_firewall/examples/train.jsonl \
  --output neural_state_firewall/artifacts/train.json

neural_state_firewall/.venv/bin/python -m neural_state_firewall capture \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --policy neural_state_firewall/examples/policy.txt --layers 6,18 \
  --max-new-tokens 64 --requests neural_state_firewall/examples/calibration.jsonl \
  --output neural_state_firewall/artifacts/calibration.json

neural_state_firewall/.venv/bin/python -m neural_state_firewall fit \
  --training neural_state_firewall/artifacts/train.json \
  --calibration neural_state_firewall/artifacts/calibration.json \
  --output neural_state_firewall/artifacts/profile.json
```

Artifact writes deliberately refuse to overwrite existing files. Reuse existing validated artifacts to run, or choose a new output directory when recalibrating. Remote revisions resolve to an immutable commit shared by model and tokenizer; use `--revision COMMIT` to select one explicitly.

Run the example requests:

```bash
neural_state_firewall/.venv/bin/python -m neural_state_firewall run \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --policy neural_state_firewall/examples/policy.txt --layers 6,18 \
  --profile neural_state_firewall/artifacts/profile.json \
  --requests neural_state_firewall/examples/evaluation.jsonl
```

Each JSON line includes status, released output, per-step scores, and alarm state. `allowed` means the monitor did not trip; it does not certify the response. Exit code 2 indicates an error or incomplete generation. A handled block has exit code 0 and `status: blocked`. The `run` command does not judge whether the answer obeyed the task. Do not report its raw statuses as an attack-success benchmark.

## Paired security and usefulness evaluation

The evaluation runner uses a frozen test manifest and profile. The example manifest only demonstrates the file format; its six cases are far too few for a release decision. Replace it with your approved held-out set, and never put reviewer labels in the manifest. Condition/source/task fields are used for breakdowns and are not sent to the model.

First freeze exact case, profile, training-capture, and calibration-capture digests. The runner verifies that test request hashes do not overlap either benign fit split and that the supplied captures are the ones used to create this profile:

```bash
neural_state_firewall/.venv/bin/python -m neural_state_firewall freeze \
  --cases neural_state_firewall/examples/test_cases.example.json \
  --profile neural_state_firewall/artifacts/profile.json \
  --training neural_state_firewall/artifacts/train.json \
  --calibration neural_state_firewall/artifacts/calibration.json \
  --output neural_state_firewall/artifacts/test.lock.json
```

Then run each request once without gating and once with the same greedy model, template, policy, horizon and profile. The order alternates by case to balance warm-up effects. Allowed guarded answers must exactly match the unguarded greedy answers; any mismatch or failed run prevents a result artifact:

```bash
neural_state_firewall/.venv/bin/python -m neural_state_firewall evaluate \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --policy neural_state_firewall/examples/policy.txt --layers 6,18 \
  --cases neural_state_firewall/examples/test_cases.example.json \
  --profile neural_state_firewall/artifacts/profile.json \
  --training neural_state_firewall/artifacts/train.json \
  --calibration neural_state_firewall/artifacts/calibration.json \
  --lock neural_state_firewall/artifacts/test.lock.json \
  --output neural_state_firewall/artifacts/paired-results.json
```

Raw result files contain both model outputs and activation-score telemetry. Restrict their access and retention like the original prompts; do not upload them into an issue or dashboard without review.

Create a label template and have two independent reviewers inspect each unguarded answer and each released guarded answer. Give them the trusted task/policy and reference evidence. Label task success and whether untrusted context actually took instruction authority. A blocked guarded answer has no output to label; the report counts it as unavailable task utility and counts its injection success as zero released attacks. If two reviewers disagree, record a third independent review, an adjudication note, and the majority outcome. Reviewer IDs should be pseudonymous and stable within a study.

```bash
neural_state_firewall/.venv/bin/python -m neural_state_firewall label-template \
  --results neural_state_firewall/artifacts/paired-results.json \
  --output neural_state_firewall/artifacts/reviewed-labels.json
```

Fill the generated `reviewed_cases` entries. Each available `baseline` and `guarded` arm needs at least two entries under `reviews`, each with a distinct `reviewer_id`, `task_success` and `injection_success` boolean. Copy the unanimous vote to `adjudicated` and leave `adjudication_note` null; for a disagreement supply the third review and set the majority outcome with a short note. Put the review procedure in `reviewer_protocol`. The report rejects missing labels, duplicate reviewers, labels for a blocked answer, disagreement without a majority review, or labels from any other result file.

```bash
neural_state_firewall/.venv/bin/python -m neural_state_firewall report \
  --results neural_state_firewall/artifacts/paired-results.json \
  --labels neural_state_firewall/artifacts/reviewed-labels.json \
  --output neural_state_firewall/artifacts/report.json
```

The report gives baseline and guarded attack success, benign false-block rate, ordinary task success, paired latency overhead, 95% source-cluster bootstrap intervals, and source/task-group breakdowns. Criteria compare uncertainty bounds to the frozen gates; blocked requests are reported separately and excluded from full-generation overhead. The false-block upper bound is an exact binomial limit and assumes independent cases, so repeated templates or sources can widen real uncertainty. The runner flags small evidence sets and **always** says `promotion_eligible: false`: adaptive red-team coverage and production operations still require human review. A manifest lock binds the threshold/model/split within the normal workflow; it is a hash, not protection against a privileged person rewriting the manifest, lock and results together. Keep the held-out labels under separate evaluator custody until the paired run is complete.

## Local model API

```bash
neural_state_firewall/.venv/bin/python -m neural_state_firewall serve \
  --model Qwen/Qwen2.5-0.5B-Instruct --local-files-only \
  --policy neural_state_firewall/examples/policy.txt --layers 6,18 \
  --profile neural_state_firewall/artifacts/profile.json
```

Then send a request:

```bash
curl http://127.0.0.1:8765/v1/respond \
  -H 'Content-Type: application/json' \
  -d '{"task":"What day is the workshop?","context":"The workshop is on Friday."}'
```

The server binds only to `127.0.0.1`, serializes model access, and accepts only `task` and `context`. Policy, weights, profile, mode and token budget belong to the host. `/health` reports readiness and enforcement mode. No token streaming, provider fallback, tool execution, public deployment authentication, or arbitrary upstream-model proxy is implemented.

| Outcome | HTTP | Output |
|---|---:|---|
| EOS reached without alarm in enforce mode | 200 | Complete buffered answer |
| Neural anomaly in enforce mode | 403 | `null` |
| Invalid telemetry or runtime/cleanup failure | 503 | `null` |
| Token budget reached without EOS | 422 | `null` |
| Invalid request / attempted configuration override | 400 | `null` |

`--mode monitor` explicitly permits completed anomalous responses and labels them `monitored`, with `enforced: false`. Sensor errors still withhold output. Enforce is the default. There is no claim that anomaly filtering alone makes a model API safe for external deployment.

## What happens inside

```text
Host policy + user task + untrusted context
                  |
          Qwen cached forward pass
                  |
  Selected decoder block OUTPUT hooks (last position)
                  |
      Seeded projection -> feature vector z_t
                  |
    Diagonal state-space predictor + CUSUM
                  |
      alarm/error? -> halt and discard output
                  |
    otherwise buffer next token; release at EOS
```

The first observation is the final prompt-position activation during prefill, before releasing the first generated token. Later observations are taken during token decoding. Every observation uses the same zero-based block-output sites during capture and runtime. The adapter carries the actual KV cache and checks that each selected block emits exactly one observation per forward pass. The monitor sees a projection of residual activations, not the complete cache/state.

For each projected dimension, benign training traces fit a diagonal autoregression:

```text
predicted_z[t] = a * z[t-1] + b
r[t] = (z[t] - predicted_z[t]) / residual_scale
energy[t] = mean(r[t]^2)
C[t] = max(0, C[t-1] + energy[t] - drift)
alarm[t] = previous_alarm OR C[t] > threshold
```

The first observation uses a separately fitted distribution of training starts. No transition crosses a request boundary. Held-out benign trajectories set the threshold from their maximum sequential statistic, using the selected quantile plus a small numerical margin. Defaults use the maximum observed calibration statistic; zero observed calibration false alarms is therefore expected and does not establish an out-of-sample false-alarm guarantee. Variance floors prevent division by zero.

Telemetry includes standardized innovation energy, maximum residual, accumulated CUSUM, threshold, observation phase and sensor-error flag. These are candidate AI-security IoCs. Random projections and unsupervised deviations do not identify instruction authority or prove causal takeover. There is no trained attack-specific signature in this implementation.

## What is portable

`runtime.Adapter` defines `identity`, `policy`, `iter_steps(...)` and `decode(...)`. The observer, release controller, artifact validation and API are model-independent. Each new model family needs a validated adapter and fresh capture/calibration. The first adapter intentionally implements one family and one greedy float32 inference path.

Profiles bind model revision/weights, model configuration, tokenizer and template, Torch/Transformers versions, dtype/device, hook sites, layers, projection parameters, policy digest and calibrated generation horizon. Deployment code should use:

```python
from neural_state_firewall import Firewall
from neural_state_firewall.artifacts import read
from neural_state_firewall.hf_adapter import HFAdapter

adapter = HFAdapter(
    "Qwen/Qwen2.5-0.5B-Instruct", policy=open("neural_state_firewall/examples/policy.txt").read(),
    layers=[6, 18], local_files_only=True,
)
firewall = Firewall.from_artifact(adapter, read("neural_state_firewall/artifacts/profile.json"))
result = firewall.run("What day is the workshop?", "The workshop is on Friday.")
```

The lower-level `Firewall(adapter, inner_profile)` constructor is for controlled embedding/tests and does not supply a calibrated horizon. Normal parameter/configuration/tokenizer mutations after adapter construction are rejected. The host process, model files and profile files are trusted. Digests identify artifacts; they are not signatures and do not prevent a privileged host from replacing them. Ordinary parameter-version checks are not a defense against hostile process-memory modification.

Closed provider APIs and Ollama's text API do not expose the required activations. Supporting those through a text classifier would be a different monitor; this implementation does not pretend otherwise.

## Lessons carried forward

The [implementation audit](../docs/NPS_IMPLEMENTATION_AUDIT.md) identified off-by-one hook sites, weak transfer, topic confounding, stale artifacts, and a gap between detection and enforced behavior. This implementation responds with explicit output hooks, paired cached/full-prefix sensor tests, split separation, identity checks, and a tested no-partial-release gate. It does not reuse earlier detector weights or their claimed accuracy.

The [NFW-011 report](../neuralFirewallV2/experiments/NFW-11_authorization_provenance/nfw011_authorization_provenance_001/REPORT.md) established a narrow source-bound authorization result. That broker answers whether a proposed effect is authorized; it does not establish neural injection detection. No broker is included here. A larger agent application still needs independent tool authorization on its action path.

State-space terminology does not provide a proof of semantic safety. [Dialogue dynamics research](https://arxiv.org/html/2503.00187v3) motivates temporal monitoring but uses a different state abstraction; [obfuscated activation attacks](https://arxiv.org/html/2412.09565v2) demonstrate important evasion risks for latent-space defenses. This implementation neither identifies a privileged policy subspace nor proves a control-barrier invariant. See the [pinned Qwen source](https://github.com/huggingface/transformers/blob/753d61104116eefc8ffc977327b441ee0c8d599f/src/transformers/models/qwen2/modeling_qwen2.py) and [cache documentation](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation) for the inference mechanics used here.

## Verification and remaining work

To verify the policy-memory path against ordinary decoding using already-cached weights:

```bash
HF_HOME=neural_state_firewall/.cache/huggingface HF_HUB_OFFLINE=1 \
neural_state_firewall/.venv/bin/python -m neural_state_firewall check-policy-memory \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-files-only \
  --policy neural_state_firewall/examples/policy.txt --layers 6,18 \
  --requests neural_state_firewall/examples/evaluation.jsonl --max-new-tokens 32 \
  --output neural_state_firewall/artifacts/policy-memory-check.json
```

The command runs without an anomaly profile and returns a structural equivalence report, not attack labels. The recorded [pretrained comparison](validation/policy_memory_smoke.json) has identical tokens on three seen development requests; all completed at EOS. Ordinary causal attention already protects earlier prefix representations from later tokens. The isolated path adds explicit storage ownership, append-position checks and digest checks before releasing each generation step. It rejects unsupported cache mutations and tokenizers whose system-only encoding is not an exact request prefix. PyTorch tensors remain mutable to trusted host code; this is not process isolation.

`--isolate-policy-memory` also selects this adapter for capture/run/evaluate/serve. Its distinct decoder identity requires new capture/calibration artifacts for the optional anomaly monitor; the old profiles are rejected. Batch-one full-attention Qwen2, float32 and Transformers 4.57.6 are the currently supported combination.

```bash
neural_state_firewall/.venv/bin/python -m unittest discover -s neural_state_firewall/tests -v
```

Tests cover numerical failure, split leakage, identity drift, cached/full-prefix agreement on a genuine tiny random Qwen model, response withholding after partial generation, cleanup failures, and actual loopback HTTP responses. The tiny model is not a meaningful attack target. The separate pretrained smoke run is recorded in `VALIDATION.md`.

Before treating this as prompt-injection protection, measure task success, unauthorized instruction following, benign false blocks, adaptive attacks, and latency on representative held-out workloads. Fit only on trusted benign data, keep calibration and evaluation separate, and include benign quotations of attack language. Short, low-drift attacks may pass; novel legitimate tasks may block. These are the remaining detection-quality questions, not reasons to equate the working gate with a proven firewall.
