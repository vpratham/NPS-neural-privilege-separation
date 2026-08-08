# Neural Firewall — Phase 1/2 prototype

Modular package implementing the architecture in the spec doc:

```
Prompt -> Forward Pass -> Hidden States -> NeuralFirewall
-> Risk Score -> Policy Decision -> Generation Continues / Intervention
```

## Module map

| Module | Responsibility |
|---|---|
| `model_interface.py` | `ModelAdapter` ABC + `HFCausalLMAdapter` (covers Qwen2.5/Llama/Mistral/Gemma via `layer_path`). No firewall logic imports a model family by name. |
| `activation_extractor.py` | Pools hidden states at the layers a `ProbeBank` needs. **Reuse seam** — swap in your Exp015-18 extraction/caching logic here. |
| `probe.py` | `PolicyDirection` artifact format (weight/bias/threshold/metadata) + `ProbeBank` loader. Runtime-only — no training code. |
| `calibration.py` | ROC-based threshold selection (target-FPR / target-recall / Youden). Called only from `train_offline.py`. |
| `intervention.py` | `reinforce`, `suppress`, `projection_removal`, `orthogonal_projection`, `activation_clipping` — same signature, swappable via `Policy`. |
| `policy.py` | `Policy` config: which probes, voting strategy (majority/weighted/average-projection/any/all), `Mode` (detect/warn/intervene/block), intervention choice. |
| `firewall.py` | `NeuralFirewall` — `score()`, `should_block()`, `intervene()`, `decide()`. Orchestrates everything above; owns no model weights or training code. |
| `train_offline.py` | **The only place a `PolicyDirection` is fit.** Explicit CLI/build step; produces artifacts the firewall loads. **Reuse seam** — port your validated per-layer logistic-regression + calibrated-threshold training here. |
| `evaluation.py` | Precision/recall/F1/FPR/AUROC against a labeled prompt set. |
| `visualization.py` | Risk timeline, per-layer contribution, projection magnitude, intervention-strength sweep, 2D trajectory plots — each returns a `Figure`. |
| `streaming.py` | Phase 4 + Phase 8: token-by-token decode loop (`StreamingFirewall.generate`) with per-token risk scoring via `firewall.score_pooled()`. Honors Mode at every step — DETECT/WARN never stop, BLOCK halts before emitting the offending token, INTERVENE keeps intervention hooks live for the whole loop. |
| `cache.py` | Generic manifest-based resumability. **Reuse seam** — swap for your existing Drive-caching/manifest code if the interface differs. |

## What's implemented vs. scaffolded

**Implemented, runnable, and covered by `test_smoke.py`:**
- Full `NeuralFirewall.score()` / `should_block()` / `decide()` for all four modes (detect/warn/intervene/block), including majority/weighted/any/all/average-projection voting across layers and max/mean combination across multiple policies.
- `score_pooled()` — the scoring core factored out so both prompt-level `score()` and per-token streaming share one code path (no drift between the two).
- `PolicyDirection` save/load round-trip with shape validation against the active model's `hidden_size()`, and precision-consistent calibration (threshold and runtime score are both computed at the float32 precision the artifact is persisted at — see below).
- `train_offline.run_training_job()` with per-layer manifest resumability.
- All five intervention functions, unit-testable independently of a model.
- Mode 3 (intervene) hook install/teardown, exercised both as a single-shot call and as a persistent-hooks streaming loop.
- **Phase 4 + Phase 8** (`streaming.py`): true token-by-token generation with per-step firewall checks, early BLOCK termination, and continuous intervention across the whole decode loop.
- `visualization.plot_streaming_result()` builds a risk-timeline figure straight from a `StreamingResult`.

**A real bug the smoke test caught and fixed:** `PolicyDirection` artifacts are persisted as float32, but the original calibration ran in float64 — meaning a borderline score could land on the wrong side of its own threshold at runtime (confirmed on the synthetic test: one layer's vote flipped). `train_offline.fit_policy_direction` now calibrates at the same float32 precision the runtime firewall scores with, so the threshold is exact for the artifact that ships.

**Scaffolded, needs your notebook's logic ported in (marked in-file):**
- `ActivationExtractor.extract()` currently does a single forward pass with hooks; batched/dataset-level extraction with fingerprint caching (Exp017/18) isn't ported yet — this needs your actual notebook code, not something a synthetic test can validate.
- `train_offline.fit_policy_direction()` uses plain sklearn `LogisticRegression` — should be swapped for whatever regularization/CV choices your notebooks settled on.
- Phase 6 cross-*layer* voting and cross-*policy* combination are both implemented; a genuine multi-probe ensemble (voting across independently-trained probes for the *same* policy, e.g. different random seeds) isn't — `ProbeBank` currently assumes one direction per (policy, layer).
- Phase 10 (unsafe intent / toxicity / privacy / copyright / prompt injection / jailbreak as independent detectors) — `Policy.policy_names` already supports multiple named policies scored and combined independently; adding a new one is a `train_offline.py` artifact, no code change.

## Smoke test

`test_smoke.py` runs the full pipeline — train_offline → ProbeBank → NeuralFirewall (all 4 modes) → StreamingFirewall → visualization — against a tiny mock `ModelAdapter` (no model download required) with a synthetic "trigger token" as the thing being detected. It's plumbing validation, not a safety benchmark: swap `MockAdapter` for `build_qwen_adapter(...)` and real `PolicyDirection` artifacts once your extraction pipeline and labeled data are ported in.

```
python3 test_smoke.py
```

## Example wiring

```python
from neural_firewall import build_qwen_adapter, ProbeBank, Policy, Mode, VotingStrategy, NeuralFirewall

adapter = build_qwen_adapter("Qwen/Qwen2.5-3B-Instruct")
probe_bank = ProbeBank.load("artifacts/", expected_d_model=adapter.hidden_size())

policy = Policy(
    policy_names=["unsafe_intent"],
    voting=VotingStrategy.MAJORITY,
    mode=Mode.BLOCK,
)

firewall = NeuralFirewall(adapter, probe_bank, policy)
decision = firewall.decide("some prompt")
print(decision.action, decision.assessment.risk_score)
```

To swap in intervention mode:

```python
policy = Policy(
    policy_names=["unsafe_intent"],
    mode=Mode.INTERVENE,
    intervention_method="projection_removal",
    intervention_strength=1.0,
)
firewall.fit(policy=policy)
```

## Streaming (Phase 4 / Phase 8)

```python
from neural_firewall import StreamingFirewall

streamer = StreamingFirewall(firewall)  # firewall.policy.mode == Mode.BLOCK or Mode.INTERVENE, etc.
result = streamer.generate("some prompt", max_new_tokens=128)

print(result.output_text)
print(result.stopped_early, result.stop_reason)

from neural_firewall import visualization
fig = visualization.plot_streaming_result(result, threshold=0.5)
fig.savefig("risk_timeline.png")
```
