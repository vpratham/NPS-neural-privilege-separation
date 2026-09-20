# Running NFW-003 in Colab

## First run: automated proxy benchmark

1. Open/upload `NFW_003_Behavioral_Proxy_Firewall.ipynb` in a **new Colab session**. Do not reuse the v2 kernel.
2. Select **Runtime → Change runtime type → GPU**. The default model is Qwen2.5-3B-Instruct, loaded directly in fp16 onto one GPU. A 16 GB GPU is the intended starting point, not a guarantee of availability or sufficient memory for modified settings.
3. Accept access conditions on the [Necent dataset page](https://huggingface.co/datasets/Necent/llm-jailbreak-prompt-injection-dataset).
4. Add an authorized Hugging Face **read** token to Colab Secrets as `HF_TOKEN`, with notebook access enabled. Alternatively, use the hidden prompt; never paste credentials into a source cell.
5. Run the installation cell first. It pins the same Transformers/datasets stack as v2 and leaves Colab's PyTorch/CUDA installation alone. If Colab asks for a runtime restart, restart before importing packages and continue from the top.
6. In the configuration cell, leave:
   ```python
   RUN_ID = 'nfw003_necent_001'
   MODE = 'run'
   RUN_GENERATION_AUDIT = False
   RUN_STRESS_TESTS = False
   RUN_CONTINUATION = True
   ```
7. Run all cells in order and authorize the Drive mount.

**No Python script, shell command, repository clone, CSV upload, or v2 result import is required.** The preparation, exploration, extraction, fitting and report code is inside the notebook.

Default output:

```text
MyDrive/NFW-003/nfw003_necent_001/
```

Start with `REPORT.md`, then `final_report.json`, `proxy_evaluation.json`, and `data_exploration.json`. Stage JSON files usually wrap their content in `payload` with a binding and checksum. These envelopes are intentional.

## What the default run does

- Scans the **entire pinned dataset split**, not just its beginning. It reports missing response labels and maintains per-source counts.
- Retains at most 64 hash-selected sources × 2 response classes × 40 examples: at most 5,120 candidates before exclusions. It never loads the complete dataset into a DataFrame.
- Excludes missing stored responses/response labels, overlength records, and sampled cross-source/conflicting-label duplicate prompts.
- Requires at least eight sources and at least ten examples of each response class in every split. Insufficient coverage stops with an explanation; it does not fall back to a leaky random-row split.
- Locks source-disjoint train/development/calibration/final partitions.
- Extracts prompt features and, by default, separate stored-response features. Feature checkpoints are reused across fitting candidates and reconnects.
- Compares four prompt detectors. Fits only on training, selects layers/hyperparameters on development, selects thresholds on calibration, and evaluates final once.
- Fits/evaluates a separate historical-response probe for the later buffered-continuation prototype.
- Saves an automatic proxy report, whether or not any generated-output review has been done.

The full dataset scan can dominate elapsed time and network usage. Resume can reread the current remote shard even when application-level progress is saved. Runtime and download sizes depend on source shards, Colab hardware and access limits; there is no fixed completion-time promise.

## Reading the metrics correctly

- **Harmful-response recall:** fraction of historically harmful responses whose associated prompts the gate blocks.
- **Safe-response block rate:** fraction of historically non-harmful responses whose prompts are blocked. This includes safe refusals to harmful requests.
- **Benign-prompt false-block rate:** blocking among examples with both prompt labels explicitly zero. Unknown prompt labels do not enter this denominator.
- **Counterfactual released harm per request:** historically harmful records that would remain unblocked, divided by all sampled requests.
- **Released harm fraction:** historically harmful records that remain unblocked, divided by released requests. An all-block gate has no released denominator, so the value is null.
- **Task success:** not identifiable from stored harm labels; stays unknown until generated audit labels exist.

The sample is balanced within retained source/response classes where available. Its harm prevalence is **not deployment prevalence**. Existing response labels do not tell us what Qwen would produce today. Annotation provenance may be incomplete. Source holdout does not establish semantic-family independence.

The calibration objective attempts 80% harmful-response recall under a 10% safe-response block cap. If impossible, the notebook chooses maximum recall within the cap and records that the recall target was missed. It does not increase the cap using final outcomes. Curves use thresholds selected beforehand on calibration.

## Reconnect after a Colab disconnect

1. End the previous runtime if it might still be writing. Never run two sessions on the same folder.
2. Open the **same notebook**, choose a GPU runtime, and keep `RUN_ID` and `CONFIG` unchanged.
3. Run from the top, remounting Drive if needed.

Completed work is loaded and validated:

| Interrupted stage | Recovery |
|---|---|
| Initial scan | Restores sampler, counts and stream position from the last 10,000-row checkpoint; may reread the current source shard |
| Tokenization/splits | Reuses a complete dataset stage; an incomplete bounded tokenization pass restarts |
| Activations | Skips individually completed example/layer-vector records |
| Classifier fitting | Reuses complete detector fits; an interrupted CPU fit restarts from saved features |
| Generation | Skips completed baseline examples; retries the in-flight example |
| Continuation/stress | Reuses completed per-example decisions |
| Review | Preserves the existing CSV; validates its content bindings and imports only explicit labels |

The model may need downloading/loading again after VM deletion; model cache is not a results checkpoint. A failed inference has a diagnostic entry under `failures/` and is not treated as successfully completed. Do not delete valid checkpoints to recover from an ordinary disconnect.

### If resume refuses

Read the mismatch name. Changes in configuration, dataset revision, model revision, tokenizer, implementation, Python/package/CUDA environment intentionally refuse inference reuse. GPU changes are logged without blocking resume; mixed-GPU runs are not guaranteed bitwise reproducible. Restore the original environment, or start a new RUN_ID. Report mode can read a complete proxy run without a GPU, but does not authorize continuing inference under another environment.

If a committed file is missing/corrupt, recover a backup or start a new run. The notebook does not pretend a checksum mismatch is harmless. Google Drive's mount can fail on quotas, too many files or synchronization; rename/fsync cannot provide an absolute remote-durability guarantee.

## Optional generated-output audit

Once the proxy run finishes, set:

```python
MODE = 'run'
RUN_GENERATION_AUDIT = True
```

Keep the same RUN_ID and CONFIG; rerun from the top. The optional flag is allowed to change because the audit design was already frozen in CONFIG. It selects up to 48 final examples by a deterministic hash, not by failure severity or historical labels.

Each baseline is generated once; an allowed prompt-gate condition reuses that exact output. If continuation is enabled, the additional condition checks token prefixes with the separate response-state probe and releases either the same full baseline text or nothing. Prefixes are not streamed to a user before checking.

This replay is a prototype of a **buffered** release gate, not a production streaming decoder. A probe trained on complete responses is applied to partial responses out of distribution; multiple checks can increase false positives. Treat these outcomes as unvalidated until audited.

The run exports `blinded_review.csv`:

- Keep `review_id`, `request`, and `response` unchanged, including the `TEXT: ` prefix.
- Fill `harmful_assistance` and `task_success` with `0` or `1` using the notebook rubric.
- Record `reviewer` honestly. Prefer a reviewer who has not inspected conditions or historical labels.
- Return the CSV to the same folder. The same request and same response share a review ID across conditions.
- The main proxy report remains usable if the CSV is incomplete; generated metrics remain `awaiting_independent_labels`.

For model-free import in a new CPU session:

```python
MODE = 'report'
RUN_GENERATION_AUDIT = False
RUN_STRESS_TESTS = False
```

Keep RUN_ID and all CONFIG values unchanged, including `RUN_CONTINUATION`; run all cells. Existing completed audit responses are loaded automatically. When labels are complete, the report becomes `complete_provisional` for that audit. This means complete annotations, **not proven reviewer accuracy**. Once an evaluation is committed, label edits cannot silently replace it.

## Optional stress tests and cross-model replication

Enable `RUN_STRESS_TESTS=True` in GPU run mode to run the fixed formatting-sensitivity tests. Decision flips are not independently validated harmful-output bypasses. The default capability-gate tests run without invoking any real tool.

For another compatible decoder model, use a new RUN_ID, immutable model revision and valid layer indices. Keep sampling/splitting/calibration settings identical. After the second run, set `REFERENCE_RUN_ID` to the first run. The comparator validates artifact hashes, identical dataset records/labels and partitions, and matching model-independent configuration. It refuses a mismatched cohort, including tokenizer-dependent length-exclusion changes.

This is **separately trained cross-model replication**, not zero-shot transfer of probe weights. Adaptive jailbreak optimization and genuine neural privilege isolation are not implemented or claimed; the notebook specifies the additional preregistered study needed for them.

## What to share back

For initial interpretation, share `REPORT.md`, `final_report.json`, `data_exploration.json`, and `proxy_evaluation.json`. The latter includes per-source scores and uncertainty but no raw prompt/response text. Avoid publicly uploading raw datasets/checkpoints/review sheets before checking source licenses and sensitive content.

For a RAM or runtime failure, share the traceback, the last completed stage, GPU name and package versions from the manifest—never the HF token. The design avoids full-corpus RAM loading, but modified sequence lengths, extra models, or runtime variability can still cause OOM.
