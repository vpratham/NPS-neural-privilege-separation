# NFW-003 — Historical-response proxy and buffered firewall laboratory

**Notebook:** [NFW_003_Behavioral_Proxy_Firewall.ipynb](NFW_003_Behavioral_Proxy_Firewall.ipynb)  
**Detailed instructions:** [COLAB_GUIDE.md](COLAB_GUIDE.md)

A standalone experiment that asks whether prompt-only activations add predictive value beyond a text classifier for **existing stored-response harm labels**, on held-out sources. No repository clone, command-line preparation, or external Python script is required in Colab.

This experiment follows NFW-002's modest behavioral improvement and low recall. It deliberately does **not** reuse AI-assisted v2 labels as independent ground truth. It replaces the previously planned NFW-03 policy-isolation slot with a prerequisite evaluation experiment; policy isolation remains future work.

## What is implemented

| Stage | Implementation | Scientific boundary |
|---|---|---|
| Data | Pinned Necent scan, resumable stream/sampler state, missingness audit, bounded deterministic sampling, duplicate quarantine, source-disjoint splits | Labeled subset is selected and potentially biased; no semantic-dedup guarantee |
| Baselines | Fixed keywords, TF-IDF/logistic, activation/logistic, development-selected score fusion | These are research baselines, not comprehensive comparisons with every safety classifier |
| Calibration | Development-only layer/C/fusion selection; recall objective with safe-response false-block cap; calibrated operating-point curve | Safe-response blocks are not the same as benign-prompt blocks |
| Final evaluation | Per-source metrics, precision/recall/AUROC/AP, paired source-bootstrap interval, explicit denominators | Historical-response blocking proxy, not prevention of new Qwen harm |
| Generated audit | Optional fixed subset, per-example greedy generation checkpoint, paired reuse, blinded annotation import | New outputs need independent labels; historical labels cannot substitute |
| Continuation | Separate stored-response probe, optional buffered token-prefix replay on generated outputs | Full-response calibration does not control prefix/multiple-check error; no text released before checks finish |
| Authority boundary | Executable deny-by-default capability-gate tests | No real tool execution, authenticated principal system, or neural privilege isolation |
| Stress/replication | Optional bounded prompt-format tests; strict same-cohort cross-model run comparison | Not a validated adaptive-jailbreak benchmark or zero-shot cross-model probe transfer |

## Persistence

All run artifacts are under `MyDrive/NFW-003/<RUN_ID>/`. There is no silent ephemeral-disk fallback. Sampling, per-example activations, model selection, generation and optional stress work have restart checkpoints. A single self-checksummed envelope commits each record/stage, and committed stage files are bound into the manifest. A changed model, dataset, tokenizer, split, detector, threshold or frozen configuration refuses reuse.

Do not run concurrent writers against one folder. Drive's remote mount is not transactional cloud storage: checksums detect many problems, but cannot guarantee against every quota, synchronization, disconnect or storage failure. Corrupt artifacts fail closed rather than silently rerunning and overwriting evidence.

## Research decision

Advance to a larger generated-output audit only if the activation or fused monitor provides useful improvement over the text-only baseline at an acceptable benign false-block cost across held-out sources. An infeasible calibration target or null result is informative. Do not optimize on final outcomes.

The default run produces a complete **proxy report without manual review**. Optional generated-output behavioral results remain explicitly unavailable until annotations are complete. Harm/refusal labels do not identify task success.

## Local verification

```sh
python3 -m unittest discover -s neuralFirewallV2/tests/unit -p 'test_nfw003_notebook.py' -v
```

Tests execute code extracted from the actual notebook. Full authenticated Necent ingestion, mounted Google Drive durability and a pretrained 3B Colab GPU run require separate integration validation; local offline tests are not evidence those services were exercised.

## Sources and data handling

- [Necent dataset card](https://huggingface.co/datasets/Necent/llm-jailbreak-prompt-injection-dataset): schema, gate, missing labels and underlying-source license obligations.
- [Datasets 4.3.0 stream checkpoints](https://huggingface.co/docs/datasets/v4.3.0/en/stream#save-a-dataset-checkpoint-and-resume-iteration): saved stream state; current-shard rereading can occur.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): finite runtime lifetimes and Drive I/O limitations.

The notebook saves sampled prompts and stored responses in **your Drive**, which can include unsafe text and source-specific restrictions. Keep raw artifacts private and check upstream permissions before publishing. Tokens are obtained through Colab Secrets/hidden input and are not written into result artifacts.
