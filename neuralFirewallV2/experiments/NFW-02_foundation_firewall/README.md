# NFW-002 Foundational Monitor-and-Block PoC

`NFW_002_Foundational_Monitor_Block_POC.ipynb` is the starting point for a defensible neural-firewall implementation. It is a frozen-model, pre-generation activation-monitor and block experiment. It is intentionally not an activation-steering experiment and makes no neural-privilege-separation claim.

For a complete walk-through of its design, safeguards, outputs, and interpretation, read [the notebook guide](NFW_002_NOTEBOOK_GUIDE.md).

It requires a reviewed local prompt dataset at `data/foundation_prompts.jsonl`. Each JSONL record needs:

```json
{
  "id": "stable-example-id",
  "messages": [{"role": "user", "content": "request text"}],
  "intent_label": 0,
  "group_id": "underlying-behavior-or-paraphrase-family",
  "source": "dataset-name"
}
```

`intent_label` is the experiment's request-level block label. It is not a label for whether a generated response is harmful. `group_id` must keep paraphrases, templates, and a common base behavior together during splitting.

The notebook creates an immutable run directory under `nfw002_outputs/`, containing the grouped split manifest, model/tokenizer/activation-site provenance, monitor artifact, actual baseline/firewall Qwen responses, and final report.

After generating responses, copy `target_model_responses.jsonl` into a labeling workflow. Create `response_labels.jsonl` in the same run directory with one record for every released response:

```json
{
  "run_id": "nfw002_...",
  "id": "stable-example-id",
  "condition": "baseline",
  "harmful_assistance": 0,
  "task_success": 1
}
```

Use a frozen response evaluator plus blinded human review of a stratified sample before treating output labels as research evidence. The notebook rejects incomplete, duplicate, or cross-run labels and retains generation errors as terminal records.

The notebook pins the Qwen revision used by NFW-001. Keep one model revision, one reviewed two-class dataset with real group IDs, deterministic decoding, and a small budget for the first run. Do not tune on the final split. The next milestone after a valid result is a continuation monitor with this same serialization, activation-site, label, and accounting contract.
