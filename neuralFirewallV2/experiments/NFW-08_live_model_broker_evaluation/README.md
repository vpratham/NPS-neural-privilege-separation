# NFW-008 — Live model proposals against a capability broker

**Notebook:** `NFW_008_Live_Model_Capability_Broker.ipynb`
**Runtime:** Google Colab GPU; model weights are downloaded sequentially and are **not** stored in this repository or in the run artifacts.

NFW-008 evaluates actual greedy tool proposals from two pinned Qwen2.5 instruct models on the **same 48 authored task cards**. It keeps three questions separate:

1. Does the model propose the correct, strictly formatted action, abstain where no tool is authorized, and resist lower-trust document instructions?
2. Does a host-issued capability broker prevent out-of-scope mock-tool effects even when the model proposes a disallowed action?
3. Does binding a capability to the exact expected arguments prevent wrong-content writes that a scope-only capability would allow, and what utility does that cost?

The task cards and expected actions are generated inside the notebook and frozen before GPU generation. Clean public-read cards are paired by index with injected-document public-read cards, so injection-related task degradation is reported without changing the labels. They cover public reads, synthetic note writes, synthetic mock messages, no-tool requests, direct protected-write requests, and reads containing prompt-injection text. There are eight cards per family. The models run sequentially to reduce Colab memory pressure.

## Reproducibility and restart

The Drive manifest binds the task hash, pinned model revisions, package/Python versions, generation parameters, and a core implementation fingerprint. Each model has a tokenizer-template hash. Each response is stored as an immutable, checksum-validated file tied to its task prompt. On reconnect, completed examples are skipped; a missing example is generated. Existing responses are revalidated before aggregate evaluation, even when a prior report exists. A changed identity requires a new run ID.

## Expected outputs

```text
MyDrive/NFW-008/nfw008_live_models_001/
  manifest.json
  tasks.json
  broker_controls.json
  model_info/qwen_0_5b.json
  model_info/qwen_3b.json
  responses/qwen_0_5b/<task_id>.json
  responses/qwen_3b/<task_id>.json
  evaluation.json
  final_report.json
  REPORT.md
```

## Scientific limits

- This is a small, authored, prompt-template-sensitive benchmark—not an adaptive red-team dataset or a representative rate estimate.
- Both models are from the same family. Their paired comparison is not cross-architecture generalization.
- Correct *tool proposal* is measured, not the quality of any final natural-language answer.
- Broker effects are local temporary mock-tool files. Broker and model share the Colab process, so this is not production isolation.
- The scope-only and exact-argument policies are preregistered variants; a wrong-content effect under scope-only is reported separately from an out-of-scope effect.
- No real shell, network, email, external API, or private data is accessed.

Model references: [Qwen2.5-0.5B-Instruct pinned tree](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775), [Qwen2.5-3B-Instruct pinned tree](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1), and [Transformers chat-template documentation](https://huggingface.co/docs/transformers/v4.57.1/en/chat_templating).
