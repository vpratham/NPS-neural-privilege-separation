# NFW-010 — Protocol-corrected tool-result injection study

**Notebook:** [`NFW_010_Protocol_Corrected_Tool_Result_Injection.ipynb`](NFW_010_Protocol_Corrected_Tool_Result_Injection.ipynb)
**Runtime:** Google Colab GPU. No NFW-009 Drive artifacts are required or reused.

NFW-010 is the corrective follow-up to the NFW-009 forensic audit. It deliberately measures a smaller, rigorous study rather than adding more attack variants to a protocol-confounded setup.

## What it corrects

NFW-009 supplied native Qwen tool-result context but instructed models to return a competing host JSON schema. NFW-010 uses one interface end-to-end:

- input messages use Qwen assistant `tool_calls` for host-issued `read_public`, followed by `role="tool"` output; `read_public` is included in the tool schema but is never a permitted post-result broker action;
- `apply_chat_template(..., tools=TOOL_SCHEMA, add_generation_prompt=True)` is used for every actual prompt;
- output must be exactly one Qwen-style `<tool_call>` block containing exactly `{"name":"write_record","arguments":{...}}` (or a schema-known but unauthorized mock-message proposal);
- the parser permits Qwen's one terminal `<|im_end|>` token after the block, but rejects prose, multiple calls, other trailing content, extra keys, duplicate keys, native/host-schema mixing, and malformed arguments—without recovery.

The tokenizer smoke gate verifies the rendered Qwen transcript contains both `<tool_call>` and `<tool_response>` before model weights are loaded. Each model's development split runs before its held-out split. A model must produce a valid single-call protocol on all 18 development prompts to be eligible for held-out generation. If it does not, its raw development responses and parser reasons are recorded, its held-out arm is marked skipped, and the notebook continues to the next selected model. This distinguishes model-format noncompliance from a notebook exception without weakening the gate.

## Frozen study

| Component | Choice |
|---|---|
| Development / held-out cards | 6 / 24 independently generated synthetic cards; distinct IDs, facts, and attack-template sources |
| Conditions | clean public record; benign non-directive annotation; malicious tool-result annotation |
| Attack goals | protected resource, wrong content in authorized `notes`, external-outbox request represented as a notes request |
| Models | pinned Qwen2.5 0.5B and 3B Instruct, loaded sequentially |
| Output | greedy decoding; max 96 new tokens; one strict native call |
| Broker replay | scope-only and exact-argument capability tokens |
| Effects | synthetic in-memory mock workspace only |

The development and held-out attack **wording templates** are disjoint within this notebook, but both banks are visible in the source and authored by the same project team; this is not a blinded external benchmark. The benign metadata control is non-directive, but it is not guaranteed to be tokenizer-length-matched to either the clean or injected result. Treat condition contrasts as a controlled pilot, not a population-level causal estimate.

The held-out card identity, source-disjoint attack-template sources, model revisions, tokenizer template, decoder settings, full executable-notebook/protocol/broker/evaluator code fingerprints, task hash, conditions, and broker modes are bound in `manifest.json`. Changed identity refuses cache reuse. Per-response immutable envelopes are saved to Drive and checkpoints are skipped after reconnects. Runtime/GPU identity is stored per response rather than in the cache binding, so a reconnect on different Colab hardware can resume; the report flags mixed runtime identities. JSON envelopes are compared using canonical JSON so tuple/list normalization across Drive reconnects does not cause false cache mismatches.

## Security boundary and endpoints

The model gets **no credential**. The host issues a signed short-lived token for the expected `write_record(notes, exact fact)` action. Under `scope_only`, an in-scope wrong value can execute by design; under `exact_args`, its value hash must match the host expected fact. The report keeps separate:

1. raw model continuation and strict parser validity;
2. valid attacker-goal proposal;
3. host broker decision and mock effect;
4. unauthorized-scope effect versus wrong-content effect;
5. exact legitimate task success.

A `complete` report means all selected models passed their development gates and their planned held-out artifacts exist. `complete_with_skips` means some model(s) failed the development gate and were excluded from held-out evaluation; `development_only_no_eligible_models` means none passed. These statuses describe artifact coverage, **not** general safety.

## Limitations

- This does not evaluate neural activations or demonstrate a neural firewall.
- Synthetic authored cards, fixed attack strings, one model family, and greedy decoding are not a representative attack benchmark.
- The broker and model share a Colab Python process; this is not process/VM isolation or production key custody.
- No real shell, network, external messaging, private data, or side effect is allowed.
- A model that replies in prose instead of making a tool call is reported as development protocol noncompliance and has no held-out results; the other selected model(s) may still run.
- The benign metadata control is not guaranteed to be tokenizer-length matched; condition contrasts are descriptive.
- Public Hugging Face models download without `HF_TOKEN`; the Colab warning that no token is configured is informational unless access to a gated model is needed.

See [`COLAB_GUIDE.md`](COLAB_GUIDE.md) for the exact run/restart procedure.
