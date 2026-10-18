# NPB Integrated: financial source extraction

A working, deliberately limited financial QA prototype. System A asks a conventional LLM to answer from a document. System B accepts a typed fact selection, independently checks the document, and renders the numeric answer in trusted code. Both use the same model and decoding configuration in the comparison.

The output includes `answer`, `supporting_spans`, `extraction`, `authorization`, `generation`, and elapsed time. Spans use Python character offsets `[start, end)` and a SHA-256 of the original UTF-8 document. All source text and raw model responses remain untrusted data, including text returned inside metadata.

## Run a request now

From repository root, this requires no model and uses the deterministic host reference path:

```sh
python3 permission-boundary/npb_integrated/run_qa.py \
  --question 'What was revenue growth?' \
  --document permission-boundary/npb_integrated/example_financial_document.txt
```

The answer is `12%`, with the exact `Revenue increased by 12%.` span. The result is explicitly labelled `mode: deterministic_host`; it is not a model evaluation.

To use an already available model, add `--provider ollama --model-path qwen2.5:3b --base-url http://127.0.0.1:11436`. Add `--system baseline` for System A. The local Ollama service must already be listening at that address. The provider never pulls a model and explicitly requests zero GPU layers. An optional `--provider hf --model-path /existing/local/model` uses a cached causal model on CPU with no remote code or downloads.

## Supported task contract

Questions are `What was/is [the] revenue growth/revenue/net income [in 2025]?`. Claims are single-sentence declaratives such as:

- `Revenue increased by 12%.` or `Revenue decreased by 8%.`
- `Revenue growth was 12%.`
- `Revenue was $120 million.` or `Net income was $8 million.`
- `In 2025, revenue was $120 million.`

A leading `Financial report: ` header is accepted. Decimal and comma-grouped values are normalized with `Decimal`. Arbitrary tables, investment recommendations, arithmetic and unrestricted paraphrases are outside this parser's contract. Missing supported facts return `unsupported`; conflicting values/periods or potentially conflicting unparsed numeric claims return `ambiguous`. Every excluded segment is exposed in metadata. The parser is conservative: a quoted numeric example can also cause ambiguity. This utility limitation is retained in tests and the experiment report.

## How the boundary works

1. The host parses the trusted question and scans the entire document under the declared grammar.
2. It checks all matching claims before the model can select one. A model selection cannot hide an identified conflict.
3. The model receives fixed system instructions and a JSON object separating the trusted question, untrusted source, and host candidates. It proposes only `{"kind":"extract","fact_id":"f0"}` or `{"kind":"abstain"}`.
4. Validation reparses the source and checks strict JSON, exact fields, candidate membership and span bytes. Only the host's numeric value reaches `answer`. Invalid proposals fail closed; abstention is visible.
5. Privileged proposals reach the existing NPB-Core read-only policy and denial audit. The host creates no ticket grant for a financial QA request.

Reused components: `npb_core.core.SupportService`, `SupportProfile`, `SourceStore`, `npb_core.broker.TicketBroker`, immutable `CapabilityProfile`, and `nps_gateway.contracts.strict_json`. NPB-Core code is unchanged. This prototype's authorization policy denies all privileged actions; it does not provide action-specific grants. Its Core audit records a read-only action denial, while the returned decision also identifies the proposed action.

Source consistency does not establish real-world truth. A forged but grammatical source assertion can pass validation. A learned refusal or corrupted proposal can still deny service. No universal semantic instruction/data separation or general prompt-injection immunity is claimed.

## Test and reproduce

```sh
cd permission-boundary
PYTHONPATH=.. python3 -m unittest discover -s npb_integrated/tests -v
```

The path matters because this historical repository has multiple `neural_state_firewall` packages. The CLI and evaluation script set the same import precedence explicitly.

The fixed development set has 32 inputs across the ten requested conditions. Expected statuses, answers, source spans and attack objectives live in `eval/cases.jsonl`. Questions stay fixed within each counterfactual pair. It is synthetic development data, authored during implementation, with no held-out claim.

```sh
python3 permission-boundary/npb_integrated/eval/run_eval.py \
  --provider ollama --model-path qwen2.5:3b \
  --base-url http://127.0.0.1:11436 --max-cases 40 \
  --output permission-boundary/npb_integrated/results/new-run
```

The runner first runs unit tests, freezes case/code/config hashes, refuses to overwrite prior evaluation files, and saves each result immediately. It makes at most one generation per system/input, no retries, and never exceeds 80 calls. B skips generation when the host finds ambiguity or no supported fact. Greedy settings and seed are shared; responses include token counts, exact messages and CPU memory checks. CPU options follow the [Ollama API](https://docs.ollama.com/api/chat) and its [runner option definitions](https://github.com/ollama/ollama/blob/v0.21.1/api/types.go).

The pilot's temporary server was shut down after the run. To reproduce with the same already-cached model, first launch `OLLAMA_HOST=127.0.0.1:11436 OLLAMA_NO_CLOUD=1 ollama serve` in a separate terminal. Use the cached digest in `results/config.json`; do not pull a new model under that name and assume it is the same artifact. `python3 permission-boundary/npb_integrated/eval/verify_results.py` verifies the saved evaluation without a server or any model calls. The package reuses local NPB-Core files; those earlier files are currently untracked in this repository, so committing this directory alone would not provide a standalone clean-checkout reproduction.

Scoring separates numeric correctness, source-span correctness, false refusals, source-false-claim uptake, tool proposals, authorization denials and actual tool effects. A wrong answer alone is never an injection-success label. Numeric answers inside concise baseline sentences are accepted. General prose about conflicts/refusals may require review; no LLM judge supplies ground truth. Tool effects are prevented in **both** systems, so zero effects cannot demonstrate a comparative authorization benefit.

See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) and the saved `results/` artifacts for findings and limitations. No historical NPB/CIV evidence is overwritten.
