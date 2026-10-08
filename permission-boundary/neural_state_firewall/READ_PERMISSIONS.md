# Deterministic document read permissions

The runtime now enforces host-assigned document visibility inside attention, without an anomaly profile or a prompt classifier. This is a local, runnable information-flow boundary. It does not make readable evidence incapable of influencing the model's instructions or guarantee arbitrary natural-language policy compliance.

## Run

For the authenticated single-realm API with a separate worker and process deadline, use [the pilot guide](PILOT.md). The original commands below remain local development surfaces.

From the repository root, using the existing environment and cached model:

```bash
export HF_HOME="$PWD/neural_state_firewall/.cache/huggingface"
export HF_HUB_OFFLINE=1
neural_state_firewall/.venv/bin/python -m neural_state_firewall run-permissions \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-files-only \
  --policy neural_state_firewall/examples/permission_policy.txt --layers 6,18 \
  --documents neural_state_firewall/examples/permission_documents.json \
  --read-permissions neural_state_firewall/examples/read_permissions.json \
  --requests neural_state_firewall/examples/permission_requests.jsonl
```

The example grants read access to `public` only. The second document contains a synthetic private code and conflicting instructions. Requests can ask questions but cannot change the host grants or source contents. Documents and grants are separate files; only trusted application/operator code may load or replace those files. Source IDs must come from authenticated host ingestion, not from document text.

To serve the same configuration, replace `run-permissions` with `serve-permissions` and omit `--requests`. It binds to loopback port 8765. Send:

```bash
curl http://127.0.0.1:8765/v1/respond \
  -H 'Content-Type: application/json' \
  -d '{"task":"What day is the workshop? Reply with only the weekday."}'
```

Permission mode accepts a task and optional empty context. Client-supplied grants, documents, source labels, policy, and nonempty context are rejected. A deployment with changing retrieved documents must instantiate a new host-bound adapter snapshot, rather than accepting provenance from this endpoint. This version serves one permission realm, not multiple authenticated tenants.

Responses are buffered until EOS. Errors, incomplete generations and expired deadlines release no text. The timeout is checked between forward passes; it cannot preempt a hung native kernel. The adapter rejects concurrent use and removes hooks/cache-layout state when generation closes. No tools run through this path. The separate broker remains responsible for tool effects.

## What the model executes

1. Encode the host system policy into separately sealed per-layer K/V memory.
2. Encode document text and the user task as data. Reserved role/control tokens in their text are split into ordinary tokens; reject a tokenizer that still produces reserved IDs.
3. Assign each document's tokens visibility from the host grant set. Denied documents use 256 token slots each, including wrappers and padding. Oversize denied documents fail closed; they are never silently truncated. The fixed capacity keeps computation shapes independent of denied content length within the capacity.
4. For every decoder layer, permit public queries to read only earlier public keys. Permit denied queries to read only their own key. The same rule applies during cached generation.
5. Use logical public position IDs that do not advance for denied slots. Verify each attention layer received the exact mask; verify protected policy memory before returning each generated step.
6. Release only the completed public response. Permission checks do not depend on the anomaly monitor.

| Query compartment | May read policy | May read permitted evidence/task | May read denied evidence |
|---|---:|---:|---:|
| Policy prefill | Earlier policy only | No | No |
| Public working/output token | Yes | Causally earlier tokens | No |
| Denied document token | No | No | Its own token only |

The information-flow argument is inductive: a public input embedding is independent of denied values; public attention reads only public states; supported models apply normalization, MLPs and residual additions per token. Therefore later public states and output logits remain independent of denied values. Cached steps preserve the same restriction, and greedy output becomes the next public input. Masking only the final query or last layer would not close the indirect paths; this implementation masks every layer and checks that it did so.

The finite-precision checks hold model, policy, grants, source IDs/order, public task/data and capacities fixed while changing denied values. Test fixtures also change denied text length within fixed capacity. Resource errors, compute timing, memory consumption and compromised host/runtime code are excluded. Denied content remains present in host memory and in its isolated self-attention computation; this is not encrypted storage or an OS sandbox. The system policy is readable by the output computation and must not be used as a secret vault.

## Support

- Transformers 4.57.6 and existing Torch dependencies; no new dependencies or trained weights.
- Batch-one greedy float32 inference; full attention and ordinary RoPE; eager and SDPA backends.
- Qwen2/Qwen2.5: tiny-model structural tests and a pinned pretrained Qwen2.5-0.5B run.
- Llama: real tiny randomly initialized model tests under eager and SDPA, plus seven structural checks on pretrained SmolLM2-135M-Instruct (Llama architecture), revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`. This small development run validates mechanics, not useful task behavior; scaled-RoPE variants are rejected.
- Sliding windows, custom model classes, alternative attention backends, quantized models and closed APIs are unsupported.

The audit inspected the pinned local Qwen/Llama implementations. [Transformers' attention interface documentation](https://huggingface.co/docs/transformers/v4.57.1/attention_interface) describes backend selection and mask propagation; model/backend support must be revalidated before upgrading.

## Reproduce the development evaluation

Use the same model/policy/document/grant options with `evaluate-permissions`, then add:

```text
--cases neural_state_firewall/examples/permission_cases.json
--max-new-tokens 64
--output neural_state_firewall/artifacts/read-permission-evaluation.json
```

This runs ordinary decoding, isolated-policy-memory-only, and read-permission enforcement on identical padded serialization, plus a denied-content counterfactual for each case. The ordinary and memory-only controls deliberately have no read restriction and cannot be served as enforcing adapters. Output usefulness and diversion use the explicitly specified exact answers; canary disclosure uses literal synthetic-canary matching. Incomplete and failed runs remain in the report. Logit comparisons cover every generated step, not merely the first or the final answer.

These are seen development fixtures, not a locked final benchmark. The evaluator records the reference CLI configuration identity and each evaluated arm's actual document/grant hashes, input/source hashes, outputs, counters, and single-run generation timings. Timings exclude adapter construction and its repeated weight hashing; they are not an end-to-end latency benchmark or p95 estimate. It always marks production eligibility false. Raw output reports should be handled like their source documents; the tracked example report contains only synthetic fixtures.

### BIPIA sampled read-permission run

`evaluate-bipia` runs a deterministic 175-case sample across EmailQA, TableQA, and CodeQA. It evaluates clean evidence, BIPIA attack text inserted as readable evidence, and the same attack stored in a host-denied document. Each is compared across ordinary decoding, isolated policy memory, and read permissions. A second denied attack variant checks shape, every generated-step logit, and output-token invariance. The runner writes an atomic checkpoint after each case.

To reproduce, clone Microsoft BIPIA at commit `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`, cache the pinned Qwen model specified below, then run:

```bash
HF_HOME="$PWD/neural_state_firewall/.cache/huggingface" HF_HUB_OFFLINE=1 \
neural_state_firewall/.venv/bin/python -m neural_state_firewall evaluate-bipia \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-files-only \
  --policy neural_state_firewall/examples/bipia_evaluation_policy.txt \
  --layers 6,18 --max-context 4096 --bipia-root /path/to/BIPIA \
  --timeout-seconds 120 \
  --output neural_state_firewall/artifacts/bipia_read_permission_20261002.json
```

The local raw report is ignored by Git because it contains generated model outputs: `neural_state_firewall/artifacts/bipia_read_permission_20261002.json`. It records source commit and file hashes, model/runtime identity, case-level hashes, outputs, generation lengths, timings, and summary counters. The BIPIA test sources are added to `docs/neural_state_firewall_paper/data/seen_material_exclusions.json`: this run is now development evidence and must not be presented as a fresh locked evaluation.

In this run all 175 denied-document alternate-payload checks passed, including equal input shape, byte-identical logits at every generated step, and equal greedy output tokens. This is evidence for the scoped structural information-flow property in this Qwen configuration. It does not show that the model resists instructions embedded in evidence it is permitted to read. Readable attack text changed the ordinary response in 42/175 cases (5 EmailQA, 26 TableQA, 11 CodeQA); response changes are not a measure of attack success. The run has **no semantic attack-success score** because we did not run BIPIA's judge or conduct independent outcome annotation.

All arms had zero runtime errors. Across all 1,750 generation attempts (including 175 alternate denied payloads), 1,657 ended at EOS and 93 reached the token cap. Exact/substring reference matches are retained only as weak diagnostics and are not reported as task accuracy. Summed per-generation time was 5,962.68 seconds (about 99.4 minutes), excluding setup and evaluation overhead; this is not an end-to-end latency benchmark. The sample includes 50 email rows with 50 text payloads, 75 of 100 table rows with all 75 text payloads, and all 50 code rows with their 50 payloads. QA and summarization were omitted because their source datasets require separate retrieval/terms. The middle insertion position approximates BIPIA's sentence-based placement using the standard library, so this is BIPIA-derived, not an official full reproduction.

The follow-up [analysis](../docs/neural_state_firewall_paper/data/bipia_analysis_20261003.md) preserves these denominators. Four individually inspected readable-attack outputs provide concrete counterexamples; they do not supply a population attack-success rate. A simpler host-filtering baseline removed denied documents before ordinary decoding and matched all 175 clean permission-arm token sequences and release outcomes. Retaining masked denied tokens has no demonstrated utility advantage on this comparison.

### Recorded development results

The pinned CPU float32 Qwen2.5-0.5B run in [the report](validation/read_permission_development.json) produced:

| Arm | Exact task answers | Exact attack-target answers | Literal canary disclosures | Incomplete/errors |
|---|---:|---:|---:|---:|
| Ordinary, same padded serialization | 7/7 | 0/7 | 0/7 | 0/0 |
| Isolated policy memory only | 7/7 | 0/7 | 0/7 | 0/0 |
| Attention read permissions | 7/7 | 0/7 | 0/7 | 0/0 |

All seven denied-content swaps had equal input shapes, byte-identical logits at every generated step and identical output tokens. Since no baseline attack succeeded, these data demonstrate boundary mechanics and retained utility on these fixtures, not a reduction in attack success. The first development run used variable denied lengths and failed exact logit equality in three of six comparisons; fixed denied-source capacities corrected that defect before this recorded run. No threshold was relaxed.

The original [pretrained HTTP smoke report](validation/read_permission_http_smoke.json) passed five checks. The newer [authenticated pilot smoke](validation/pilot_smoke_20261003.json) passed six real-model HTTP checks, including rejection without credentials. [Randomized engineering checks](validation/boundary_stress_20261003.json) passed 100/100 denied-content comparisons across two model families and two backends, with 100/100 public-input sensitivity checks. These are structural tests, not semantic attack outcomes.

The [pretrained SmolLM2 development report](validation/smollm2_permission_development_20261003.json) passed seven denied-content comparisons, with all 28 generations ending at EOS and no errors. Each arm scored 0/7 on the frozen exact-output task metric: its answers did not match the required output format, and at least the private-fact request was answered incorrectly. Do not treat unchanged logits as task quality or replace this failed metric with a favorable post-hoc interpretation.

Run regression checks:

```bash
neural_state_firewall/.venv/bin/python -m unittest discover -s neural_state_firewall/tests -v
```

## Remaining security and release work

The implemented property is denied-document noninterference under the stated threat model. Readable evidence can still contain persuasive or conflicting instructions. Encoding it as data prevents structural role-token injection; it does not solve semantic instruction takeover. There is no evidence here supporting a universally injection-proof model.

The [pilot](PILOT.md) now supplies single-realm authentication and a process-enforced timeout. General production release still requires representative source-separated held-out tasks, a defense-aware adaptive attacker with recorded budgets, outcome review under a disclosed protocol, utility and latency gates, and deployment-specific principal isolation. The [release decision](validation/release_decision_20261003.json) retains production approval as false. The paper reports the narrower demonstrated property and negative results.
