# Reproduce and verify the frozen robustness run

## Source identity and dependency availability

The reference is `c6970dbe38669dd0b6dedfcab7c48cc53a9fdf55`. `freeze.json` records all 40 original package hashes and 14 runtime-import file hashes. The required untracked files are exactly:

- `permission-boundary/npb_core/__init__.py`
- `permission-boundary/npb_core/core.py`
- `permission-boundary/npb_core/broker.py`

They were available in the evaluated workspace; their exact bytes were used. This evaluation does not make a fresh checkout self-contained. A missing or mismatched dependency must be restored from the verified project evidence, not replaced with another implementation. The verification script fails closed on any mismatch. Other runtime dependencies are tracked and listed in the same manifest.

`protocol_freeze.json` records the UTC freeze time, all eight case/scoring/execution protocol hashes, 48-case distribution, 67 source/attacker-span checks, 41 expected-answer fixture checks, 17 scorer tests, repeat IDs and budgets. Fixtures are scorer unit controls, not generated model responses. No held-out model output existed at that freeze. `execution_config.json` records the actual start and model identity. Original cases, repeated cases, code, prompts and results under `npb_integrated/` were never changed.

## No-inference verification

From repository root:

```sh
git rev-parse c6970db^{commit}
python3 -m unittest discover -s permission-boundary/npb_integrated_heldout/tests -v
python3 permission-boundary/npb_integrated_heldout/verify_results.py
(cd permission-boundary/npb_integrated_heldout && shasum -a 256 -c SHA256SUMS)
```

The verifier recomputes source, dependency, protocol and scoring checks; validates case joins, 96 primary and 16 repeated system results, CPU records and the call limit; compares all repeated outputs; and reproduces the 23 host failures without inference. An inference guard raises if a reproduction reaches a provider. The script also checks D07/D08 attacker-span removal at pure host preflight; it generates no clean model answer.

Run one exact failure:

```sh
python3 permission-boundary/npb_integrated_heldout/reproduce_failure.py --case D07
```

Re-run the original tests with the correct historical package precedence:

```sh
cd permission-boundary
PYTHONPATH=.. python3 -m unittest discover -s npb_integrated/tests -v
```

## Model execution command

The run used a temporary local Ollama 0.21.1 service with the pre-existing cached `qwen2.5:3b` artifact; no pull occurred. Model digest, options, exact prompts and token counts are in `execution_config.json` and each generation record. The host was Apple M2/Python 3.14.6. The service was stopped when the run finished. Server startup initialized Metal capability discovery; all model requests used `num_gpu: 0` and all 76 saved generations reported `size_vram: 0`.

The original execution commands were:

```sh
OLLAMA_HOST=127.0.0.1:11436 OLLAMA_NO_CLOUD=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_MODELS=/Users/prathamvasa/.ollama/models ollama serve
python3 permission-boundary/npb_integrated_heldout/run_eval.py
```

The model tag must resolve to digest `357c53fb659c5076de1d65ccb0b397446227b71a42be9d1603d46168015c9e4b`; the harness checks it. Two identical frozen transport instances respect the original provider's 80-call instance cap; a global 120-call budget applies. No retries or dummy generations occur. The requested 48×2 system pass makes only 66 model calls because B returns early on 30 inputs. Repeats add ten, for 76 total. This is an explicit consequence of preserving B, not missing/fabricated model responses.

The runner uses exclusive file creation and will not overwrite this execution. A future model repeat must be a separately identified execution in a new evaluation directory with the same frozen source/protocol hashes; do not delete or overwrite these files. This dataset is now exposed and must not be reused as untouched held-out data after tuning.

## Artifact interpretation

`baseline_results.jsonl` and `defended_results.jsonl` each have 48 records. `repeat_*_results.jsonl` each have eight. Host-only results correctly have `generation: null`, zero model attempts and no token count. `scores.json` retains all frozen deterministic scores. `review_queue.json` lists 13 baseline prose results awaiting human semantic review. No LLM judge or independent human adjudication supplies replacement labels.

The authored diagnostic cases are not a representative sample; no population confidence interval is claimed. Numerical equivalence scoring is deliberately conservative about multiple values and nuanced prose, so report review counts with full category denominators rather than treating missing judgments as failures or successes. Source-span validity measures emitted provenance, with coverage separately disclosed.

All 23 defense failures and original outputs are in `FAILURE_ANALYSIS.md`. `failure_index.json` is the compact list. `verification.json` records final checks. `SHA256SUMS` covers every retained evaluation file except itself and Python bytecode caches. Checksums show byte preservation, not semantic correctness.
