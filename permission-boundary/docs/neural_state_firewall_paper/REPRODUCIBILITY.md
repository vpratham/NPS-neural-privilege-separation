# Reproducibility

## Scope

The reported numerical claim is reproducible only with the local raw artifact and the pinned BIPIA/model snapshots. The artifact contains generated outputs and is intentionally ignored by Git. Its SHA-256 is `c300803ae666c1dfa83e5cc0e2e9cc70929910a718126e277f466ab0fcc6693a`. This repository releases the runner, documentation, source hashes, and aggregate results rather than raw BIPIA-derived records.

## Inputs

- Repository commit containing `neural_state_firewall/bipia_evaluation.py` and `read_permissions.py`.
- Microsoft BIPIA checkout: `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`.
- Qwen/Qwen2.5-0.5B-Instruct: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Python environment with the repository's `neural_state_firewall/requirements.txt`; recorded run versions were PyTorch 2.6.0 and Transformers 4.57.6.

## Rerun

```sh
HF_HOME="$PWD/neural_state_firewall/.cache/huggingface" HF_HUB_OFFLINE=1 \
neural_state_firewall/.venv/bin/python -m neural_state_firewall evaluate-bipia \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-files-only \
  --policy neural_state_firewall/examples/bipia_evaluation_policy.txt \
  --layers 6,18 --max-context 4096 --bipia-root /path/to/BIPIA \
  --timeout-seconds 120 \
  --output neural_state_firewall/artifacts/bipia_read_permission_YYYYMMDD.json
```

The runner refuses to overwrite an existing artifact and checkpoints each completed case. It hashes BIPIA source files, policy, model/tokenizer configuration, source contexts, attacks, documents, and generated tokens.

The recorded runs used an Apple M2 with 8 GiB memory, macOS 26.6.2, Python 3.11.12, CPU float32, PyTorch 2.6.0, and Transformers 4.57.6. This hardware description is provenance, not a latency claim.

## Regenerate sanitized analysis safely

Use a new review directory on every run. The analysis command can create private review-packet material; reusing a directory that later contains human labels could overwrite those labels.

```sh
MPLCONFIGDIR=/private/tmp/mpl \
neural_state_firewall/.venv/bin/python -m neural_state_firewall.bipia_analysis \
  --report neural_state_firewall/artifacts/bipia_read_permission_20261002.json \
  --bipia-root /path/to/BIPIA \
  --data-dir docs/neural_state_firewall_paper/data \
  --figure-dir docs/neural_state_firewall_paper/figures \
  --review-dir neural_state_firewall/artifacts/reviews_YYYYMMDD
```

The tracked analysis JSON/CSV and both figures reproduce byte-for-byte from the recorded inputs. The review directory remains local and is not a source of manuscript denominators or semantic labels.

## Verify the host-filtering comparison

The separate raw artifact is `neural_state_firewall/artifacts/bipia_drop_denied_20261003.json` (SHA-256 `5572f51f2b46b13bbbcb9dabb00b65775039634dd22289eadda499466ef011b4`). Its sanitized aggregate is `data/drop_denied_baseline_20261003.json`.

```sh
python3 - <<'PY'
import json
r = json.load(open('neural_state_firewall/artifacts/bipia_drop_denied_20261003.json'))
s = r['summary']
assert (s['n'], s['same_generated_tokens'], s['same_eos'],
        s['same_released_output'], s['errors'], s['completed']) == (175, 175, 175, 175, 0, 166)
print('host filtering matched all 175 clean read-permission outputs')
PY
```

This baseline omits host-denied documents before ordinary decoding. Its per-result generation timings sum to 492.40 seconds; full runner wall time was 524.68 seconds including setup. It is security-equivalent for the measured clean-output property and is a separate-run diagnostic, not a latency comparison.

## Reproduce the pretrained Llama mechanics check

The local artifact is `neural_state_firewall/validation/smollm2_permission_development_20261003.json` (SHA-256 `da5e9b3d2d024de92d30301149e81665e4238f1ac1c7b0576f86c1a1f60275c8`). It records seven synthetic development cases, 7/7 structural passes, 28 EOS attempts, and zero errors.

```sh
HF_HOME="$PWD/neural_state_firewall/.cache/huggingface" HF_HUB_OFFLINE=1 \
neural_state_firewall/.venv/bin/python -m neural_state_firewall evaluate-permissions \
  --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac --local-files-only \
  --policy neural_state_firewall/examples/permission_policy.txt --layers 6,18 \
  --documents neural_state_firewall/examples/permission_documents.json \
  --read-permissions neural_state_firewall/examples/read_permissions.json \
  --cases neural_state_firewall/examples/permission_cases.json \
  --max-context 2048 --max-new-tokens 64 --timeout-seconds 60 \
  --output neural_state_firewall/validation/smollm2_permission_development_RERUN.json
```

The fixture's exact-output task checks are deliberately strict and scored 0/7 in each arm because sentence-form answers did not equal the literal expected strings; they are not utility results.

## Reproduce the readable-injection development challenge

The retained artifact is `neural_state_firewall/validation/readable_challenge_20261003.json` (SHA-256 `2fb999e40a4f250630a11b27574fa7a703d037afaf6d437f117be5740d76998d`). It contains 56 recorded attempts using four author-created clean cases and six fixed literal-marker templates. Three earlier challenge processes were inadvertently concurrent and later terminated; their per-attempt records were not retained. Therefore timeout attribution, timing, and total launch/attack budget are unknown. It is development material, not an independent, held-out, adaptive, fixed-budget, or general-ASR evaluation. No threshold or template was retuned after the incident.

```sh
HF_HOME="$PWD/neural_state_firewall/.cache/huggingface" HF_HUB_OFFLINE=1 \
neural_state_firewall/.venv/bin/python -m neural_state_firewall.development_challenge \
  --manifest neural_state_firewall/examples/readable_challenge_cases.json \
  --policy neural_state_firewall/examples/permission_policy.txt \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-files-only \
  --output neural_state_firewall/validation/readable_challenge_RERUN.json
```

The command refuses to overwrite an output. Run one process at a time, preserve failed/error attempts in every denominator, and never execute generated text.

## Verify the recorded result

```sh
python3 - <<'PY'
import json
p = 'neural_state_firewall/artifacts/bipia_read_permission_20261002.json'
r = json.load(open(p))
assert len(r['rows']) == 175
assert r['structural_invariance_passed'] is True
assert all(x['denied_content_invariance'] for x in r['rows'])
assert r['semantic_attack_success_scored'] is False
regular = [a for x in r['rows'] for c in x['conditions'].values()
           for a in c.values() if isinstance(a, dict) and 'error_type' in a]
alternate = [x['conditions']['attack_denied']['read_permissions_denied_variant']['alternate']
             for x in r['rows']]
assert len(regular) + len(alternate) == 1750
assert sum(a['ended_with_eos'] for a in regular + alternate) == 1657
assert not any(a['error_type'] for a in regular + alternate)
print('175 structural checks passed; semantic attack success intentionally unscored')
PY
```

Then compile the manuscript with the Tectonic commands in `README.md`. The draft compiled with Tectonic 0.17.0 without unresolved citations or overfull boxes. Verify that Tables 1--2 match the artifacts.

## What cannot be reproduced from this package alone

Raw BIPIA-derived contexts, attack strings, and completions are not tracked here. QA and summarization were not evaluated. The run is CPU-only and its summed generation time is not a deployment latency measurement. Reproduction establishes the stated structural comparison; it does not supply semantic labels or prove prompt-injection resistance for readable evidence.
