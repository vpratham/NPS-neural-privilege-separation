# BIPIA read-permission analysis

This reproducible analysis covers the saved sampled BIPIA evaluation artifact, SHA-256 `c300803ae666c1dfa83e5cc0e2e9cc70929910a718126e277f466ab0fcc6693a`. It validates the artifact's 175 selected cases and source-file hashes against BIPIA commit `a004b69ec0dd446e0afd461d98cb5e96e120a5d0` before exporting aggregates.

The run used `Qwen/Qwen2.5-0.5B-Instruct` at commit `7ae557604adf67be50417f59c2c2f167def9a775`, CPU `float32`, greedy decoding, layers 6 and 18, projection dimension 8, seed 17, and context limit 4096. It contains 50 EmailQA, 75 TableQA, and 50 CodeQA cases. WebQA and summarization were not run because the upstream repository requires separately obtained source datasets.

Across the 1,750 recorded generations (nine regular generations and one alternate denied-payload generation per case), 1,657 ended in EOS, 93 reached the generation limit, and zero recorded an error. The incomplete count is retained in every denominator; it was concentrated in TableQA and CodeQA.

All 175 alternate denied-payload comparisons passed: each preserved input shape, every recorded generation-step logit byte sequence, and generated tokens under the read-permission arm. This is a structural isolation result for denied documents in this exact configuration.

Readable attack text produced exact response differences from the clean control in 42/175 ordinary-arm cases, 42/175 isolated-memory cases, and 51/175 read-permission cases. These are output-difference counts only. They are not an attack-success rate, a refusal rate, or a utility score.

No complete semantic review has been performed. The ignored local packet contains the attack goal, task request, reference, attacked output, and paired clean control; its separate mapping is cryptographically bound to the packet and raw artifact. Its label template is blank by default. Four selected, output-hash-bound qualitative observations are retained privately, but have no prevalence denominator and are not an ASR. BIPIA's official semantic evaluation or documented human review is required before any claim about prompt-injection success or resistance.

Reproduce with:

```sh
MPLCONFIGDIR=/private/tmp/mpl /Users/prathamvasa/Desktop/research/venv/bin/python -m neural_state_firewall.bipia_analysis \
  --report neural_state_firewall/artifacts/bipia_read_permission_20261002.json \
  --bipia-root /private/tmp/nps-BIPIA \
  --data-dir docs/neural_state_firewall_paper/data \
  --figure-dir docs/neural_state_firewall_paper/figures \
  --review-dir neural_state_firewall/artifacts
```

The tracked exports are `bipia_analysis_20261003.json`, `bipia_analysis_20261003.csv`, `bipia_readable_output_changes.png`, and `bipia_denied_structural_invariance.png`. Raw model outputs and the review packet are intentionally ignored.
