# NPB Integrated held-out robustness evaluation

This directory evaluates the implementation frozen at `c6970db`. No file in `../npb_integrated/` is modified. The case set is new after that freeze, authored with knowledge of the development limitations; it is a diagnostic challenge set rather than a blinded or representative deployment benchmark.

Read [PROTOCOL.md](PROTOCOL.md), [SCORING_PROTOCOL.md](SCORING_PROTOCOL.md), `cases.jsonl`, `ground_truth.jsonl`, and `freeze.json` for predeclared conditions. The four categories each contain twelve cases. Python character offsets identify exact source and attacker text; document hashes are over UTF-8 bytes.

The original 36 tests are captured in `original_tests.log`. Runtime dependencies, including the exact untracked NPB-Core files, are hashed in `freeze.json`. Reproduction requires those same files; no substitute implementation is bundled or silently imported.

`run_eval.py` imports the frozen assistants and provider. It preserves their early abstentions, uses two identical transport instances for the same local CPU model, records actual calls, and enforces the global budget. It writes raw results exclusively and refuses to overwrite previous output. The model service must be local and already contain the original cached digest.

Results: 48 primary inputs and eight predeclared repeats, 112 system requests and 76 actual CPU model calls. B completed 25/48 tasks (18 numeric answers and seven justified abstentions), and unnecessarily abstained on 23. It returned no exact attacker markers in six trials. Baseline A had 33 definitely correct completions, one extraction error, one exact-marker redirection and 13 review-required prose responses. No frozen defense or scoring file changed during execution.

Read [FINAL_REPORT.md](FINAL_REPORT.md), with every defense failure preserved in [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md). [REPRODUCIBILITY.md](REPRODUCIBILITY.md) documents exact commands, untracked dependency limitations and verification. The temporary model service has been stopped; saved-result verification and all host-failure reproductions need no model.
