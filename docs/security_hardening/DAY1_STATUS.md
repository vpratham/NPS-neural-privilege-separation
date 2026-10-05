# Day 1 execution status — October 5 update

**Plan window:** October 4–10, 2026. **Code/evidence baseline:** `3ebbb4c6ad22e26a614b3591cf11864f97ba6f3a`. This page reports current work; a task start or test passed on a development fixture does not imply a release gate passed.

## Completed locally

- Froze the first deployment scope, threat matrix, candidate trust contract and numerical release criteria in `SPRINT_CONTRACT.md` and `release_gates_v1.json`. These specify intended gates; the runtime/profile/evaluation freeze is still pending.
- Added `neural_state_firewall/split_manifest.py`, a small metadata-only validator for split sizes, benign-only fit/calibration partitions, duplicate request hashes, source/template/pair/document/request overlap, and known excluded sources. It reports that final evaluation is **never authorized by metadata alone**. Five regression tests cover those checks.
- Added `data_manifest_v1.json` for seven already-seen permission fixtures. The check reports 0 training, 0 calibration, 0 final-test, and 0 adaptive-test cases; requested targets remain 200, 100, 300, and 10. Fresh rights review, source-separated corpus and custody are absent. The program exits 2 for this incomplete assembly, as intended.
- Added a Colab-only baseline runner that refuses local execution/CPU fallback, verifies a 167,741-byte source archive by SHA-256, applies a cooperative single-run lock, bounds subprocesses and saves evidence. Its two local regression tests pass.
- Added the October 4 author facts supplied so far to the manuscript source and submission documents: `prathamv1102@gmail.com`, no conflicts, and no funding. Affiliation, ORCID, fee budget, license and final author checks remain open. Recompiled the nine-page PDF with Tectonic and visually checked page 1; the email and unresolved author fields display cleanly.

## Verification performed

- Restored firewall suite: **86 passed** when rerun with loopback access after five socket tests were blocked by the sandbox on the first invocation.
- Separate action-broker suite: **34 passed**.
- New split-manifest/Colab-runner tests: **7 passed**.
- Legacy notebook suite: **58 attempted, 56 passed, 2 import errors** because scikit-learn is unavailable (`test_nfw002_v2_notebook.py`, `test_nfw003_notebook.py`).
- The full firewall suite has not yet been rerun after adding the seven new tests.

## Colab result: pinned baseline and development mechanics completed

The original Colab CLI 0.6.0 allocated a Tesla T4 (15 GiB reported memory) and accepted the source archive. Its Python 3.13.15 runtime reached environment setup, then `python -m venv` failed because that image does not provide working `ensurepip`. I preserved the hardware and setup-failure report at `neural_state_firewall/artifacts/day1_colab_setup_failure.tar.gz` (local ignored artifacts; do not publish it as model results).

Google's official CLI changelog documents an idle-session keep-alive correction in 0.6.0; I upgraded the installed CLI to 0.7.4 and changed the runner to make a pip-less venv and target it with the system pip. Two subsequent runs exposed missing archive contents (the gateway fixture, then `nps_gateway`); those failures stopped before GPU generation and were retained as ignored local artifacts. The corrected archive SHA-256 is `664752df3cdf8fd7241187e32c317d2b0fb6e0daf4c37a857617cae74c71e0dc`.

The third run completed on a T4 with Python 3.13.15, Torch 2.6 and Transformers 4.57.6. Both pinned suites passed (firewall: 86 tests; gateway/broker: 34 tests). The 0.5B Qwen permission evaluation then completed in 303.622 seconds: seven already-seen cases, four arms per case (28 generation attempts). The runner's final gate passed, which requires 7/7 structural checks and all 28 attempts ending with EOS without an execution error. This is development-mechanics evidence only; the raw case-level JSON/log bundle was not recovered before Colab pruned the idle session, so its SHA-256 and detailed per-arm records are unavailable. The session-history record is retained locally under ignored artifacts. No fresh semantic evaluation, GPU performance benchmark, monitor calibration or efficacy result was produced.

## Still blocking the fresh efficacy gate

No locally verified fresh corpus has been found. The exclusion registry now marks historical XSTest material seen because earlier extraction/scoring work consumed it, including a metadata-intended read that unexpectedly exposed embedded records. BIPIA, NeurAlchemy and prior fixtures also remain excluded. The metadata validator cannot establish novelty, source rights, semantic deduplication, payload binding or independent custody by itself.

The next useful work is to acquire and source-review a genuinely fresh evaluation corpus before freezing a final-test protocol. Until those steps occur, readable-evidence resistance, permission-path monitor enforcement, general production approval and journal submission all remain unapproved/incomplete.
