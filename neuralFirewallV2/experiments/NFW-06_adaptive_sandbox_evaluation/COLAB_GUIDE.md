# NFW-006 Colab guide

1. Open `NFW_006_Adaptive_Sandboxed_Capability_Firewall.ipynb` in Google Colab.
2. Run all cells in order.
3. Authorize Google Drive access when prompted.

No GPU or model download is required. The experiment runs the broker, adaptive attack generator, mock side effects, and audit telemetry on CPU.

Artifacts are written to:

```text
MyDrive/NFW-006/nfw006_adaptive_sandbox_001/
```

Important artifacts:

```text
manifest.json
attack_cases.json
attack_progress.json
attack_results.json
security_results.json
final_report.json
REPORT.md
```

## Reconnecting after a Colab disconnect

Run the notebook again with the same `RUN_ID`. The adaptive attack loop checkpoints every `CHECKPOINT_BATCH` cases and continues from the saved `attack_progress.json`. Do not run two writers against one run folder.

Use a new `RUN_ID` after changing code, seed, case count, checkpoint size, or experiment identity. The manifest and stage hashes intentionally refuse incompatible cache reuse.

## Success criteria

A successful run should report:

```text
status: complete
unauthorized_side_effects: 0
security_results.all_passed: true
security_results.audit_chain_valid: true
```

The experiment is intentionally self-contained and does not require NFW-002, NFW-003, NFW-004, or NFW-005 Drive data.
