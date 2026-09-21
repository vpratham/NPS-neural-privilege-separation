# NFW-007 Colab guide

1. Open `NFW_007_Integrated_Boundary_Utility_Telemetry.ipynb` in Google Colab.
2. A CPU runtime is sufficient; a GPU is not used.
3. Select **Runtime → Run all** and authorize Google Drive mounting.
4. Read `MyDrive/NFW-007/nfw007_integrated_boundary_001/REPORT.md` and `final_report.json`.

The notebook is self-contained and does not read NFW-002 through NFW-006 data. A separate Google Drive account is fine.

## Default configuration

```text
RUN_ID = nfw007_integrated_boundary_001
SEED = 20260921
N_EPISODES = 400
N_BENIGN_EPISODES = 100
BATCH_SIZE = 25
```

Each attack episode has eight requests. The benign episodes each have an authorized read, write, and mock message; some also contain recoverable request mistakes to expose alert false positives. Only local temporary mock-tool files are changed.

## Drive artifacts

```text
manifest.json
controls.json
batches/batch_*.json
benign_episodes.json
alerts.json
audit_events.jsonl
final_report.json
REPORT.md
```

## Resume after a disconnect

Run the same notebook again with the same configuration and `RUN_ID`. Completed immutable batches are validated and skipped. A missing batch is regenerated from independent episodes. If a registered artifact is missing or altered, the notebook stops rather than silently reusing it. Do not run two notebook instances against the same Drive folder.

Use a **new `RUN_ID`** after changing code, Python version, seed, episode counts, or batch size. Code-object fingerprints are part of the manifest identity. In Colab, edit the setup cell directly if needed; the `NFW007_*` environment variables are primarily for local automated tests.

## Expected checks

```text
status: complete
security_controls.all_passed: true
endpoints.unauthorized_side_effects: 0
endpoints.unexpected_allowed_requests: 0
endpoints.benign_task_success_rate: 1.0
endpoints.audit_chains_valid: true
endpoints.benign_audit_chains_valid: true
```

`authorized_in_scope_reads` should be nonzero: the deliberately disclosed read token may permit a valid read. This is not a bypass. `synthetic_alert_precision`, `synthetic_alert_recall`, and `benign_episode_false_alert_rate` are only results for this scripted workload. A nonzero false-alert rate is expected because some benign sessions intentionally make two mistakes.
