# Reproducibility

## Environment and inputs

- Repository revision at preservation: `95c4040bb2a37630ef5de111d2c9aca40c602a7d`.
- TAB source revision: `ec95937fc54337e05fdffa707516f32c2b6ac243`; URLs, retrieval time, byte counts and SHA-256 values are in `evidence/tab/retrievals.json`.
- Local NPB result: `neural_state_firewall/artifacts/bipia_read_permission_20261002.json`, SHA-256 `c300803ae666c1dfa83e5cc0e2e9cc70929910a718126e277f466ab0fcc6693a`.
- All local input file hashes are in `evidence/local_source_manifest.json`.
- Frozen protocol/sample hashes are in `evidence/protocol_freeze.json`. Do not modify the protocol or selection manifests after inspection.

## Commands

Run from repository root:

```sh
python3 permission-boundary/task_aligned_metric_audit/reproduce.py
shasum -a 256 permission-boundary/task_aligned_metric_audit/*
git rev-parse HEAD
```

The first command is standard-library only. It performs no network access, does not invoke a model, and writes no files. It verifies frozen protocol files, all saved source hashes, every quoted evidence span, selected identifiers, local source/request/reference/document construction, score joins, and the reported counts. Its captured output is `analysis.json`.

## Missing evidence and assumptions

- No human semantic review was performed. Pending semantic labels stay `INDETERMINATE`.
- No assertion is made that retained TAB verifier code exactly generated historical results.
- No external AlignSentinel/CBC artifacts were authenticated in this audit.
- The local NPB run is not an official BIPIA evaluation and its reference-match fields are not attack-success measurements.
- No weights, evaluation agents, or training jobs were run.

Final artifact checksums are recorded in `SHA256SUMS`; preservation and final verification details are recorded in `verification.json`.
