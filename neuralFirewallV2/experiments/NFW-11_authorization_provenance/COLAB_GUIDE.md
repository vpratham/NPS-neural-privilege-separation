# Run NFW-011 in Google Colab

NFW-011 is a **CPU-only, model-free** replay. It does not load Transformers, download model weights, call an API, or require a GPU.

## Prepare Drive

1. Upload/open `NFW_011_Authorization_Provenance_Replay.ipynb` in Colab.
2. Copy the NFW-010 archive `nfw-10-results.zip` into the Drive account you will use for this run. For example:

   ```text
   MyDrive/NFW-010/nfw-10-results.zip
   ```

   You do not need access to the original Drive account if you have a copy of the archive. The archive should contain run `nfw010_protocol_corrected_003`.

3. Run all notebook cells in order. In **Configuration**, change `INPUT_PATH` if the archive is stored elsewhere. The notebook mounts Drive and writes outputs under:

   ```text
   MyDrive/NFW-011/nfw011_authorization_provenance_001/
   ```

4. Review `REPORT.md` and `evaluation.json` in that output directory.

If you already extracted the results, set `INPUT_PATH` to the extracted `nfw-10-results` directory instead of the ZIP file.

## What the run verifies

The notebook safely extracts only the expected NFW-010 run subtree, validates immutable payload checksums, confirms Qwen 3B passed the development gate, cross-checks all 72 held-out response hashes against NFW-010's evaluation records, then replays the frozen proposals through scope-only, source-bound, and oracle-bound mock authorization. It fails closed if artifacts are missing or inconsistent.

No model generation occurs. The source-bound arm uses a host-verified public-record receipt and does not give its issuer the expected-answer label. The oracle arm intentionally uses that answer as an upper bound. Since these authored tasks are exact-copy tasks, the source fact equals the expected content; do not generalize this result to arbitrary task authorization.
