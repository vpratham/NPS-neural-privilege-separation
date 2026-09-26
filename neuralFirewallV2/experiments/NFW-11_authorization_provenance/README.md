# NFW-011 — Authorization provenance replay

NFW-011 addresses the narrowest important follow-up to NFW-010: **does argument binding only work because the experiment's host already knows the expected answer, or can the host bind permission to an independently trusted source record?**

**Colab notebook:** [`NFW_011_Authorization_Provenance_Replay.ipynb`](NFW_011_Authorization_Provenance_Replay.ipynb) · **Run guide:** [`COLAB_GUIDE.md`](COLAB_GUIDE.md). The notebook is self-contained and CPU-only. The local script below is a convenience entrypoint for repository users.

This is a CPU-only replay of the frozen Qwen 3B held-out outputs in `../NFW-10_protocol_corrected_injection/nfw-10-results/`. It performs no model loading, generation, network access, or real tool effects. The three modes are:

1. **Scope-only:** permit `write_record(notes, *)` without a content constraint.
2. **Source-bound:** derive the argument hash from a host-verified public-record receipt. The issuer API accepts only the record identity/value, not the task object, expected answer, attacker target, injection condition, or model output.
3. **Oracle-bound:** bind to the task's expected arguments. This is an intentionally privileged upper-bound control, not a deployable authorization strategy.

The question is deliberately narrow. NFW-010 tasks ask the model to write the exact public record returned by a trusted host tool. That makes the source receipt a plausible permission source in this fixture. It does **not** solve authorization for arbitrary natural-language tasks, ambiguous user requests, transformed or summarized data, multiple-source synthesis, or actions where policy constraints are semantic rather than exact-copy.

## Run

From the repository root:

```bash
python3 neuralFirewallV2/experiments/NFW-11_authorization_provenance/run_authorization_provenance.py
python3 -m unittest neuralFirewallV2.tests.unit.test_capability_broker -v
```

The script verifies NFW-010 immutable-envelope payload hashes and the Qwen 3B development gate before replay. Outputs are written to `nfw011_authorization_provenance_001/REPORT.md` and `evaluation.json` alongside this guide. The run is deterministic because it reuses existing model outputs; capability nonces and signatures are not reported as study outcomes.

## What this prototype is and is not

`src/policy/capability_broker.py` is a reusable CPU-testable reference for strict proposal parsing, signed one-use capabilities, scope checks, source/oracle argument binding, expiration, replay denial, and atomic in-memory mock writes. It is **not production-ready**: its key is process memory, replay state is not durable across restart, `TrustedRecord` must be created only after real upstream authentication, and execution stays in the same process. The mock workspace is the only effect sink.

## Interpretation safeguards

- The primary comparison is source-bound versus scope-only; oracle-bound is only an upper bound.
- The source-bound arm is not independent of the authored task construction: expected content equals the public source fact by design.
- The same Qwen 3B outputs and task set are reused. This is not an independent replication, new attack set, or cross-model study.
- Broker effect prevention does not imply model alignment, injection resistance, neural policy isolation, or general-purpose content safety.
- A production design still needs authenticated principals and source adapters, policy derivation for non-copy tasks, protected key custody, durable atomic replay/revocation state, isolated executors, deployment observability, and independent security review.
