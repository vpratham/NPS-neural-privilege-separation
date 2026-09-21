# NFW-005 — Reproducible adversarial capability-firewall evaluation

**Notebook:** `NFW_005_Reproducible_Adversarial_Capability_Firewall.ipynb`  
**Guide:** `COLAB_GUIDE.md`

NFW-005 stress-tests the NFW-004 veto-gated capability boundary. It is deliberately a Colab notebook because the optional Qwen activation smoke test needs a hosted GPU; the broker, wire parser, fuzz suite, audit-chain checks and security invariants are CPU-only.

The model remains an untrusted proposer. The external broker is default-deny and accepts authority only from trusted host-side capability tokens. Model claims, signatures, administrator fields, approvals and tool names do not grant authority. A neural attestation can deny or require approval, but cannot grant or upgrade privilege.

## Coverage

- Strict JSON wire parser: duplicate keys, non-finite JSON, size limits, Unicode, malformed/truncated input and unknown fields.
- HMAC-scoped expiring capability tokens, key epochs, explicit revocation, replay/nonces and constant-time signature checks.
- Tamper-evident chained audit events.
- Deterministic randomized adversarial requests with Drive checkpoints every batch.
- Optional forked broker smoke test; this is not a production sandbox.
- Optional pinned Qwen activation smoke test; norms are observation-only, not safety scores.
- Immutable manifest, package/model identity, seed, stage checksums and report artifacts.

No real shell, network, filesystem, API or external tool is called. Accepted requests are authorization decisions only.

## Scientific claim

The notebook can support a narrow claim about the tested broker invariants. It cannot establish universal harmful-output prevention, policy isolation inside model weights, adaptive robustness, secure production cryptography, or novelty priority. A production system still requires a separately isolated broker process/service, protected key custody, authenticated principals, resource authorization, TOCTOU defenses and a security review.
