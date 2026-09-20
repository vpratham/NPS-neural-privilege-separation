# NFW-004 — Veto-Gated Neural Privilege Separation

**Notebook:** `NFW_004_Veto_Gated_Neural_Privilege_Separation.ipynb`  
**Run guide:** `COLAB_GUIDE.md`

This is the architectural pivot following the NFW-002 and NFW-003 results. Prompt-level activation scores are useful as a risk signal but do not provide a reliable security boundary. NFW-004 places authority in a host-side capability broker and gives the neural monitor a **veto-only** role.

The model can propose an action. It cannot mint a capability, claim an administrator role, add authority fields, or upgrade a denied request. The broker is default-deny, validates typed schemas, verifies scoped expiring HMAC-signed capability tokens, rejects replays, and records decisions. The notebook executes no real tools, shell, network, filesystem, or external APIs.

The optional GPU cell loads the pinned Qwen revision on Colab and records activation norms for three non-executing proposals. Those norms are deliberately labelled observation-only; they are not a trained safety detector. The broker remains secure when the neural cell is disabled, unavailable, or wrong.

## Research claim

The defensible claim is:

> A veto-only neural attestation signal can be coupled to an external capability boundary so that model-generated content cannot directly acquire authority.

This is a capability-boundary PoC, not proof of universal harmful-output prevention, policy isolation inside model weights, adaptive robustness, causal intervention, secure production cryptography, or cross-model transfer. Novelty must be checked against current literature separately.

## Artifacts

```text
MyDrive/NFW-004/<RUN_ID>/
  manifest.json
  security_tests.json
  integration_demo.json
  gpu_attestation_demo.json       # only if GPU cell is enabled
  final_report.json
  REPORT.md
```

Artifacts are Drive-backed and checksummed. A changed implementation/configuration/model/package environment refuses reuse; choose a new `RUN_ID`. GPU identity is recorded as observational provenance and does not invalidate a reconnect, but mixed hardware is not guaranteed bitwise identical.
