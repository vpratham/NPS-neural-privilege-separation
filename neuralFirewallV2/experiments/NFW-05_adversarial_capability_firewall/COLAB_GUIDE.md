# NFW-005 Colab guide

1. Open `NFW_005_Reproducible_Adversarial_Capability_Firewall.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → GPU**.
3. Run all cells from the top.
4. Authorize Drive mounting.
5. Keep the defaults for the first run:

```python
RUN_ID = 'nfw005_adversarial_broker_001'
RUN_GPU_SMOKE = True
RUN_SUBPROCESS_SMOKE = True
REVIEW_ONLY = False
N_RANDOM_CASES = 2000
```

Artifacts are written to:

```text
MyDrive/NFW-005/nfw005_adversarial_broker_001/
```

The broker/fuzz evaluation does not require an LLM. `RUN_GPU_SMOKE=True` additionally loads the pinned Qwen revision on the Colab GPU and records three activation norms. Those values are observation-only and never authorize actions.

## Reconnect

Use the same notebook, `RUN_ID`, configuration, Python/package environment and model revision. Run all cells again. The manifest and stage envelopes refuse changed identities. Fuzz outcomes are checkpointed in batches; prior outcomes are retained when a disconnect occurs. GPU hardware is observational provenance, so a different assigned GPU does not automatically invalidate the run, although bitwise equality is not guaranteed.

Do not run two writers against one Drive folder. If an artifact is missing or corrupt, restore a backup or use a new run ID rather than bypassing checks.

For a CPU-only broker run, set:

```python
RUN_GPU_SMOKE = False
RUN_SUBPROCESS_SMOKE = False
```

For report-only reopening of a completed run:

```python
REVIEW_ONLY = True
RUN_GPU_SMOKE = False
RUN_SUBPROCESS_SMOKE = False
```

Keep all identity/configuration values unchanged.

## Read the report

`REPORT.md` and `final_report.json` must show:

- `status: complete`
- `security_results.all_passed: true`
- `fuzz_summary.allowed_without_trusted_token: 0`
- `security_results.audit_chain_valid: true`
- `security_results.external_actions_executed: 0`
- model capability minting/upgrading: `false`

A valid request with a trusted host-issued capability may be authorized. This is intentional. The security property is that model output alone cannot obtain that capability.

## Limitations

The optional multiprocessing smoke test demonstrates a separate broker worker boundary inside the Colab VM, not a hardened sandbox. Production deployment must separate the broker service, protect signing keys, authenticate principals, enforce resource-level policy and defend against time-of-check/time-of-use races. The deterministic fuzz set is broad but is not formal verification or an adaptive attacker benchmark. No external action is executed by this notebook.
