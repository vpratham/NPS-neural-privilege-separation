# NFW-007 — Integrated boundary, utility, and telemetry evaluation

**Notebook:** `NFW_007_Integrated_Boundary_Utility_Telemetry.ipynb`  
**Execution:** Google Colab CPU runtime; no model download or previous Drive data required.

NFW-007 moves beyond the NFW-006 all-denied attack count. It measures four distinct properties in one reproducible run:

1. **Authority boundary:** strict request parser, host-issued scoped capabilities, expiry, revocation, replay, signed-token tampering, subject/resource confusion, veto, and approval.
2. **Effects and utility:** authorized read/write/message controls and denied attempted changes to local temporary mock-tool files. The attacker receives a valid read capability, so an allowed in-scope read is separated from an unauthorized allowance.
3. **Adaptive attacks:** eight feedback-dependent requests per independent episode; saved as immutable Drive batches that can be resumed after Colab disconnects without resetting replay state within a partially completed episode.
4. **Observability:** in-run HMAC audit-chain verification, chained JSONL event export, simple incident correlation, benign-session false-alert checks, including recoverable operator mistakes, and episode-level confidence intervals.

Default run: 400 attack episodes (3,200 attack requests) and 100 benign episodes (300 legitimate tasks plus scripted benign mistakes). The principal security endpoint is **unauthorized local mock-tool side effects**. The report also gives legitimate-task success, benign false-alert rate, and synthetic alert precision/recall so a deny-everything system cannot appear successful.

## Scientific boundaries

- The attacker is a deterministic agent-request simulator, not a sampled LLM or an independent red team.
- Temporary local files are *mock tools*, not an OS/container/VM sandbox or production data.
- The broker and mock tools run in the same Colab Python process. A model capable of arbitrary Python execution would bypass this reference architecture.
- Capability signing keys are generated in memory and are not exported. Audit chains are checked during the run, but exported tags cannot be independently reverified later without separate key custody.
- Synthetic alert precision/recall does not demonstrate real-world IDS/SIEM performance.
- No cross-model or neural-detector claim is established here.
- The reported zero-event binomial upper bound is illustrative under an iid-episode assumption; scripted episodes and shared attack logic limit that assumption. It is not a bound over all attacks.

The paper-ready claim is therefore about **this tested broker interface and mock-tool workload**, not a universal neural firewall.
