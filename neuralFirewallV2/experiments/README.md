# Neural Firewall experiments

Each NFW experiment should answer one falsifiable question, use a frozen configuration, and write a manifest with model revision, tokenizer/template, activation site, data IDs, split IDs, code revision, seeds, and artifact hashes.

| Series | Purpose | Status |
|---|---|---|
| NFW-00 | Legacy artifact integration and reproducibility | Historical artifacts available |
| NFW-01 | First runtime firewall prototype | Historical; known limitations documented in the audit |
| [NFW-02](NFW-02_foundation_firewall/) | Target-model monitor-and-block baseline | Current foundational experiment |
| NFW-03 | Policy-isolation measurements | Planned after NFW-02 |
| NFW-04 | Policy-state invariance | Planned |
| NFW-05 | Adaptive attacks | Planned |
| NFW-06–09 | Hardening, combined defense, cross-model evaluation, final benchmark | Planned |

Do not use the final evaluation partition to choose layer, threshold, persistence, intervention strength, or judge. Every expected example needs one terminal status: `ok`, `blocked`, or `error`; errors remain in the report.
