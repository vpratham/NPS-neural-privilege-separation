# Firewall workstreams

The active firewall work lives in two separate folders in this NPS repository:

- [`permission-boundary/`](permission-boundary/) contains host-controlled document read permissions, a limited document-Q&A pilot, and a provider-independent per-instance capability gate that separates reading, disclosure and action proposals. It does not stop instructions in evidence the model is allowed to read; action proposals still require the trusted broker.
- [`neural-runtime-monitor/`](neural-runtime-monitor/) contains experimental activation-trajectory monitoring and an earlier capability-gate prototype. The anomaly scores remain observation-only and do not authorize effects.

Both folders are tracked by this repository's single Git history. They are separate workstreams, not nested Git repositories. The original top-level sources and research artifacts remain as the historical integration baseline.
