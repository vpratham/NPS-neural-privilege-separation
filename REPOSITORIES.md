# Firewall workstreams

The active firewall work lives in two separate folders in this NPS repository:

- [`permission-boundary/`](permission-boundary/) contains host-controlled document read permissions and the limited document-Q&A pilot. Its claim is denied-source noninterference under documented runtime assumptions. It does not stop instructions in evidence the model is allowed to read.
- [`neural-runtime-monitor/`](neural-runtime-monitor/) contains activation-trajectory monitoring, calibration and evaluation. It remains experimental and is not approved to block user responses.

Both folders are tracked by this repository's single Git history. They are separate workstreams, not nested Git repositories. The original top-level sources and research artifacts remain as the historical integration baseline.
