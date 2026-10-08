# Separate firewall workstreams

The active security work is split into two independent GitHub repositories:

- [NPS Permission Boundary](https://github.com/vpratham/nps-permission-boundary) contains host-controlled document read permissions and the limited document-Q&A pilot. Its claim is denied-source noninterference under documented runtime assumptions.
- [Neural Runtime Monitor](https://github.com/vpratham/neural-runtime-monitor) contains activation-trajectory monitoring, calibration and evaluation. It remains experimental and is not approved to block user responses.

The parent repository ignores its local staging copies so each child keeps an independent history. Existing source and research artifacts in the parent are retained as the historical integration baseline.
