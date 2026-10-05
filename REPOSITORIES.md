# Separate firewall workstreams

The active security work is split into two independent local Git repositories:

- [`separate-repositories/permission-boundary`](separate-repositories/permission-boundary) contains host-controlled document read permissions and the limited document-Q&A pilot. Its claim is denied-source noninterference under documented runtime assumptions.
- [`separate-repositories/neural-runtime-monitor`](separate-repositories/neural-runtime-monitor) contains activation-trajectory monitoring, calibration and evaluation. It remains experimental and is not approved to block user responses.

The local parent repository ignores these directories so each child keeps an independent history. They are not Git submodules, and GitHub cannot expose them through this parent until each child has a remote and the parent records submodule URLs. Push/create permissions are currently unavailable because the configured GitHub CLI credential is invalid. Existing source and research artifacts in the parent are retained as the historical integration baseline.
