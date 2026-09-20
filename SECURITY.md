# Security Policy

This repository contains experimental AI-safety research code. It is not a
deployed security product and should not be used as the sole safety layer for a
production system.

## Reporting Issues

If you find a vulnerability, unsafe failure mode, data leak, or reproducibility
problem, please open an issue with:

- the affected experiment or file path;
- the model, dataset, and run configuration if known;
- steps to reproduce;
- expected and observed behavior; and
- any relevant artifact hashes or logs.

For sensitive reports, avoid posting private data, credentials, gated dataset
contents, or exploit details that would materially increase misuse risk.

## Claim Boundary

Current experiments may study harmful-request detection, blocking, activation
monitoring, and intervention. These are research artifacts. They do not establish
general model safety, robust jailbreak resistance, or neural privilege
separation unless a specific experiment explicitly validates those claims under
a stated threat model.
