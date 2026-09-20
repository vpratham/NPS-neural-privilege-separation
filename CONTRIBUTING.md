# Contributing

Thanks for helping improve Neural Privilege Separation. This repository is a
research workspace, so the main contribution standard is evidence hygiene:
claims should be traceable to experiments, artifacts, or clearly labeled theory.

## Contribution Priorities

- Reproducible experiments with frozen configs, seeds, splits, and manifests.
- Clear evaluation code for target-model outputs, not only detector scores.
- Documentation that separates established results from hypotheses.
- Small, reviewable changes to experiment notebooks and supporting utilities.
- Audits that identify confounds, failure modes, and missing controls.

## Evidence Standards

When adding an experiment or result, include:

- model ID and revision;
- dataset source, split policy, and hashes where practical;
- prompt serialization / chat template details;
- activation layer and pooling conventions;
- detector or controller artifact provenance;
- calibration procedure and threshold policy;
- final evaluation denominator, failures, and limitations.

Avoid claiming security, robustness, privilege separation, or generalization
unless the experiment directly tests that property.

## Artifact Policy

Keep small metadata, manifests, configs, and metric summaries in Git. Store large
checkpoints, model weights, activation dumps, and bulky generated outputs in a
release or external artifact store with checksums.

## Research Tone

Prefer precise claims:

- "This probe is predictive on this held-out split."
- "This monitor reduced judged harmful assistance on this run."
- "This intervention changed behavior under this controlled condition."

Avoid overbroad claims:

- "The model is safe."
- "The firewall is robust."
- "The policy state is protected."

Those stronger statements require explicit threat models, adaptive evaluation,
capability-retention evidence, and independent validation.
