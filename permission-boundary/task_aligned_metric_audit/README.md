# NPB prompt-injection metric conflation audit

This is a bounded, no-inference reanalysis of saved benchmark records. It asks whether an original score can hide meaningfully different observed behaviors: source-opinion attribution (A), false-claim acceptance (B), and unauthorized behavioral redirection (C).

The answer is **INCONCLUSIVE**. The selected records establish some bounded C labels, but no selected record supports all three labels. TAB's complete released score vector already contains a separate `injection.executed` field, so a scalar reward collision is not a demonstrated measurement failure. The local NPB records are explicitly non-official BIPIA exact/substring diagnostics, not attack-success scores.

## Scope

**VERIFIED FACT:** This directory contains 35 predeclared record units: all eight saved TAB sample trajectories at commit `ec95937fc54337e05fdffa707516f32c2b6ac243`, plus 27 saved local NPB outputs (three predeclared email cases × three conditions × three arms). No model was invoked, trained, or downloaded.

**SUPPORTED INFERENCE:** Existing raw records may support a narrow annotation/scoring extension if independent truth and attribution labels are recovered.

**UNVERIFIED CLAIM:** Existing benchmark rankings, defenses, or general prompt-injection metrics are materially wrong because of the observed sample.

**OPEN QUESTION:** Whether an independently truth-annotated, task-matched slice produces complete A/B/C labels and a consequential score collision.

Read [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md), [ANNOTATION_PROTOCOL.md](ANNOTATION_PROTOCOL.md), [METRIC_CONFLATION_ANALYSIS.md](METRIC_CONFLATION_ANALYSIS.md), and [FINAL_DECISION.md](FINAL_DECISION.md). Run `python3 reproduce.py` to verify every saved-source hash, evidence quotation, selection identifier, and reported descriptive count.
