# Robustness analysis

## What the slice can test

The 27 local units cover three email cases and source positions start, middle, and end; each case is evaluated in clean, readable-attack, and denied-attack conditions for three arms. The three arms generated identical outputs within each case, so the selected slice contains no observed arm ranking difference. Its effective task coverage is three cases, not 27 independent task draws.

The eight TAB records span eight different software tasks. They have one trajectory per task, and therefore cannot compare defenses or expose a same-task defense ranking. Five retain historical score objects; two retain empty result objects; one retains no result object. Four labels have unresolved C status and all eight have unresolved A/B status.

## Sensitivity checks

| Factor | Check | Result | Interpretation |
| --- | --- | --- | --- |
| Annotation ambiguity | Exclude all `INDETERMINATE` | 22 C-applicable records remain, but no A/B-applicable record | no complete three-way inference remains possible |
| Task family | Local units are email only; TAB units are distinct software tasks | no within-family replication | generalization is **UNVERIFIED** |
| Attack family | local Task Automation only; TAB task-plausible distractors | no matched cross-family estimate | generalization is **UNVERIFIED** |
| Source position | one local case at each of start/middle/end | all three arms remain output-identical per case | position is confounded with case/content; no causal position result |
| Original scoring | local scores are false for clean and attacked outputs; TAB vector contains execution field | no complete-score C collision in comparable pairs | no demonstrated score-conflation effect |
| Ground truth | no independent false-claim reference or human attribution review | A/B remain unavailable or pending | semantic conclusion is not estimable |

No population confidence interval is reported: the sample is a frozen convenience/release census, not a random sample. Zero identified conflating pairs is an observed count, not evidence that the population rate is zero.

**Strongest alternative explanation:** the observed differences reflect expected coarse task-utility scoring and incomplete semantic ground truth, rather than a consequential benchmark defect. Local clean-control reference mismatches make it especially unsafe to interpret wrong answers as injection outcomes.
