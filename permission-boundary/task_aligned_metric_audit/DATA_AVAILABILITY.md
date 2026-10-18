# Data availability

## Inspected records

| Source | Units selected | Saved responses | Linked original scores | Available fields | Limitation |
| --- | ---: | ---: | ---: | --- | --- |
| TAB author sample, pinned `ec95937` | 8 trajectories | 7 | 5 | trusted task, trajectory, selected source observations, historical reward/result where retained, `task_completed` and `injection` verifier fields where retained | `fix-code-vulnerability` contains no agent response; no independent truth/original opinion oracle; one trajectory lacks a visible directive exposure; exact historical scoring-code correspondence is unverified |
| NPB-local BIPIA-derived run | 27 responses | 27 | 27 | trusted policy/request reconstructed from frozen inputs, source/attack content, answer key, output, `reference_exact`, `reference_contained`, arm/condition/position | not official BIPIA; scores are explicitly exact/substring diagnostics; no independent false-source-claim treatment |
| AlignSentinel/CBC | 0 | 0 | 0 | none authenticated for this audit | **UNAVAILABLE** in this audit; no inference follows |

## Field status

**OBSERVED:** the saved TAB trajectories/verifier objects, local output records, original score fields, hashes, and the original NPB run's model metadata. Exact pointers and quotation spans are in [ANNOTATED_RECORDS.jsonl](ANNOTATED_RECORDS.jsonl).

**DERIVED:** local trusted task/source reconstruction, source-document hashes, selected-case joins, output equality, label counts, and pair counts. The rules are frozen in [ANNOTATION_PROTOCOL.md](ANNOTATION_PROTOCOL.md) and checked by `reproduce.py`.

**UNVERIFIED:** that the retained TAB verifier source exactly produced each historical result; semantic attribution judgments; source-claim falsity for the selected records; historical model intent; and any model-internal causal explanation.

**UNAVAILABLE:** independent truth labels for a designated false source claim in the selected records; a completed human semantic review; a matched multi-defense external response set; authenticated AlignSentinel/CBC response-and-scorer artifacts.

## Preserved evidence

The pre-audit snapshot in `evidence/preservation_before.json` protected 553 prior files and verified 389 entries from prior checksum manifests against repository revision `95c4040bb2a37630ef5de111d2c9aca40c602a7d`. This audit only adds files beneath this directory. The frozen sample manifests are `evidence/sample_plan.json` and `evidence/secondary_sample_plan.json`; their protocol hashes are in `evidence/protocol_freeze.json`.
