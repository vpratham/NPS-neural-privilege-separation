# Host-enforced document read-permission paper

This package is a submission-preparation draft for a narrow systems/measurement paper. Its supported claim is conditional denied-document noninterference in the pinned Qwen and SmolLM2 development configurations described in the manuscript. It does not claim that the system prevents prompt injection in readable evidence.

## Build

```sh
cd docs/neural_state_firewall_paper
tectonic manuscript.tex
```

The package was compiled locally with Tectonic 0.17.0. `REPRODUCIBILITY.md` lists source-level checks and the pinned evaluation inputs.

## Evidence

The raw 175-case result is local and ignored because it contains generated model outputs:

```text
neural_state_firewall/artifacts/bipia_read_permission_20261002.json
```

It records BIPIA revision `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`, model revision `7ae557604adf67be50417f59c2c2f167def9a775`, case hashes, outputs, and all-step comparisons. The paper reports 175/175 denied-payload invariance checks, 175/175 output-equivalent clean host-filtering comparisons, and 42/175 ordinary readable-attack output changes. The latter is descriptive only, never attack success.

The BIPIA source revision is consumed development material in `data/seen_material_exclusions.json`. Do not relabel it as a locked test.

## Package map

- `manuscript.tex` — article source with author/contact placeholders.
- `references.bib` — cited primary literature.
- `STUDY_PROTOCOL.md` — retrospective protocol v2 and deviations.
- `REPRODUCIBILITY.md` — exact rerun and verification guide.
- `SUBMISSION_CHECKLIST.md` — tasks that must be complete before submission.
- `cover_letter.txt` — editable draft with required placeholders.
- `SOFTWAREX_METADATA.md` — working metadata template for the primary venue target.

The author is Pratham Vasa. Affiliation, email, ORCID, funding, conflict, ethics/data-availability, and AI-use disclosures remain explicit placeholders. Do not submit until completed.
