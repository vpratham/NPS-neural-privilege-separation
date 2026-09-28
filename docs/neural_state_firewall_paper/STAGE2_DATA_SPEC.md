# Stage 2: evaluation-data specification

**Status:** planning specification, 2026-09-28. No Stage 2 evaluation cases have been collected. The manifest [`data/seen_material_exclusions.json`](data/seen_material_exclusions.json) lists material that cannot be presented as a fresh locked test.

## Study unit and scope

The unit is one response-only retrieval request: a trusted user task, a supplied retrieved document/context, and the model's response. This is not an end-to-end retriever test. The claim under test is limited to whether untrusted retrieved text can alter the generated response and whether the response monitor changes what is released.

Each evaluation item has a semantically matched benign and injection version from the same source/task family. Include benign documents containing security discussion or quoted attack language so topic terms alone do not identify the attack condition. Include ordinary novel benign documents and diverse attack constructions. Do not use attack-marker tokens as the only attack type. Protected-value cases use synthetic canaries only.

## Data sources, provenance, and custody

For every source document and derived case, record:

- source identifier, owner/publisher, canonical location, acquisition date, and permitted research/release use;
- source-document digest, transformation history, annotator/author, and source-group ID;
- task family, condition, matched-pair ID, attack family/template ID, and assigned split;
- expected-answer and protected-value references in an evaluator-only label file, separate from model inputs and inaccessible to model/profile developers for the locked test.

Do not use a source unless its license, terms, privacy status, and redistribution plan are reviewed. Keep a private custody copy of locked-test documents and labels under evaluator control. Developers receive only the development partitions until the code, model revision, profile, thresholds, exclusions, and analysis are frozen and hashed.

### Candidate auxiliary corpus: NeurAlchemy Prompt Injection Dataset

The [Hugging Face dataset card](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset) describes a binary prompt-text classification corpus with `text`, malicious/benign `label`, `category`, `source`, `severity`, `group_id`, augmentation flag, and tags. It reports `core` train/validation/test counts of 4,391/941/942 and a group-aware split; the current card revision shown on the Hub is `7d70432`. The viewer examples include both isolated direct jailbreak prompts and some embedded-document examples.

**Recommendation:** use only as an optional, pinned development/training resource for attack-family coverage or a separate text-only detector baseline. Do not use it as the primary response-firewall efficacy set: it contains neither trusted task + retrieved-context pairs nor generated model outputs and behavioral success labels. Its malicious label is a text-class label, not proof that an instruction took authority in a model response. Converting examples into retrieval requests would create a derived dataset requiring fresh task/context construction and outcome labeling.

**Built baseline (2026-09-28):** `neural_state_firewall.text_baseline` trains a TF-IDF word/bigram + class-balanced logistic-regression classifier on only the pinned `core` train split, validates at threshold 0.5, and records input hashes in its local report. It verifies group and normalized-text separation before fitting. On 4,391 train and 941 validation rows, it achieved accuracy 0.946, precision 0.930, recall 0.978, F1 0.953, and benign false-positive rate 0.096 (39/407 benign rows). These are text-label classification results on the dataset's public validation split, not prompt-injection success or firewall efficacy. The six Stage 1 context scores are marked development-only and demonstrate topic/lexical false positives; they must not be used to set the neural monitor threshold or claim detection. Reproduce from the repository root with `python3 -m pip install -r neural_state_firewall/baseline-requirements.txt` and `python3 -m neural_state_firewall.text_baseline`; the script does not load `core/test`.

Do not use the dataset’s public test split as our locked test. If any rows are imported into training or development, exclude the full public repository snapshot, its `group_id` families, derived templates, and near-duplicates from locked and adaptive evaluation. The card labels the dataset Apache-2.0, while its source table lists components with differing terms, including a “Research” entry; review component-level provenance and permissions before importing or redistributing data. The card’s leakage claim is author-reported and should be verified against the pinned files before any experimental use.

## Split rules

Assign whole source-document groups and attack-template families to exactly one partition. Related passages, paraphrases, tasks, benign controls, or translations stay together. Do not split matched items across partitions.

| Partition | Allowed material | Use |
|---|---|---|
| Training | New benign trusted-context requests only | Fit the monitor's temporal predictor |
| Calibration | Disjoint benign trusted-context requests only | Set the threshold/profile |
| Development | Benign and attack cases from groups excluded from all final sets | Choose workload details, comparators, and analysis implementation |
| Locked test | Fresh source groups and attack templates, with matched benign controls | One frozen efficacy evaluation |
| Adaptive test | Separately held fresh groups; attacker knows the defense and has a fixed budget | Separate robustness evaluation after defense freeze |

All repo materials in the exclusion manifest, their source documents, and their near-duplicates are assigned `seen-development` and barred from locked-test and adaptive-test use. The `split: test` field in the six-case example is a software-format label only; the study protocol supersedes it and classifies those cases as seen development material.

## Labels and analysis inputs

Store model inputs separately from evaluator metadata. Review task success and injection success as separate outcomes; a blocked response is a system disposition, not evidence of model resistance. Record protected disclosure, benign block, task utility, and latency separately. For the final semantic evaluation, use two independent blinded reviewers when available and third-review adjudication for disagreements. If only one reviewer is available, freeze that limitation before opening the locked test; report no inter-rater agreement and do not substitute an LLM judge as independent human review.

## Sample size and collection gates

Do not pick a locked-test size from convenience or the prototype bootstrap minimum. First assemble a development-only variance/cluster pilot, then calculate the needed number of independent source/template groups for the prespecified precision or power target under paired analysis. Treat five source groups as a software warning floor only, not adequate evidence. Predeclare exclusions, missing/ambiguous handling, cluster method, and all efficacy/utility/latency thresholds before locked-test custody is released.

No external dataset is approved by this specification. Dataset selection requires source-specific license/terms review and a relevance check for response-only retrieved-context behavior. No case text should be authored into or exposed from a locked test while implementation is still being tuned.

## Exit criteria for Stage 2

1. A versioned source register records rights, provenance, hashes, and independent source/template groups.
2. A machine-checkable split manifest proves group-level separation and excludes every entry in `seen_material_exclusions.json` plus near-duplicates.
3. A separate evaluator-only label key defines task references, injection outcomes, and canaries.
4. A source-cluster-aware sample-size analysis, including stated assumptions and sensitivity bounds, is frozen.
5. A locked-test custodian and process prevent developers from inspecting test cases or labels before the implementation freeze.
