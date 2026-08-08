# Experiment 001

Date:
2026-06-14

Question:
Can policy information be decoded from hidden states?

Model:
Qwen 2.5 1.5B Instruct

Dataset:
4800 generated prompts

Method:
- Loaded Qwen 2.5 1.5B on a Tesla T4 GPU.
- Extracted hidden states for every prompt.
- Collected all 29 hidden-state tensors (embedding layer + transformer layers).
- Stored activations in float16 format.
- Serialized activations using pickle.

Output:
- activations.pkl
- Size: 3.12 GB
- Prompts processed: 4800
- Layers extracted: 29
- Hidden dimension: 1536

Result:
Successful activation extraction. Full hidden-state dataset generated for downstream probing experiments.

Notes:
- Experiment focused on activation extraction only.
- No probing, classification, neuron ranking, or intervention experiments performed yet.
- Activation dataset backed up separately and excluded from Git tracking.

Next Steps:
Experiment 002 - Layer Probing and Policy Signal Discovery

---

# Experiment 002

Date:
TBD

Question:
Can policy information be linearly decoded from hidden states?

Model:
Qwen 2.5 1.5B Instruct

Dataset:
4800 prompts with Safe/Unsafe labels

Method:
- Load activation dataset generated in Experiment 001.
- Construct binary labels (Safe = 0, Unsafe = 1).
- Mean-pool token activations for each prompt.
- Train linear probes (Logistic Regression).
- Evaluate every transformer layer independently.
- Measure Accuracy, Precision, Recall, and ROC-AUC.
- Identify layers containing the strongest policy signal.

Planned Outputs:
- probe_results.csv
- layer_auc.png
- best_probe.pkl

Success Criteria:
- Hidden-state probes perform significantly above chance.
- Policy information is recoverable from activations alone.
- Peak-information layers are identified.

Result:

* Activation dataset from Experiment 001 was successfully loaded and transformed into layer-wise feature matrices using mean-pooled hidden-state activations.
* Logistic Regression probes were trained independently on all 29 transformer layers.
* Probes achieved near-perfect classification performance (ROC-AUC ≈ 1.0) across nearly all layers.
* Initial interpretation suggested that policy labels were highly recoverable from hidden states.
* Follow-up diagnostics were performed to investigate the unexpectedly strong performance.
* Capability and intent distributions were balanced across policy classes and did not explain the result.
* Topic analysis revealed complete separation between allowed and restricted examples:
    * Shared topics between classes: 0
    * Allowed-only topics: 80
    * Restricted-only topics: 80
* The probe could therefore classify examples using topic identity rather than policy information.

Conclusion:
Experiment 002 successfully validated the activation extraction and probing pipeline. However, the dataset contained a major confound: policy labels were perfectly correlated with topic categories. Because there was no topic overlap between allowed and restricted examples, probe performance cannot be interpreted as evidence of an isolated policy representation. The observed results demonstrate that semantic topic information is strongly encoded in hidden states and is sufficient to recover policy labels in the current dataset.

The central NPS hypothesis remains unresolved. To test whether policy information exists independently of topic, a new dataset is required in which topic is held constant while policy varies. Future experiments should use paired prompts within the same domain (e.g., allowed and restricted cybersecurity requests) to isolate policy-related representations from semantic content.


Experiment 003 — BeaverTails Category-Held-Out Safety Probe

Objective

Test whether policy/safety representations generalize across unseen harm domains.

Unlike Experiments 001 and 002, which used random train/test splits, Experiment 003 evaluates whether a probe trained on some harm categories can generalize to entirely unseen categories.

This is a stronger test of the Neural Privilege Separation hypothesis because it reduces the possibility that the probe is merely learning topic information.

⸻

Dataset

Source:

* BeaverTails (PKU-Alignment)

Construction:

* 14 harm categories
* 150 unsafe prompts per category
* 150 safe prompts per category

Final dataset:

* 4200 total samples
* 2100 safe
* 2100 unsafe

Categories:

* animal_abuse
* child_abuse
* controversial_topics,politics
* discrimination,stereotype,injustice
* drug_abuse,weapons,banned_substance
* financial_crime,property_crime,theft
* hate_speech,offensive_language
* misinformation_regarding_ethics,laws_and_safety
* non_violent_unethical_behavior
* privacy_violation
* self_harm
* sexually_explicit,adult_content
* terrorism,organized_crime
* violence,aiding_and_abetting,incitement

⸻

Dataset Audit

Class Balance:

* Safe: 2100
* Unsafe: 2100

Category Balance:

* 150 safe
* 150 unsafe

for every category.

Prompt Diversity:

* Total prompts: 4200
* Unique prompts: 3740
* Duplicates: 460

Prompt Length Audit:

* Unsafe mean length: 13.68 words
* Safe mean length: 13.61 words

Question Audit:

Unsafe prompts ending with “?”:

* 73.5%

Safe prompts ending with “?”:

* 72.1%

Conclusion:

No obvious length or interrogative-form confound was detected.

⸻

Evaluation Protocol

Model:

* Qwen2.5-1.5B-Instruct

Representation:

* Mean pooled hidden states

Probe:

* Logistic Regression

Cross Validation:

* 7 folds

Each fold:

* Train on 12 categories
* Test on 2 unseen categories

Fold size:

* Train: 3600 samples
* Test: 600 samples

This evaluation protocol is significantly more difficult than a random train/test split.

⸻

Results

Best Layer:

* Layer 12

Metrics:

* AUC: 0.561
* Accuracy: 0.545
* F1: 0.563

Observed behavior:

* Performance remained only slightly above chance across all layers.
* No strong middle-layer safety signal emerged.
* No layer exceeded AUC 0.57.

⸻

Interpretation

The result suggests that the learned representation does not strongly generalize across unseen harm categories.

Possible explanations:

1. Policy representations are weak or highly entangled with semantic topic information.
2. BeaverTails labels contain substantial ambiguity and noise.
3. The target variable (is_safe) may not cleanly correspond to harmful intent.
4. The category-held-out evaluation removes topic cues that earlier experiments may have relied upon.

⸻

Key Observation

Manual inspection revealed several prompts labeled as safe despite appearing harmful or policy-relevant.

Examples included:

* requests involving scams
* requests involving private addresses
* requests involving theft-related behavior

This raises concerns about label quality for probing harmful intent.

⸻

Outcome

Experiment 003 did not provide strong evidence for a domain-general safety representation under category-held-out evaluation.

However, the experiment successfully ruled out a simpler explanation:

A probe trained on some harm categories does not automatically generalize to unseen categories.

This makes Experiment 003 an important control study for the NPS hypothesis.

⸻

Next Experiment (Proposed)

Experiment 003B

Construct a deliberately clean dataset containing:

Safe:

* benign informational requests
* educational questions
* everyday assistance requests

Unsafe:

* clearly harmful requests
* criminal assistance requests
* self-harm instructions
* violence instructions

Goal:

Reduce label ambiguity and determine whether stronger safety signals emerge under cleaner supervision.