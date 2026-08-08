# Experiment 00 — RepE Harmfulness Replication

## Objective
Replicate the core RepE finding that harmfulness-related information is encoded in model activations and can be decoded using simple probes.

## Research Question
Can hidden-state activations distinguish harmful instructions from benign instructions?

## Dataset
- Harmful prompts: AdvBench
- Benign prompts: ShareGPT
- Initial target: 500 harmful + 500 benign prompts
- Balanced binary classification setup

## Methodology
1. Extract hidden-state activations from the model.
2. Compare pooling strategies:
   - Mean pooling across tokens
   - Final-token pooling
3. Train linear probes (Logistic Regression) on activations.
4. Evaluate layer-wise decoding performance.
5. Visualize representations using PCA and UMAP.
6. Record best-performing layer and probe configuration.

## Metrics
- Accuracy
- F1 Score
- ROC-AUC

## Expected Outputs
- Layer-wise probe performance
- Best-performing layer
- PCA visualization
- UMAP visualization
- Activation dataset for future experiments

## Success Criterion
Probe performance significantly above chance (50%), indicating that harmfulness-related information is decodable from hidden states.

## Results
- Activation extraction pipeline completed.
- Linear probing pipeline completed.
- Harmfulness signal successfully decoded above chance.
- Experiment established a working baseline for subsequent NPS experiments.

## Key Takeaway
This experiment serves as a pipeline validation study. A positive result demonstrates that behavior-relevant information can be recovered from hidden states, but does not by itself establish the existence of a policy-specific representation.

## Next Experiment
Experiment 01: Controlled Policy Representation Probing

Goals:
- Reduce topic leakage.
- Separate policy from semantic topic.
- Compare random-split and topic-held-out evaluation.
- Determine whether policy signals generalize beyond specific domains.