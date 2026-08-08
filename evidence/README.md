safe unsafe classification wasn't working as expected, moved on to visualisation of refusal behavior to indicate where policy layer activation actually occurs for research


Experiment 00 — Safe vs Unsafe Probing

* Activation extraction
* PCA
* UMAP
* Linear probes
* Negative result (AUC ≈ 0.52)

Experiment 02 — Refusal Prediction

* Refusal labeling via model behavior
* Layer-wise probes
* Best AUC ≈ 0.73–0.74
* Stable across seeds

Experiment 03 — Policy Vector Discovery

* Refusal centroid
* Compliance centroid
* Activation difference analysis
* Top refusal neurons
* Policy vector extraction
* Policy vector AUC ≈ 0.728
