# Neural Privilege Separation

Neural Privilege Separation (NPS) is an AI-security research project studying whether a language model can maintain policy-relevant computation under attacker-controlled input. The long-term objective is a neural firewall: an internal monitor and controller that reduces unsafe or unauthorized behavior while preserving legitimate capability.

The project distinguishes five claims that need different evidence:

| Claim | Evidence required |
|---|---|
| A representation is readable | Held-out activation-probe performance |
| A representation is causal | Controlled intervention with behavioral outcomes |
| A monitor improves security | Actual target-model outputs, independently judged |
| Policy state is protected | Measured resistance to unauthorized influence |
| The mechanism generalizes | Held-out domains, attacks, and model families |

The first two are useful scientific steps. They are not security guarantees by themselves.

## Current direction

The repository contains exploratory experiments, legacy activation artifacts, and a new foundational monitor-and-block proof of concept. The recommended entry point is [NFW-002](neuralFirewallV2/experiments/NFW-02_foundation_firewall/), which creates one reproducible activation-monitoring experiment with:

- canonical Qwen chat serialization;
- explicit, verified activation-site conventions;
- grouped train/development/calibration/final splits;
- frozen detector artifacts and provenance manifests;
- complete response accounting; and
- behavioral evaluation on actual target-model outputs.

Read the [implementation audit](docs/NPS_IMPLEMENTATION_AUDIT.md) before interpreting historical results. It documents known dataset, evaluation, and runtime limitations in prior experiments.

## Repository layout

```text
datasets/                 Prompt sources, taxonomies, and generated datasets
docs/                     Research charter and implementation audit
experiments/              Historical NPS notebooks and exploratory runs
neural_firewall/          Earlier modular firewall prototype and experiments
neuralFirewallV2/         Current research program and NFW experiment series
results/                  Historical result summaries and archived artifacts
theory/                   Notes on the NPS hypothesis
```

## Research sequence

```text
Readable representation
        ↓
Validated monitor-and-block baseline
        ↓
Selective causal controller
        ↓
Policy-conditioned authority experiments
        ↓
Adaptive robustness and cross-model validation
```

The next objective is not to claim a finished neural firewall. It is to obtain a complete, target-model behavioral evaluation for the monitor-and-block baseline. That result determines whether to invest in continuation monitoring and intervention.

## Getting started

1. Review [NFW-002's experiment guide](neuralFirewallV2/experiments/NFW-02_foundation_firewall/README.md).
2. Prepare a reviewed two-class prompt dataset with stable IDs and paraphrase/behavior group IDs.
3. Run `NFW_002_Foundational_Monitor_Block_POC.ipynb` in Colab or a CUDA environment.
4. Independently label the target model's released responses before opening the final report cell.

## Scope and limitations

NPS is an active research project. Existing activation probes and interventions are experimental; they should not be represented as a deployed safety guarantee. A blocked request is not, by itself, proof that a harmful output was prevented, and a high probe AUC is not proof of privilege separation.

## Reference materials

- [NPS Charter](docs/NPS_Charter.md)
- [Implementation audit](docs/NPS_IMPLEMENTATION_AUDIT.md)
- [Neural Firewall v2 program](neuralFirewallV2/README.md)
