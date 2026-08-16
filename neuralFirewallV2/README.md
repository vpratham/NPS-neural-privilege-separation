# Neural Firewall v2

Neural Firewall v2 is the engineering branch of the NPS (Neural Privilege Separation) project focused on building a coherent, inference-time neural firewall for modern language models.

The objective is to move beyond behavioral jailbreak detection toward an internal security boundary that monitors and constrains policy-relevant model representations during inference, while using capability hardening as a second layer of defense.

## Core Idea

The system is being developed around two complementary mechanisms:

1. **Neural firewall** — monitor policy-relevant internal activations and constrain unsafe internal states during inference.
2. **Capability hardening** — reduce or suppress dangerous capabilities so that bypassing the runtime boundary is less useful and less recoverable.

The long-term goal is to determine whether policy-relevant computation can be treated as **privileged internal state**: user-controlled input should be able to influence legitimate task computation without being able to arbitrarily rewrite the state governing policy.

## Research Progression

```text
Representation
      ↓
Causal control
      ↓
Policy geometry
      ↓
Inference-time firewall
      ↓
Policy isolation
      ↓
Policy invariance
      ↓
Adversarial testing
      ↓
Capability hardening
      ↓
Combined defense
      ↓
Cross-model validation
```

The central research question is:

> **Can we identify, control, and protect a policy-relevant internal representation from attacker-controlled input while preserving legitimate model capabilities?**

## Relationship to the NPS Research

This repository does not start from zero.

It builds on the experimental work already conducted in the parent NPS project, including activation-space representation analysis, probe-direction experiments, causal interventions, layer-wise analysis, causal sweeps, and activation-space firewall prototypes.

Existing results are treated as prior evidence and will be integrated into this implementation rather than unnecessarily reproduced.

Where an old result is used as a foundation for the firewall, it will be reproduced or validated through the standardized pipeline in this repository.

The previous research establishes useful evidence around representation and causal intervention. It does **not** by itself establish that a protected policy state exists or that a neural firewall provides robust security. Those are the primary new questions of Neural Firewall v2.

## Current Development Model

**Qwen/Qwen2.5-3B-Instruct** is the initial reference model because the previous NPS experiments provide the strongest existing foundation for activation-space and causal analysis on this model.

Additional models will be introduced only after the core firewall mechanism is established.

Cross-model validation will then test whether the underlying security mechanism generalizes within and across model families.

## Repository Structure

```text
neuralFirewallV2/
├── documentation/
│   ├── 2306.03341v6.pdf
│   ├── 2307.10163v1.pdf
│   ├── 2310.01405v4.pdf
│   ├── 2403.03218v7.pdf
│   ├── NPS_Mathematical_Framework_v0_1.tex
│   └── NPS_Mathematical_Framework_v0_2.tex
├── results/
├── src/
└── README.md
```

### documentation/

Contains the core research papers and the mathematical framework informing the neural firewall.

### results/

Stores experiment outputs, summaries, checkpoint metadata, and generated research artifacts.

### src/

Contains reusable implementation code for model loading, activation extraction, policy representation, policy scoring, causal intervention, firewall enforcement, capability hardening, and evaluation.

Experiments should consume these components rather than reimplementing model hooks and intervention logic independently in each notebook.

## Development Roadmap

### NFW-00 — Legacy Integration & Reproducibility

Integrate the strongest existing NPS artifacts into a canonical representation and intervention interface.

This stage should:

- identify the validated policy representations from previous experiments;
- load existing probe directions and intervention parameters;
- preserve artifact metadata and provenance;
- reproduce key legacy results on a validation set; and
- expose the resulting functionality through reusable code in `src/`.

**Deliverable:** a canonical policy-representation and intervention layer that the firewall can use.

### NFW-01 — Neural Firewall v0

Convert the validated representation and causal intervention mechanism into a reusable inference-time firewall.

Conceptually:

```text
hidden state
    ↓
policy representation
    ↓
safety state
    ↓
intervention / constraint
    ↓
continue inference
```

**Deliverable:** the first working `NeuralFirewall` implementation.

### NFW-02 — Baseline Security & Capability Evaluation

Compare the baseline model with the initial firewall using standardized metrics for attack success, false positives, intervention behavior, and legitimate capability retention.

### NFW-03 — Policy Isolation

Measure whether attacker-controlled input can move the protected policy representation.

A central quantity is:

$$
\sup_{x' \in \mathcal{A}(x)} d_P(P(x'),P(x)).
$$

The question is not merely whether the model refused. The question is how much control attacker-controlled input has over the protected internal policy state.

### NFW-04 — Policy Invariance

Test whether the model can recreate an unsafe policy state after intervention and whether the protected state can be maintained across subsequent layers.

Target property:

$$
P_\ell \in \mathcal{P}_{\mathrm{safe}}
\Rightarrow
P_{\ell+1} \in \mathcal{P}_{\mathrm{safe}}.
$$

### NFW-05 — Adversarial Firewall Attacks

Actively optimize against the firewall rather than evaluating only conventional jailbreak prompts.

The attack suite will progress from behavioral attacks to adaptive attacks against the internal security mechanism, including attempts to bypass monitored layers or recreate unsafe states after intervention.

### NFW-06 — Capability Hardening

Investigate representation-level capability suppression as a second line of defense.

The initial direction includes RMU-inspired forget/retain objectives:

$$
L =
L_{\mathrm{forget}}
+
\alpha L_{\mathrm{retain}}.
$$

The objective is not simply to erase information, but to reduce dangerous capability while retaining legitimate capability.

### NFW-07 — Combined Defense

Evaluate the complete system:

$$
\boxed{\text{Neural Firewall} + \text{Capability Hardening}}
$$

using controlled comparisons:

| System | Firewall | Hardening |
|---|---:|---:|
| Baseline | No | No |
| Firewall only | Yes | No |
| Hardening only | No | Yes |
| Full system | Yes | Yes |

### NFW-08 — Cross-Model Validation

Evaluate the mechanism beyond Qwen2.5-3B-Instruct, first within the same family and then across model families.

The objective is to establish whether the mechanism generalizes rather than merely producing a model-specific result.

### NFW-09 — Final Benchmark

Produce a standardized security/capability benchmark across validated models and attack classes.

## Colab-First Engineering

All experiments are designed to run on Google Colab.

Requirements:

- Google Drive-backed experiment storage;
- automatic checkpointing;
- resumable experiments after runtime disconnects;
- configuration and seed capture;
- JSON/CSV result artifacts;
- model and dataset manifests;
- small validation runs before expensive sweeps; and
- reusable code in `src/` rather than notebook-only implementations.

Intended workflow:

```text
GitHub
  ↓
Google Colab
  ↓
Google Drive
  ↓
checkpoints / activations / results
  ↓
analysis and reproducible reports
```

Every expensive experiment should be resumable.

Every result should have enough metadata to reproduce how it was generated.

## Evaluation Philosophy

### A Probe Is Not a Firewall

High probe accuracy establishes that a representation is readable. It does not establish causal control, policy isolation, invariance, adversarial robustness, or security.

Likewise, a successful intervention is not by itself a security guarantee.

The project therefore evaluates increasingly strong properties:

$$
\boxed{
\text{Read}
\rightarrow
\text{Control}
\rightarrow
\text{Protect}
\rightarrow
\text{Attack}
\rightarrow
\text{Harden}
\rightarrow
\text{Generalize}
}
$$

The eventual target is an internal security boundary with an explicit attacker model, measurable policy-state isolation, a protected policy invariant, acceptable capability retention, and resistance to recovery attacks.

## Success Criteria

### Level 1 — Policy Representation

Demonstrate a policy-relevant internal representation that generalizes beyond superficial prompt features.

### Level 2 — Neural Firewall

Demonstrate that the representation can be causally controlled and enforced during inference with improved security and acceptable capability preservation.

### Level 3 — Neural Privilege Separation

Demonstrate bounded influence of attacker-controlled input over the protected policy state under an explicit threat model, together with policy-state invariance and adversarial robustness.

Level 3 is the long-term research objective and should not be claimed until the required evidence exists.

## Current Status

**Repository stage:** Architecture and implementation planning

**Reference model:** Qwen2.5-3B-Instruct

**Existing foundation:** NPS activation-space, probe, intervention, and causal-sweep results

**Immediate next milestone:** `NFW-00 — Legacy Integration & Reproducibility`

The first implementation task is to turn the strongest existing NPS representation and causal-intervention artifacts into a canonical, reusable interface for the neural firewall.

## Research Principles

1. **Reuse validated prior work.** Existing NPS experiments are inputs to this project, not work that should be discarded.
2. **Separate evidence from hypothesis.** A result is only claimed to establish what its experiment actually tests.
3. **Do not equate probing with causality.** Representation readability and causal control are separate claims.
4. **Do not equate intervention with security.** A firewall must be tested against adaptive attacks.
5. **Define the threat model explicitly.** Security claims are conditional on the attacker's capabilities and constraints.
6. **Preserve legitimate capability.** A system that simply refuses everything is not a successful firewall.
7. **Prefer reusable engineering over notebook duplication.** Experiments should call the same core implementation.
8. **Validate before scaling.** Prove the mechanism on the reference model before adding more models or expensive sweeps.
9. **Make every security claim falsifiable.** Each claimed property must correspond to a measurable experiment.
10. **Treat bypass as expected.** The firewall should be designed and evaluated under the assumption that an adaptive attacker will search for alternative routes.

## Long-Term Objective

The final system is intended to implement:

$$
\boxed{
\text{Protected Policy State}
+
\text{Constrained Internal Transition}
+
\text{Capability Preservation}
+
\text{Adversarial Robustness}
}
$$

The central hypothesis is:

> **User-controlled input can influence legitimate capability computation, but should not be able to arbitrarily rewrite the protected state governing policy.**

This repository is the engineering effort to test that hypothesis and, if supported by evidence, turn it into a working neural firewall.
