# Neural Privilege Separation (NPS)

Neural Privilege Separation (NPS) is an AI security research project investigating whether policy representations inside transformer models can be identified, isolated, and protected from user-controlled influence.

## Motivation

Modern LLM safety primarily relies on external mechanisms:

- System prompts
- Guardrails
- Moderation layers
- Input filtering
- Output filtering

These approaches operate outside the model's internal computation and remain vulnerable to prompt injection, jailbreaks, and instruction hierarchy attacks.

NPS explores whether policy constraints can become a property of computation itself.

## Core Research Question

Can policy representations inside transformer models be:

1. Identified
2. Measured
3. Manipulated
4. Protected

without significantly degrading useful capabilities?

## Current Findings

### Completed

- Exp001: Policy signal detection
- Exp002: Probe training and validation
- Exp003: Policy vector discovery
- Exp004: RepE baseline
- Exp005: Refusal representation analysis
- Exp006: Policy steering
- Exp007: Jailbreak mapping
- Exp008: Compliance mapping
- Exp009: Nonlinear compliance probing

### Key Results

- Policy information is detectable from activations.
- Linear probes achieved approximately AUC ≈ 0.73.
- Candidate policy vectors were identified.
- Steering these vectors causally alters compliance behavior.
- Compliance changes monotonically under intervention.

## Repository Structure

```text
datasets/    Prompt datasets and taxonomies
experiments/ Experiment notebooks and logs
artifacts/   Learned vectors and generated artifacts
results/     Experiment summaries
evidence/    PDFs and supporting material
theory/      Research notes and architecture ideas