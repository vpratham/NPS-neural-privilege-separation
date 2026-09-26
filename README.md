<div align="center">

# Neural Privilege Separation

**A research program for internal policy-state protection in language models.**

[![Status](https://img.shields.io/badge/status-active%20research-0f766e)](#current-status)
[![Stage](https://img.shields.io/badge/stage-monitor--and--block%20baseline-2563eb)](#current-status)
[![Reference Model](https://img.shields.io/badge/reference-Qwen2.5--3B--Instruct-7c3aed)](neuralFirewallV2/)
[![Claims](https://img.shields.io/badge/security%20claims-explicitly%20scoped-b45309)](#scope-and-limitations)

</div>

Neural Privilege Separation (NPS) studies whether a language model can maintain
policy-relevant computation under attacker-controlled input. The long-term goal
is a **neural firewall**: an internal monitor and controller that reduces unsafe
or unauthorized behavior while preserving legitimate capability.

This repository contains a research archive, a prompt-injection request/response
firewall prototype, and a separate action authorization broker. The firewall
gates model answers against application policy and explicitly marked untrusted
context. The broker handles source-bound tool effects.

## Prompt-injection firewall

```bash
python3 -m nps_gateway firewall-demo --provider ollama --model qwen2.5:3b
```

The firewall checks model responses against the task, service-owned policy, and
untrusted retrieved context before returning a response. It provides a local
OpenAI-compatible API endpoint:

```bash
python3 -m nps_gateway firewall-serve \
  --policy-file examples/gateway/firewall-policy.txt \
  --provider ollama --model qwen2.5:3b --judge-model qwen2.5:3b
```

See the [firewall guide](docs/PROMPT_INJECTION_FIREWALL.md) and
[live validation evidence](docs/PROMPT_FIREWALL_VALIDATION.json). The local
smoke test passed one clean task and blocked one injected response. The policy
judge is model-based, so this is a useful prototype rather than a guarantee.

The separate [action broker guide](docs/WORKING_SYSTEM.md) describes source-bound
permissions for tool actions. It is not the text firewall. The [evidence review](docs/WORKING_SYSTEM_EVIDENCE.md)
explains both how earlier results shaped the boundaries.

Python 3.10+; no Python runtime dependencies.

The sections below describe the research background and earlier milestones.

## Why This Exists

Most guardrails inspect text at the boundary: the prompt, the response, or both.
NPS asks a different question:

> Can policy-relevant computation be identified, monitored, and eventually
> protected inside the model's internal state?

The project separates five claims that are often conflated:

| Claim | Evidence required |
|---|---|
| A representation is readable | Held-out activation-probe performance |
| A representation is causal | Controlled intervention with behavioral outcomes |
| A monitor improves security | Actual target-model outputs, independently judged |
| Policy state is protected | Measured resistance to unauthorized influence |
| The mechanism generalizes | Held-out domains, attacks, and model families |

The first two are useful scientific steps. They are not security guarantees by
themselves.

## Current Status

| Area | Status |
|---|---|
| Theory | Draft mathematical framework for NPS and neural-firewall security objectives |
| Historical experiments | Activation probes, policy-vector experiments, causal pilots, and audits |
| Application firewall | OpenAI-compatible, buffered request/response prototype |
| Neural firewall | NFW-002 monitor-and-block research baseline; internal policy-state protection is not established |
| Reference model | `Qwen/Qwen2.5-3B-Instruct` |
| Security claim | No robust NPS claim yet; current work is a scoped research baseline |

The current application prototype uses role-separated untrusted context and
withholds complete responses pending a policy check. It is a practical text-level
defense layer. Internal neural policy-state protection remains a distinct research
goal and has not been demonstrated.

## Research Roadmap

```text
Readable representation
        |
        v
Validated monitor-and-block baseline
        |
        v
Selective causal controller
        |
        v
Policy-conditioned authority experiments
        |
        v
Adaptive robustness and cross-model validation
```

The eventual NPS target is an internal security boundary with:

- an explicit attacker model;
- measurable policy-state isolation;
- a protected policy invariant;
- acceptable capability retention; and
- adaptive robustness evidence.

## Repository Map

```text
datasets/                 Prompt sources, taxonomies, and generated datasets
docs/                     Charter, implementation audit, and forensic notes
experiments/              Historical notebooks and exploratory runs
neural_firewall/          Earlier modular firewall prototype and experiments
neuralFirewallV2/         Current NFW experiment series and engineering branch
results/                  Historical result summaries and archived artifacts
theory/                   Mathematical framework and NPS theory notes
```

## Start Here

| Goal | Entry point |
|---|---|
| Run the prompt-injection firewall | [Firewall guide](docs/PROMPT_INJECTION_FIREWALL.md) |
| Integrate protected tool actions | [Action broker](docs/WORKING_SYSTEM.md) |
| See how the accumulated findings shaped it | [Evidence review](docs/WORKING_SYSTEM_EVIDENCE.md) |
| Understand the research claim boundary | [Implementation audit](docs/NPS_IMPLEMENTATION_AUDIT.md) |
| Read the theory foundation | [NPS mathematical framework](theory/NPS_Mathematical_Framework_v0_2.tex) |
| Run the current monitor baseline | [NFW-002 experiment guide](neuralFirewallV2/experiments/NFW-02_foundation_firewall/README.md) |
| Review the engineering program | [Neural Firewall v2](neuralFirewallV2/README.md) |
| See the original research vision | [NPS charter](docs/NPS_Charter.md) |

## Current Experiment Track

The recommended current entry point is
[NFW-002](neuralFirewallV2/experiments/NFW-02_foundation_firewall/), a
foundational activation-monitoring experiment with:

- canonical Qwen chat serialization;
- explicit activation-site conventions;
- grouped train/development/calibration/final splits;
- frozen detector artifacts and provenance manifests;
- complete response accounting; and
- behavioral evaluation on actual target-model outputs.

NFW-002 is intentionally a **monitor-and-block baseline**. It does not establish
neural privilege separation, policy-state invariance, adaptive robustness, or a
deployed safety guarantee.

## Paper Groundwork

The theory draft frames NPS as a conditional security objective:

```text
identify policy-relevant state
        |
isolate trusted policy configuration from untrusted influence
        |
constrain internal transitions
        |
measure behavioral security and capability preservation
```

The current paper-ready contribution is best positioned as a **formal framework
and evaluation ladder** for neural-firewall research, supported by preliminary
monitoring experiments and explicit negative/limitation findings.

## Scope and Limitations

NPS is active research. Existing activation probes and interventions are
experimental and should not be represented as deployed safety guarantees.

Important boundaries:

- A high probe AUC is not proof of privilege separation.
- A blocked request is not, by itself, proof that harmful output was prevented.
- A successful intervention is not automatically a security guarantee.
- Robustness claims require an explicit attacker model and adaptive evaluation.
- Historical results must be interpreted through the implementation audit.

## Contributing

Contributions should preserve the distinction between evidence and hypothesis.
See [CONTRIBUTING.md](CONTRIBUTING.md) for experiment, artifact, and reporting
guidelines.
