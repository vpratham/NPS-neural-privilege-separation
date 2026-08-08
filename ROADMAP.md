---

### ROADMAP.md

```markdown
# NPS Roadmap

## Phase 1 — Evidence Gathering

### Exp001 — Policy Detection
Status: Complete ✓

Determine whether policy information is detectable in activations.

### Exp002 — Probe Training
Status: Complete ✓

Validate robustness of policy decoding.

### Exp003 — Policy Vector Discovery
Status: Complete ✓

Identify latent directions associated with policy behavior.

---

## Phase 2 — Representation Analysis

### Exp004 — RepE Baseline
Status: Complete ✓

Establish representation engineering baselines.

### Exp005 — Refusal Representation
Status: Complete ✓

Investigate latent representations associated with refusal behavior.

### Exp006 — Policy Steering
Status: Complete ✓

Test causal manipulation of policy directions.

---

## Phase 3 — Behavioral Mapping

### Exp007 — Jailbreak Mapping
Status: Complete ✓

Study interactions between jailbreak prompts and policy representations.

### Exp008 — Compliance Mapping
Status: Complete ✓

Measure compliance under policy interventions.

### Exp009 — Nonlinear Compliance Probing
Status: Complete ✓

Investigate nonlinear structure within policy representations.

---

## Phase 4 — Policy Protection

### Exp010 — Policy Subspace Protection
Status: Next Experiment

Research Question:

Can policy-related latent directions be stabilized against user-controlled perturbations while preserving useful capabilities?

Success Criteria:

- Reduced jailbreak success
- Preserved benign capabilities
- Stable policy representations

---

## Phase 5 — Neural Firewall

### Exp011 — Neural Firewall Prototype
Status: Planned

Develop a prototype architecture that monitors and intervenes on policy representations during inference.

### Exp012+ — Neural Privilege Separation

Investigate explicit separation of:

- User representations
- Capability representations
- Policy representations

with protected communication pathways.

---

## Long-Term Objective

Create a security architecture where policy constraints are enforced as part of the model's internal computation rather than external prompting.