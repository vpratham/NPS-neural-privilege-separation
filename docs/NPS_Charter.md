# **Neural Privilege Separation (NPS)**

## **Research Vision**

Current LLM security largely relies on external controls such as:

-   System prompts
-   Guardrails
-   Input sanitization
-   Output validation
-   Moderation layers
-   Defense-in-depth architectures

These approaches operate outside the model’s core computation and remain vulnerable to prompt injection, jailbreaks, context poisoning, and instruction hierarchy manipulation.

This project investigates whether security can be implemented as a property of the model’s internal computation rather than as a property of prompts.

The long-term goal is to explore a new security primitive for LLMs analogous to privilege separation in operating systems.

----------

# **Core Hypothesis**

Modern transformer models may internally encode at least three distinct categories of information:

1.  User Intent
2.  Capability
3.  Policy

Current models likely store these representations in entangled latent spaces.

This entanglement allows user-controlled inputs to influence policy-related computations, enabling prompt injection and jailbreak attacks.

If policy-related representations can be identified, isolated, and protected, it may become possible to build LLM architectures that are inherently more resistant to instruction override attacks.

----------

# **Research Question**

Can policy representations inside transformer models be identified, separated, and protected from user-controlled influence?

----------

# **Long-Term Objective**

Develop a framework called:

## **Neural Privilege Separation (NPS)**

where policy representations function similarly to privileged memory in operating systems.

Desired property:

User inputs should influence task execution but should not be able to overwrite protected policy representations.

Conceptually:

User Domain -> Capability Domain -> Policy Domain -> Output

rather than:

User Prompt + Policy Prompt -> Shared Latent Space -> Output

----------

# **Research Roadmap**

## **Phase 1: Evidence Gathering**

Goal:

Determine whether policy information exists as a measurable latent signal.

Questions:

-   Can hidden states predict safe vs unsafe requests?
-   Can hidden states predict model refusal behavior?
-   Can policy information be detected independently of prompt text?

Success Criteria:

-   Hidden-state classifier performs significantly above chance.
-   Policy information is detectable from activations alone.

Deliverables:

-   Activation extraction pipeline
-   Hidden-state dataset
-   Initial classifier results

----------

## **Phase 2: Domain Discovery**

Goal:

Investigate whether User, Capability, and Policy occupy separable latent structures.

Questions:

-   Can User Intent be decoded?
-   Can Capability be decoded?
-   Can Policy be decoded?
-   Are these signals independent?

Methodology:

-   UMAP
-   PCA
-   Linear probes
-   Representation analysis

Success Criteria:

Evidence of latent clustering or separable directions.

Deliverables:

-   Visualizations
-   Probe accuracies
-   Latent-space analysis

----------

## **Phase 3: Policy Manifold Discovery**

Goal:

Identify latent directions associated with policy adherence.

Questions:

-   Do safe and unsafe prompts occupy different regions?
-   Do successful jailbreaks converge toward common latent trajectories?
-   Is there a measurable policy axis?

Methodology:

-   Probe weight analysis
-   Representation steering
-   Activation statistics

Success Criteria:

Discovery of candidate policy vectors.

Deliverables:

-   Policy-direction measurements
-   Latent-space mapping

----------

## **Phase 4: Neural Firewall**

Goal:

Intervene directly in latent space.

Architecture:

Prompt  
↓  
LLM  
↓  
Activation Monitor  
↓  
Policy Intervention  
↓  
Output

Questions:

-   Can policy violations be detected before output generation?
-   Can unsafe trajectories be redirected?
-   Can safety be enforced without degrading capabilities?

Success Criteria:

Reduced jailbreak success with minimal capability loss.

Deliverables:

-   Neural firewall prototype
-   Activation steering experiments

----------

## **Phase 5: Neural Privilege Separation**

Goal:

Explore explicit separation of cognition domains.

Hypothesis:

Instead of a single hidden state:

H

The model may benefit from:

H_user

H_capability

H_policy

with controlled communication pathways.

Questions:

-   Can policy representations be isolated?
-   Can policy representations be protected?
-   Can user inputs be prevented from modifying privileged representations?

Success Criteria:

Prototype architecture demonstrating privileged policy representations.

Deliverables:

-   Architectural proposal
-   Experimental implementation
-   Evaluation against jailbreak attempts

----------

# **Initial Experimental Goal**

The first objective is intentionally modest.

We are NOT trying to solve jailbreaks.

We are trying to answer:

“Do policy representations exist as identifiable latent structures?”

Success on this question determines whether Neural Privilege Separation is worth pursuing.

----------

# **Immediate Next Step**

Build a Colab notebook that:

1.  Loads a small open-source model.
2.  Extracts hidden states.
3.  Creates labeled prompt datasets.
4.  Visualizes latent representations.
5.  Trains probes for:
    -   User Intent
    -   Capability
    -   Policy

The outcome of this experiment will determine the direction of future research.

----------

# **Career Goal**

Produce original, evidence-based AI security research that demonstrates expertise in:

-   LLM security
-   Adversarial machine learning
-   Mechanistic interpretability
-   Representation learning
-   Neural architecture design

with the eventual aim of contributing novel security mechanisms for advanced AI systems.

----------

# **Guiding Principle**

Do not ask:

“How do we build a better guardrail?”

Ask:

“How do we make policy constraints part of the computation itself?”


-----------

What this means for the next month

I would focus exclusively on:

RepE Replication Track

1. Finish activation extraction
2. Build harmful/harmless dataset
3. Train probes
4. Generate PCA/UMAP
5. Reproduce harmfulness detection

NPS Extension Track

6. Add jailbreak variants
7. Compare latent representations
8. Measure policy persistence

If we can show:

Harmfulness remains encoded even when the model complies

that alone is already the beginning of a workshop paper and directly motivates Neural Privilege Separation.

That’s a much clearer trajectory than jumping immediately to “neural firewall” or new architectures. The RepE paper gives us the ladder; our contribution is climbing one rung higher and asking what happens to policy representations when they lose control of behavior.