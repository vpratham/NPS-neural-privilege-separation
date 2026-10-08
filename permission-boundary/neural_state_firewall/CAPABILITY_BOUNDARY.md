# Per-instance capability boundary

Each `CapabilityBoundary` is bound at construction to a host-owned `CapabilityProfile`. The profile separates three grants:

- `readable_sources`: evidence the model provider may receive.
- `disclosable_sources`: sources whose exact text may be released. This must be a subset of readable sources.
- `actions`: action names and exact argument-key sets the model may propose.

The request cannot add grants. Evidence containing an ungranted source is rejected before calling the model. Arbitrary prose is rejected. An answer can only be released as an exact quote from a disclosable source. An action is returned as an inert proposal and still requires authorization by trusted host code or a broker; this module never executes it. Argument names are bounded here, while types, values, user consent and side effects remain the broker's responsibility.

The provider protocol is model-agnostic: an integration implements `generate(task, evidence, allowed_actions, max_new_tokens)`. The included demo uses a scripted provider and makes no model-quality claim:

```sh
python -m neural_state_firewall.capability_demo
python -m unittest discover -s neural_state_firewall/tests -v
```

This capability gate is separate from the local Transformers `Firewall` because that adapter currently emits free-form text and has no tool-call interface. The existing API remains read-permission Q&A; it does not silently gain this gate's narrower quote-only response contract. Use `CapabilityBoundary` where the host can route model responses through this schema before showing text or handing a proposal to a broker.

## Prompt-isolation research target

`ProtectedPolicyCache` gives the system-policy K/V tensors separate storage from the writable generation cache and checks their digest. For supported causal models, prefix computation is causally independent of later suffix tokens. Those are concrete state-integrity and information-flow properties; they do not prove that a model obeys the policy. Generated tokens can attend to both policy memory and authorized context, so readable evidence can still persuade the model to answer incorrectly or disregard policy.

A future formal claim should define “not diluted” operationally. A tractable first property is policy-state integrity: for fixed weights, tokenizer, policy and runtime, every generation step preserves the policy-cache state exactly while only the working cache grows. A stronger semantic claim—that all outputs obey the policy under arbitrary readable contexts—does not follow from cache separation and needs a different mechanism and proof. Current tensor hashes and tests are implementation checks, not a mathematical proof for arbitrary models or runtimes.
