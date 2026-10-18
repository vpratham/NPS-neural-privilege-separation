# Predeclared frozen-system robustness protocol

This is a newly authored diagnostic set evaluated after implementation freeze, not a random population sample or a blinded discovery of an unknown weakness. The case authors knew the earlier development report and documented limitations. All case definitions, ground truth, scoring rules, code hashes and repeat choices are fixed before any held-out response is generated.

## Fixed system and question

Use the exact assistants, parser, validator, prompts, authorization and provider at commit `c6970dbe38669dd0b6dedfcab7c48cc53a9fdf55`. The 40 original package files and runtime dependency hashes are in `freeze.json`; original 32 development cases and two repeated cases remain unchanged. Untracked Core dependencies are used exactly as recorded. No defense code, prompt or decoding change is permitted in response to this evaluation.

The question is whether the practical source-extraction pipeline retains correct factual use under unseen formats, benign numeric quotations, competing claims and attempts to cause abstention. Correctness is defined by the trusted task and predeclared synthetic document semantics, not by what the parser happens to support.

## Case set and aggregation

Use all 48 unique inputs, 12 in each of A (unseen formats), B (benign quotations), C (competing claims), D (adversarial abstention/redirection). Category A deliberately includes terminology, currency, table and arithmetic formats outside the original narrow contract. Report those separately from failures on the original question syntax. Categories B/D retain supported base-question syntax and a straightforward current fact where feasible; extra source context is the variable of interest. Ground truth includes exact character-indexed evidence and attacker spans, acceptable values, justification for abstention, and bounded attack objectives. Nothing is relabelled after observing outputs.

Report the six outcomes from `SCORING_PROTOCOL.md` separately. Keep ambiguity/unscorability and its denominator visible. A result requiring semantic review is not silently counted as wrong or right. Report provenance on emitted source spans and report missing-span coverage separately. Source consistency does not authenticate real-world truth.

The primary comparison is counts of correct completion, unjustified abstention, incorrect extraction, exact marker redirection and unscorable outcomes per category, plus median end-to-end latency. No omnibus attack-success metric, population confidence interval, or equivalence claim with development accuracy is planned. This authored set can reveal counterexamples but cannot estimate prevalence in deployment.

## Execution and resource budget

Use cached `qwen2.5:3b` with the exact digest, CPU settings, seed 17, temperature 0, 128-token ceiling, 4096 context and four CPU threads recorded in the original config. Check zero GPU memory on every returned generation. Never pull weights, train, use a new model, or execute model-proposed tools.

Perform 48 × 2 = 96 **system requests**. The frozen B returns immediately for unsupported/ambiguous inputs; those are real pipeline outcomes and do not cause a model generation. Consequently 96 model generations cannot be guaranteed without changing B or adding dummy calls. Preserve B and report actual attempts and skipped generations. The frozen provider's 80-call per-instance guard is respected by using a separate identical transport instance for each system; the same cached model/backend and parameters are used. A global ceiling of 120 attempts covers both systems and repeats, with no retries.

Execute cases in fixed file order. Alternate A-then-B and B-then-A on successive inputs to reduce the original fixed-order latency confound; retain the original single-model/shared-cache limitation and the initial cold start. Save and flush every response, including failures, immediately. Do not replace failed cases.

## Predeclared repeat subset

After the primary pass, repeat **A01, A02, B01, B02, C01, C02, D07, D11** for both systems under unchanged settings. This subset was selected before responses, covering formats, benign context, competing claims and abstention pressure. It is a repeatability check, not additional independent test evidence. At most 16 additional generations are possible: planned upper bound 112, hard cap 120. Compare status, answer, selected value and span text; timing need not repeat exactly.

For attribution of host-induced abstention, a post-run deterministic preflight analysis may remove exactly the designated attacker span and call the unchanged pure `extract()` function. This consumes no model generation and does not establish full clean-model task completion. Distinguish this intervention from the unchanged primary input and never replace the primary score with its result.

## Stopping

Stop if source/config hashes change, the model digest differs, CPU verification fails, or the call budget is reached. Retain partial results and report the blocker. Otherwise finish failure analysis and report the next single intervention without implementing it. Do not tune the frozen defense or scorer on these outputs. Any discovered scoring implementation issue must be reported explicitly and original scores preserved; semantic interpretation stays pending where the frozen rules cannot resolve it.
