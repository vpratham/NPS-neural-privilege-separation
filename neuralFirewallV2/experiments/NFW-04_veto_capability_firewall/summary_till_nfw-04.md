 We started with the idea of a neural firewall: a safety layer inside
  or around an AI model that prevents it from producing harmful content
  or taking dangerous actions.

  The project evolved through several stages.

  ———

  ## 1. Can we detect unsafe requests inside the model?

  We first examined the model’s internal activity when it received
  different prompts.

  The idea was:

  - Safe prompts create one type of internal pattern.
  - Unsafe or adversarial prompts create another pattern.
  - A small detector could recognize those patterns and block suspicious
    requests.

  This detector is called a monitor.

  ### What we learned

  The model’s internal activity does contain some safety-related
  information. However:

  - It does not work consistently across different datasets.
  - It misses many unsafe requests.
  - It can incorrectly block safe requests.
  - Performance drops when the requests look different from the examples
    used during training.

  So this approach is useful as a warning signal, but it is not reliable
  enough to be the only safety mechanism.

  ———

  ## 2. Why existing labeled datasets were useful but insufficient

  We used datasets containing:

  - User prompts.
  - Stored model responses.
  - Labels indicating whether a response was harmful.
  - Information about the source of each example.

  This allowed us to test the monitor without manually reviewing every
  new model response.

  However, these datasets have important limitations:

  - Many examples have no response label.
  - Labels may come from different models or unknown review processes.
  - A safe response may simply be a refusal to a harmful request.
  - A label describing an old response does not guarantee what a new
    model will do.

  - Some sources are very different from others.

  Therefore, these datasets are useful for research, but they cannot
  prove that a live model will always behave safely.

  ———

  ## 3. What NFW-002 and NFW-003 showed

  ### NFW-002

  NFW-002 tested whether a monitor could detect unsafe prompts before
  the model generated an answer.

  It demonstrated that the system could:

  - Inspect the prompt.
  - Decide whether to block it.
  - Save progress safely.
  - Resume after a Colab disconnection.
  - Produce reproducible reports.

  But its detection coverage was limited.

  ### NFW-003

  NFW-003 compared several approaches:

  - Simple keyword matching.
  - A normal text-based classifier.
  - Internal model activity.
  - A combination of text and internal activity.

  The results showed that the internal activity detector was somewhat
  informative, but not strong enough to provide dependable protection.

  The most important conclusion was:

  > A model’s internal activity can provide useful safety information,
  > but it should not be trusted as the only authority controlling real-
  > world actions.

  ———

  ## 4. The major design change

  Instead of asking:

  > Can we perfectly predict every harmful answer?

  we changed the question to:

  > Can we prevent the model from gaining authority, even if it makes a
  > mistake?

  This is a much more achievable security goal.

  The model may still produce bad text. But it should not be able to:

  - Run shell commands.
  - Read private files.
  - Send network requests.
  - Transfer money.
  - Change system settings.
  - Grant itself administrator access.
  - Pretend that it has permission.

  ———

  ## 5. The capability broker

  The capability broker is an independent permission system outside the
  model.

  Think of it like a security guard.

  The model can say:

  > “I would like to look up this public fact.”

  But the model cannot decide whether it is allowed to do that.

  The broker checks:

  - What action is being requested.
  - Whether that action is allowed.
  - Which user is requesting it.
  - Which resources may be accessed.
  - Whether the permission is still valid.
  - Whether the permission has already been used.

  Only then can the action be approved.

  ———

  ## 6. Default deny

  The broker follows a simple rule:

  > If something is not clearly allowed, reject it.

  This is called default deny.

  For example:

  - Public fact lookup: allowed if permission exists.
  - Unknown tool: rejected.
  - Shell command: rejected unless explicitly supported.
  - Extra unexpected argument: rejected.
  - Missing permission: rejected.
  - Malformed request: rejected.

  This is safer than assuming that unfamiliar requests are harmless.

  ———

  ## 7. Capability tokens

  A capability token is a temporary permission slip.

  It records things such as:

  - Who received the permission.
  - What action is allowed.
  - Which resource can be accessed.
  - When the permission expires.
  - A unique identifier.

  The model cannot create these tokens.

  Only trusted host-side code can issue them.

  The token system also protects against:

  ### Scope abuse

  A permission for reading a project file cannot be used to perform a
  different action.

  ### Expiration

  An old permission stops working after its allowed time.

  ### Replay

  A used permission cannot simply be reused repeatedly.

  ### Fake authority claims

  If the model says:

  “I am an administrator”

  the broker ignores that statement unless real trusted permission
  exists.

  ———

  ## 8. The neural monitor’s new role

  The neural component is now veto-only.

  That means it can say:

  - “Reject this.”
  - “Pause and require human approval.”
  - “This request looks risky.”

  But it cannot say:

  - “Grant this model more permission.”
  - “Ignore the broker.”
  - “Create a new capability.”
  - “Upgrade this request from low-risk to administrator-level.”

  This is important because neural predictions are imperfect.

  If the neural monitor makes a mistake, the worst result should usually
  be an unnecessary rejection—not unauthorized access.

  ———

  ## 9. Human approval

  Some actions may be too sensitive for automatic approval.

  In those cases, the broker can require a human decision.

  For example:

  - Reading a public document may be automatic.
  - Modifying files may require approval.
  - Sending an email may require approval.
  - Financial or administrative actions may require explicit approval.

  The model cannot approve its own request.

  ———

  ## 10. What the GPU demonstration did

  The Colab notebook loaded Qwen on a GPU and examined its internal
  activity for a few example requests.

  This was only a demonstration that:

  - The model can be inspected.
  - Its internal signals can be passed to the safety layer.
  - The safety layer can use those signals to veto a request.

  The measured activation values are not a safety score. They were not
  trained to reliably recognize harmful behavior.

  The important result is that the broker stayed safe regardless of
  those values.

  ———

  ## 11. What the tests proved

  All 11 security tests passed.

  The tests confirmed that:

  - A valid permission is accepted.
  - A reused permission is rejected.
  - A permission for one action cannot be used for another.
  - Model-provided administrator claims are ignored.
  - A neural veto blocks a request.
  - Human approval is enforced when required.
  - Expired permissions are rejected.
  - Unknown tools are rejected.
  - Unexpected arguments are rejected.
  - Malformed requests are rejected.
  - No real external action was executed.

  The main security properties were:

  The model cannot create permission.
  The model cannot increase its permission.
  The neural signal cannot grant permission.
  Unknown actions are rejected.

  ———

  ## 12. What this project is now

  The project is no longer best described as:

  > “A system that perfectly detects and blocks harmful model outputs.”

  It is better described as:

  > A system where the model can suggest actions, but an independent
  > permission layer controls whether those actions can happen.

  The neural model helps identify risk, but the external broker enforces
  authority.

  ———

  ## 13. What it does not yet prove

  This is still a prototype.

  It does not yet prove:

  - That the model cannot generate harmful text.
  - That all jailbreaks are detected.
  - That the broker is production-secure.
  - That the model process is fully isolated.
  - That cryptographic keys are safely protected.
  - That real tools can be safely connected.
  - That the system works across every model.
  - That the idea is scientifically novel.

  For real deployment, the broker would need to run separately from the
  model, with protected keys, strong user authentication, sandboxing,
  secure logging, and careful handling of race conditions.

  ———

  ## The simplest summary

  We discovered that trying to make the model perfectly recognize every
  unsafe thought is unreliable.

  So we changed the design:

  1. Let the model suggest actions.
  2. Treat the model as untrusted.
  3. Put permissions in a separate security system.
  4. Allow the neural monitor to block or warn.
  5. Never allow the neural monitor or model to grant authority.
  6. Reject anything not explicitly permitted.

  That is the foundation of the current veto-gated neural privilege
  separation design.



