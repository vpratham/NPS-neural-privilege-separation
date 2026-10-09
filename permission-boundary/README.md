# NPS Permission Boundary

Workstream for host-controlled model permissions. It enforces document read permissions and includes a provider-independent capability gate for per-instance read, disclosure and action-proposal grants. Action proposals still require a trusted broker; the current local Q&A API remains free-form and is not protected by the quote-only gate. It does not contain the anomaly detector or claim to stop instructions in readable evidence.

The staged research, data, hardware, measurement, and proof plan is [`NPB_PLAN.md`](NPB_PLAN.md).

## Run

Install the pinned optional runtime dependencies and follow [`neural_state_firewall/DOCUMENT_QA.md`](neural_state_firewall/DOCUMENT_QA.md) for the authenticated local document-Q&A pilot. The core claim, supported model/runtime versions, limitations, and reproducible permission tests are in [`READ_PERMISSIONS.md`](neural_state_firewall/READ_PERMISSIONS.md).

Run the package tests with:

```sh
python -m unittest discover -s neural_state_firewall/tests -v
```

This repository deliberately has no prompt-injection classifier or neural anomaly monitor. Tool execution remains outside this repository; generated text is untrusted, and privileged effects require a separate host-authorized mediator.
