# NPS Permission Boundary

Standalone repository for host-controlled model read permissions. It enforces which document sources the model may read and buffers responses until complete generation. It does not contain the anomaly detector and does not claim to stop instructions in readable evidence.

## Run

Install the pinned optional runtime dependencies and follow [`neural_state_firewall/DOCUMENT_QA.md`](neural_state_firewall/DOCUMENT_QA.md) for the authenticated local document-Q&A pilot. The core claim, supported model/runtime versions, limitations, and reproducible permission tests are in [`READ_PERMISSIONS.md`](neural_state_firewall/READ_PERMISSIONS.md).

Run the package tests with:

```sh
python -m unittest discover -s neural_state_firewall/tests -v
```

This repository deliberately has no prompt-injection classifier or neural anomaly monitor. Tool execution remains outside this repository; generated text is untrusted, and privileged effects require a separate host-authorized mediator.
