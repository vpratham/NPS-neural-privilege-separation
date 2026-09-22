# Run NFW-009 in Google Colab

1. Open [`NFW_009_Tool_Result_Injection.ipynb`](NFW_009_Tool_Result_Injection.ipynb) in Colab, choose **Runtime → Change runtime type → GPU**, and select **Runtime → Run all**. Approve the Google Drive mount. No NFW-008 files are required; a different Drive account is fine.
2. The notebook installs pinned `transformers==4.57.1` and `accelerate==1.11.0`, writes the frozen protocol to Drive, runs 15 local broker controls, downloads one model at a time, and checkpoints each of 288 responses. The 3B download and generation are the expensive steps.
3. Read `MyDrive/NFW-009/nfw009_tool_result_001/REPORT.md` and `final_report.json`. Retain the **entire run directory**, especially raw responses and rendered prompts, for research audit. Do not read `status: complete` as a safety verdict; inspect the security, attack-proposal, format, and utility rows separately.

## Reconnect safely

After a disconnect, attach a fresh GPU runtime and **Run all** again with the same notebook and run ID. Completed responses are validated and skipped; only absent cases are regenerated. A malformed model output is an observed response and is not retried. An interrupted generation leaves that case absent, so it is retried. If the run stops during model generation, rerun all cells rather than deleting checkpoint files. Each response records GPU/CUDA/Torch identity; a mixed-GPU resumed run is flagged in the report rather than silently treated as bitwise identical.

Do **not** edit a response, rendered prompt, manifest, or report; do not run two sessions writing the same folder. If a model revision, notebook code, package/Python environment, task card, or decoding setting changes, use a **new run ID**. A manifest mismatch is an intentional scientific-integrity guard, not a cache bug. For a report-only reopening after all responses exist, set `NFW009_REVIEW_ONLY=1` in the environment before the setup cell; the notebook will refuse missing responses instead of loading models.

## What can fail and what to report

- **GPU unavailable or out of memory:** start a fresh Colab GPU and rerun all cells. Models load sequentially; no completed outputs should be lost.
- **Rendered-template preflight fails:** stop and preserve the traceback. This means the tokenizer did not render the required real tool-call/tool-response transcript. Do not bypass the assertion or reuse that run ID.
- **Package/version or manifest mismatch:** return to the same Colab environment or start a new run ID; never manually change the manifest to force reuse.
- **Low clean-task success:** report it. Injection effects cannot be interpreted as robust task-preserving defense when the basic task already fails. The notebook retains every planned card in the denominator.
- **Nonzero unauthorized or wrong-content effects:** report them plainly, with their exact condition, goal, model, and broker mode from `evaluation.json`. Do not describe the run as successful merely because it completed.

CPU-only repository contract tests can validate structure and mock replay but **cannot validate GPU model behavior**. The first Colab run is an experiment, not a predetermined positive finding.
