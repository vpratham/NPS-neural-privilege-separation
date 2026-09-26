# Running NFW-010 in Colab

1. Create a new Colab notebook and upload/open `NFW_010_Protocol_Corrected_Tool_Result_Injection.ipynb`.
2. Select **Runtime → Change runtime type → T4 GPU** (or equivalent GPU).
3. Run the first cell, then mount Drive before the setup cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Confirm the setup cell prints the run ID and selected model names you expect **before weights download**. By default, the notebook tries both models sequentially; a development-gate failure for one model is recorded and does not prevent the next model from running. To run only Qwen 3B, set this in a cell after installation and before setup:

```python
import os
os.environ['NFW010_MODELS'] = 'qwen_3b'
os.environ['NFW010_RUN_ID'] = 'nfw010_qwen3b_only_001'
```

Then run the remaining cells in order. The default output is:

```text
MyDrive/NFW-010/nfw010_protocol_corrected_003/
  manifest.json
  tasks.json
  broker_controls.json
  model_info/<model>.json
  development_protocol_gate/<model>.json
  responses/<model>/{development,heldout}/<task>__<condition>.json
  evaluation.json
  final_report.json
  REPORT.md
```

## Reconnect and resume

Re-mount the same Drive and rerun every cell in order with the same `RUN_ID`. Immutable files are validated and existing per-response checkpoints are skipped. Do **not** delete or edit a run artifact to bypass a mismatch. If you change code, models, cards, prompts, decoding, selected model, or protocol, use a new run ID before running:

```python
import os
os.environ['NFW010_RUN_ID'] = 'nfw010_protocol_corrected_003'
```

## Development gate and warning messages

The notebook generates six development cards before held-out cards. The parser must accept all 18 development continuations for a model to enter its held-out arm. If a model responds in prose or emits malformed calls, its development results are preserved, held-out is marked skipped for that model, and the notebook continues with the next selected model. Do not convert prose claims into actions or weaken the parser after seeing held-out outputs.

The no-`HF_TOKEN` warning is informational for these public models. The notebook requests greedy decoding with `do_sample=False` and does not pass irrelevant sampling parameters (`temperature`, `top_p`, `top_k`). A model-specific BOS configuration notice or missing-token warning is informational; distinguish those from the explicit per-model development-gate status.

## Review-only mode

After a completed run, set this before executing setup to verify every immutable response and regenerate deterministic evaluation/reporting without loading models:

```python
import os
os.environ['NFW010_REVIEW_ONLY'] = '1'
```

This notebook only uses synthetic local mock state. Never replace mock tools with shell, network, email, filesystem, or production capabilities in Colab.
