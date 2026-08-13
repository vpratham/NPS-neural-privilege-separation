import torch

from neural_firewall.model_interface import build_qwen_adapter
from neural_firewall.activation_extractor import ActivationExtractor

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYERS = [19, 20, 21, 22]
PROMPT = "What is the capital of France?"

adapter = build_qwen_adapter(MODEL, dtype="auto")
extractor = ActivationExtractor(adapter, LAYERS)

# ------------------------------------------------------------
# A: production extractor
# ------------------------------------------------------------

production = extractor.extract(PROMPT)

# ------------------------------------------------------------
# B: direct reference-style layer-input capture
# ------------------------------------------------------------

reference = {}

handles = []

for layer_idx in LAYERS:
    layer = adapter.get_decoder_layer(layer_idx)

    def make_hook(idx):
        def hook(module, inputs):
            reference[idx] = inputs[0].detach().clone()
        return hook

    handles.append(
        layer.register_forward_pre_hook(make_hook(layer_idx))
    )

inputs = adapter.tokenize(PROMPT)

with torch.no_grad():
    adapter.model(**inputs)

for h in handles:
    h.remove()

# ------------------------------------------------------------
# Compare last valid token
# ------------------------------------------------------------

attention_mask = inputs["attention_mask"]
last_idx = int(attention_mask.sum(dim=1)[0].item() - 1)

print("\nEXTRACTION EQUIVALENCE")
print("=" * 60)

for layer in LAYERS:

    prod = production.pooled[layer][0].float()
    ref = reference[layer][0, last_idx].float()

    max_abs = (prod - ref).abs().max().item()
    mean_abs = (prod - ref).abs().mean().item()

    cosine = torch.nn.functional.cosine_similarity(
        prod.unsqueeze(0),
        ref.unsqueeze(0),
        dim=1,
    ).item()

    print(
        f"layer {layer}: "
        f"cosine={cosine:.10f} "
        f"max_abs={max_abs:.8e} "
        f"mean_abs={mean_abs:.8e}"
    )
