import torch

from neural_firewall.model_interface import build_qwen_adapter
from neural_firewall.activation_extractor import ActivationExtractor

MODEL = "Qwen/Qwen2.5-3B-Instruct"
LAYERS = {19, 20, 21, 22}

prompts = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "What is two plus two?",
    "Describe the water cycle.",
]

adapter = build_qwen_adapter(
    MODEL,
    dtype="auto",
)

extractor = ActivationExtractor(
    adapter,
    LAYERS,
    pooling="last_token",
)

result = extractor.extract_batch(prompts)

print("\nBATCH EXTRACTION TEST")
print("=" * 60)

for layer in sorted(result.pooled):
    x = result.pooled[layer]

    print(
        f"layer {layer}: "
        f"shape={tuple(x.shape)} "
        f"dtype={x.dtype}"
    )

    assert x.shape == (len(prompts), 2048)

print("\nPASS")