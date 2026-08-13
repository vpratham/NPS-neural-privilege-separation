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

batch_result = extractor.extract_batch(prompts)

print("\nBATCH VS SINGLE")
print("=" * 60)

for i, prompt in enumerate(prompts):
    single = extractor.extract(prompt)

    for layer in sorted(LAYERS):
        a = batch_result.pooled[layer][i].float()
        b = single.pooled[layer][0].float()

        max_abs = (a - b).abs().max().item()
        mean_abs = (a - b).abs().mean().item()

        cosine = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
            dim=1,
        ).item()

        print(
            f"prompt {i+1}, layer {layer}: "
            f"cosine={cosine:.10f}, "
            f"max_abs={max_abs:.8e}, "
            f"mean_abs={mean_abs:.8e}"
        )