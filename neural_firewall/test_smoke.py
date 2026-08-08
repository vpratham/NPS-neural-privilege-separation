"""
test_smoke.py

Not a unit test suite — a single end-to-end wiring check using a tiny mock
ModelAdapter (2 fake decoder layers, d_model=8) so the full pipeline
(train_offline -> ProbeBank -> NeuralFirewall.score/decide -> streaming
loop -> visualization) can be validated without downloading a real model.
Swap MockAdapter for build_qwen_adapter(...) to run against the real
validation target once weights are available.
"""

import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from neural_firewall import (
    Mode,
    NeuralFirewall,
    Policy,
    ProbeBank,
    StreamingFirewall,
    VotingStrategy,
)
from neural_firewall.model_interface import ModelAdapter
from neural_firewall.train_offline import run_training_job
from neural_firewall import visualization

D_MODEL = 8
N_LAYERS = 3
VOCAB = 50


class TinyDecoderLayer(nn.Module):
    """Small residual linear layer (no saturating nonlinearity) — keeps the
    synthetic trigger direction linearly separable across layers so this
    smoke test validates plumbing rather than fighting tanh saturation."""

    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.proj.weight.mul_(0.05)
            self.proj.bias.zero_()

    def forward(self, x):
        return (x + self.proj(x),)


class MockAdapter(ModelAdapter):
    """Minimal ModelAdapter implementation for wiring tests. The 'unsafe'
    concept here is entirely synthetic: hidden states get a bump along a
    fixed synthetic direction whenever the input contains token id 1
    ('trigger token'), which is what the trained probe should learn to
    detect. This validates the plumbing, not real safety semantics."""

    def __init__(self):
        self.layers = nn.ModuleList([TinyDecoderLayer(D_MODEL) for _ in range(N_LAYERS)])
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self._trigger_direction = torch.randn(D_MODEL)
        self._trigger_direction /= self._trigger_direction.norm()

    def tokenize(self, prompt: str, **kwargs):
        # toy "tokenizer": char codes mod VOCAB, token id 1 reserved as the
        # synthetic trigger, injected if the substring "TRIGGER" is present
        ids = [ord(c) % VOCAB for c in prompt[:6]] or [0]
        if "TRIGGER" in prompt:
            ids = [1] + ids
        input_ids = torch.tensor([ids])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def num_layers(self):
        return N_LAYERS

    def get_decoder_layer(self, layer_idx):
        return self.layers[layer_idx]

    def hidden_size(self):
        return D_MODEL

    def device(self):
        return torch.device("cpu")

    def _forward_all_layers(self, input_ids):
        x = self.embed(input_ids)
        has_trigger = (input_ids == 1).any(dim=1, keepdim=True).unsqueeze(-1).float()
        hs_by_layer = {}
        for i, layer in enumerate(self.layers):
            (x,) = layer(x)
            x = x + has_trigger * self._trigger_direction * 8.0  # synthetic signal, large margin
            hs_by_layer[i] = x
        return hs_by_layer

    # expose a `.model` + `__call__` so ActivationExtractor's hasattr check
    # + hook-based extraction works the same way it would for a real HF model
    @property
    def model(self):
        return self

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        hs_by_layer = self._forward_all_layers(input_ids)
        for idx, handle_layer in enumerate(self.layers):
            for hook_id, hook in handle_layer._forward_hooks.items():
                hook(handle_layer, (), (hs_by_layer[idx],))
        return hs_by_layer

    def generate_step(self, input_ids, attention_mask=None, past_key_values=None):
        hs_by_layer = self._forward_all_layers(input_ids)
        logits = torch.randn(1, input_ids.size(1), VOCAB)
        return logits, None, {k: v for k, v in hs_by_layer.items()}

    class tokenizer:
        @staticmethod
        def decode(ids):
            return "".join(chr(65 + (i % 26)) for i in ids)


def main():
    torch.manual_seed(0)
    adapter = MockAdapter()

    # --- synth a labeled dataset: prompts with/without the trigger ---
    prompts = [f"TRIGGER_{i}" for i in range(30)] + [f"benign_{i}" for i in range(30)]
    labels = np.array([1] * 30 + [0] * 30)

    from neural_firewall.activation_extractor import ActivationExtractor
    extractor = ActivationExtractor(adapter, set(range(N_LAYERS)), pooling="last_token")
    acts_by_layer = {i: [] for i in range(N_LAYERS)}
    for p in prompts:
        extracted = extractor.extract(p)
        for i in range(N_LAYERS):
            acts_by_layer[i].append(extracted.pooled[i].detach().numpy()[0])
    acts_by_layer = {i: np.stack(v) for i, v in acts_by_layer.items()}

    # --- offline training (the ONLY place fitting happens) ---
    artifact_dir = Path("/tmp/nf_smoke_artifacts")
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    directions = run_training_job(acts_by_layer, labels, "synthetic_trigger", artifact_dir)
    print(f"[train_offline] fit {len(directions)} PolicyDirection artifacts")
    for d in directions:
        print(f"  layer {d.layer_idx}: threshold={d.threshold:.3f} "
              f"achieved_recall={d.metadata['calibration_achieved_recall']:.2f} "
              f"achieved_fpr={d.metadata['calibration_achieved_fpr']:.2f}")

    # --- load artifacts at "runtime" (no retraining) ---
    bank = ProbeBank.load(artifact_dir, expected_d_model=adapter.hidden_size())
    assert bank.policies() == ["synthetic_trigger"]

    # --- BLOCK mode on a prompt-level score ---
    policy = Policy(policy_names=["synthetic_trigger"], voting=VotingStrategy.MAJORITY, mode=Mode.BLOCK)
    firewall = NeuralFirewall(adapter, bank, policy)

    unsafe_assessment = firewall.score("TRIGGER_test")
    benign_assessment = firewall.score("benign_test")
    print(f"[score] TRIGGER prompt risk={unsafe_assessment.risk_score:.3f} "
          f"exceeded={unsafe_assessment.exceeded_threshold}")
    print(f"[score] benign prompt  risk={benign_assessment.risk_score:.3f} "
          f"exceeded={benign_assessment.exceeded_threshold}")
    assert unsafe_assessment.exceeded_threshold, "trigger prompt should be flagged"
    assert not benign_assessment.exceeded_threshold, "benign prompt should NOT be flagged"

    decision = firewall.decide("TRIGGER_test")
    print(f"[decide] action={decision.action}")
    assert decision.action == "block"

    # --- WARN mode explanation ---
    warn_policy = Policy(policy_names=["synthetic_trigger"], mode=Mode.WARN)
    firewall.fit(policy=warn_policy)
    warn_decision = firewall.decide("TRIGGER_test")
    print(f"[warn] action={warn_decision.action} explanation={warn_decision.assessment.explanation}")
    assert warn_decision.action == "warn"

    # --- INTERVENE mode: suppress, then re-check that the intervened score drops ---
    intervene_policy = Policy(
        policy_names=["synthetic_trigger"],
        mode=Mode.INTERVENE,
        intervention_method="projection_removal",
        intervention_strength=1.0,
    )
    firewall.fit(policy=intervene_policy)
    intervene_decision = firewall.intervene("TRIGGER_test")
    print(f"[intervene] pre-intervention risk={intervene_decision.assessment.risk_score:.3f} "
          f"action={intervene_decision.action}")
    assert intervene_decision.intervened

    # --- streaming: DETECT mode, per-token risk trajectory ---
    detect_policy = Policy(policy_names=["synthetic_trigger"], mode=Mode.DETECT)
    firewall.fit(policy=detect_policy)
    streamer = StreamingFirewall(firewall)
    result = streamer.generate("TRIGGER_stream", max_new_tokens=5)
    print(f"[streaming/detect] generated {len(result.output_ids)} tokens, "
          f"risk trajectory: {[round(r.assessment.risk_score, 2) for r in result.trajectory]}")
    assert len(result.trajectory) == 5

    # --- streaming: BLOCK mode should halt early since trigger persists via kv-cache prefix ---
    block_policy = Policy(policy_names=["synthetic_trigger"], mode=Mode.BLOCK)
    firewall.fit(policy=block_policy)
    streamer = StreamingFirewall(firewall)
    result = streamer.generate("TRIGGER_stream_block", max_new_tokens=5)
    print(f"[streaming/block] stopped_early={result.stopped_early} reason={result.stop_reason} "
          f"n_output_tokens={len(result.output_ids)}")

    # --- visualization: confirm figures build without error ---
    fig = visualization.plot_streaming_result(result, threshold=0.5)
    fig.savefig("/tmp/nf_smoke_risk_timeline.png")
    print("[visualization] saved /tmp/nf_smoke_risk_timeline.png")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
