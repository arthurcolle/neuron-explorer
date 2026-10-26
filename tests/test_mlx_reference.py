"""Optional full-attention numerical parity; requires both backends, never downloads weights.

Hybrid DeltaNet implementations have different numerical precision, so
cross-library bitwise parity is not claimed.
The native hybrid path is separately tested against its unhooked MLX model.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(importlib.util.find_spec("torch") is None,
                               reason="optional PyTorch reference backend is not installed")


def test_float32_full_attention_logits_and_neurons_match_reference(tmp_path):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from runtime_mlx import Runtime

    torch.manual_seed(421)
    config = Qwen3_5TextConfig(
        hidden_size=128, intermediate_size=256, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        vocab_size=128, linear_num_value_heads=2, linear_num_key_heads=1,
        linear_key_head_dim=128, linear_value_head_dim=128,
        full_attention_interval=1, tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 1.0, "mrope_section": [11, 11, 10]},
    )
    config._attn_implementation = "eager"
    reference = Qwen3_5ForCausalLM(config).eval()
    model = TextModel(TextModelArgs.from_dict({**config.to_dict(), "full_attention_interval": 1}))
    weights = {k: mx.array(v.detach().numpy()) for k, v in reference.state_dict().items()}
    # With no convs in this fixture, MLX-LM cannot detect HF offset RMSNorm weights.
    suffixes = ("input_layernorm.weight", "post_attention_layernorm.weight",
                "model.norm.weight", "q_norm.weight", "k_norm.weight")
    weights = {k: v + 1.0 if k.endswith(suffixes) else v for k, v in weights.items()}
    model.load_weights(list(model.sanitize(weights).items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    runtime = Runtime(str(tmp_path))
    runtime.model, runtime.lm = model, model.model
    runtime._install_hooks()
    tokens = [1, 3, 8, 13, 21, 34]
    captures = {}
    hooks = []
    for index, layer in enumerate(reference.model.layers):
        def capture(module, args, index=index):
            captures[index] = args[0][0].detach().float().numpy().copy()
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(capture))
    try:
        with torch.inference_mode():
            expected = reference(torch.tensor([tokens]), use_cache=False).logits[0].float().numpy()
        actual = np.asarray(model(mx.array([tokens]))[0].astype(mx.float32))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
        acts = runtime.full_activations(tokens)
        for layer in range(4):
            np.testing.assert_allclose(acts[layer], captures[layer], rtol=3e-4, atol=2e-5)
    finally:
        for hook in hooks:
            hook.remove()
