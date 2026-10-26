"""Integration checks against real MLX Qwen3.5 modules with small random weights."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
mx = pytest.importorskip("mlx.core")
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from runtime_mlx import Runtime


def tiny_runtime(tmp_path):
    config = dict(model_type="qwen3_5_text", hidden_size=128, intermediate_size=256,
                  num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=64, vocab_size=128, linear_num_value_heads=2,
                  linear_num_key_heads=1, linear_key_head_dim=128,
                  linear_value_head_dim=128, full_attention_interval=4,
                  tie_word_embeddings=True,
                  rope_parameters={"type": "default", "rope_theta": 10000,
                                   "partial_rotary_factor": 1.0})
    mx.random.seed(71)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=config))
    model.eval()
    mx.eval(model.parameters())
    runtime = Runtime(str(tmp_path))
    runtime.model, runtime.lm = model, model.language_model.model
    return runtime


def test_wrappers_are_transparent_and_capture_heads(tmp_path):
    runtime = tiny_runtime(tmp_path)
    ids = mx.array([[1, 2, 3, 4]])
    before = np.asarray(runtime.model(ids).astype(mx.float32))
    runtime._install_hooks()
    after = np.asarray(runtime.model(ids).astype(mx.float32))
    np.testing.assert_array_equal(before, after)
    acts = runtime.full_activations([1, 2, 3, 4], heads=True)
    assert acts.shape == (4, 4, 256)
    assert np.isfinite(acts).all()
    for layer in range(4):
        assert runtime.heads["x"][layer].shape == (4, 128)
        assert runtime.heads["rms"][layer].shape == (4,)
        pattern = runtime.head_pattern(layer, 0)
        assert pattern.shape == (4, 4)
        np.testing.assert_allclose(np.triu(pattern, 1), 0, atol=1e-7)
        np.testing.assert_allclose(np.abs(pattern).sum(-1), 1, atol=1e-5)
        np.testing.assert_allclose(runtime.head_row(layer, 0, 2), pattern[2, :3], atol=1e-5)
    assert runtime.heads["q"][0]["kind"] == "linear"
    assert runtime.heads["q"][3]["kind"] == "full"


def test_actual_qwen_mlp_and_head_interventions(tmp_path):
    runtime = tiny_runtime(tmp_path)
    runtime._install_hooks()
    ids = mx.array([[1, 2, 3]])
    baseline = np.asarray(runtime.model(ids).astype(mx.float32))
    runtime._push = [(1, 3, "add", mx.array([8.0]))]
    intervened = np.asarray(runtime.model(ids).astype(mx.float32))
    runtime._push = None
    assert np.max(np.abs(intervened - baseline)) > 1e-5
    runtime._heads = [(0, None, 0.0)]
    ablated = np.asarray(runtime.model(ids).astype(mx.float32))
    runtime._heads = None
    assert np.max(np.abs(ablated - baseline)) > 1e-5
    np.testing.assert_array_equal(np.asarray(runtime.model(ids).astype(mx.float32)), baseline)
