"""Offline tiny-array tests; run with Python 3.12 and the installed MLX stack.

No model downloads, tokenizer loading, external inference or torch dependency.
"""
import importlib.util
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace
import unittest

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel, TextModelArgs


spec = importlib.util.spec_from_file_location(
    "mlx_heads_under_test", Path(__file__).resolve().parents[1] / "explorer/runtime_mlx_heads.py")
heads_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heads_module)


def caps():
    return {key: {} for key in ("q", "writes", "y", "s", "x", "rms")}


class Harness(heads_module.HeadsMixin):
    def __init__(self, lm=None):
        self.lock = threading.RLock()
        self._db_lock = threading.Lock()
        self._db = sqlite3.connect(":memory:")
        self.lm = lm
        self.heads = caps()
        self._capture_heads = None
        self._heads = None

    def ensure_loaded(self):
        if self.lm is None:
            raise AssertionError("test unexpectedly tried loading a model")

    @staticmethod
    def _numpy(x):
        return np.array(x.astype(mx.float32))

    @staticmethod
    def weight(mod):
        if isinstance(mod, nn.QuantizedLinear):
            return mx.dequantize(mod.weight, mod.scales, mod.biases,
                                 group_size=mod.group_size, bits=mod.bits).astype(mx.float32)
        return mod.weight.astype(mx.float32)

    def down_col(self, layer, n):
        return self.weight(self.lm.layers[layer].mlp.down_proj)[:, n]


def tiny_model():
    mx.random.seed(7)
    args = TextModelArgs(hidden_size=8, intermediate_size=6, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                         linear_num_key_heads=2, linear_num_value_heads=4,
                         linear_key_head_dim=2, linear_value_head_dim=2,
                         linear_conv_kernel_dim=3, full_attention_interval=2,
                         vocab_size=16,
                         rope_parameters={"type": "default", "rope_theta": 10000,
                                          "partial_rotary_factor": 1.0})
    lm = Qwen3_5TextModel(args)
    lm.train()  # tiny key dims use the installed recurrence, not its Dk>=32 Metal kernel
    for layer in lm.layers:
        layer.input_layernorm.weight = mx.linspace(0.6, 1.7, 8)
        layer.post_attention_layernorm.weight = mx.linspace(0.5, 1.8, 8)
    return lm


def capture_layer(h, i, residual):
    """Independent forward computation; compare against the installed module."""
    layer = h.lm.layers[i]
    a = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
    x = layer.input_layernorm(residual)
    B, T, _ = x.shape
    full = hasattr(layer, "self_attn")
    h._capture_qk(i, a, x, mask="causal" if full else None)
    if full:
        q, gate = mx.split(a.q_proj(x).reshape(B, T, a.num_attention_heads, -1), 2, axis=-1)
        q = a.rope(a.q_norm(q).transpose(0, 2, 1, 3))
        k = a.rope(a.k_norm(a.k_proj(x).reshape(B, T, a.num_key_value_heads, -1)).transpose(0, 2, 1, 3))
        v = a.v_proj(x).reshape(B, T, a.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=a.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3) * mx.sigmoid(gate)
        proj = a.o_proj
    else:
        qkv = a.in_proj_qkv(x)
        conv_in = mx.pad(qkv, [(0, 0), (a.conv_kernel_size - 1, 0), (0, 0)])
        q, k, v = mx.split(nn.silu(a.conv1d(conv_in)), [a.key_dim, 2 * a.key_dim], axis=-1)
        q = q.reshape(B, T, a.num_k_heads, a.head_k_dim)
        k = k.reshape(B, T, a.num_k_heads, a.head_k_dim)
        v = v.reshape(B, T, a.num_v_heads, a.head_v_dim)
        inv = a.head_k_dim ** -0.5
        q = inv ** 2 * mx.fast.rms_norm(q, None, 1e-6)
        k = inv * mx.fast.rms_norm(k, None, 1e-6)
        raw, _ = gated_delta_update(q, k, v, a.in_proj_a(x), a.in_proj_b(x),
                                   a.A_log, a.dt_bias, use_kernel=False)
        z = a.in_proj_z(x).reshape(B, T, a.num_v_heads, a.head_v_dim)
        h._capture_gnorm(i, a.norm, (raw, z))
        y = a.norm(raw, z)
        proj = a.out_proj
    y = y.reshape(B, T, -1)
    unchanged = h._capture_head_output(i, proj, y, h.n_heads(i))
    np.testing.assert_allclose(h._numpy(proj(unchanged)), h._numpy(a(x, mask="causal" if full else None)), rtol=3e-5, atol=2e-6)
    r = residual + proj(y)
    h._capture_heads["x"][i] = h._numpy(layer.post_attention_layernorm(r)[0])
    rn = h._numpy(r[0])
    h._capture_heads["rms"][i] = np.sqrt((rn * rn).mean(axis=-1) + layer.post_attention_layernorm.eps)
    return r + layer.mlp(layer.post_attention_layernorm(r))


class PatternTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.addCleanup(self.h._db.close)

    def test_full_grouped_softmax_and_causality(self):
        rng = np.random.default_rng(3)
        c = {"kind": "full", "q": rng.normal(size=(4, 4, 3)).astype("f"),
             "k": rng.normal(size=(2, 4, 3)).astype("f"), "scale": 3 ** -0.5}
        self.h.heads["q"][0] = c
        patterns = self.h.layer_patterns(0)
        self.assertEqual(patterns.shape, (4, 4, 4))
        np.testing.assert_allclose(patterns.sum(-1), 1, atol=1e-7)
        for head in range(4):
            np.testing.assert_array_equal(np.triu(patterns[head], 1), 0)
            np.testing.assert_allclose(patterns[head], self.h.head_pattern(0, head))
            for t in range(4):
                logits = c["k"][head // 2, :t + 1] @ c["q"][head, t] * c["scale"]
                expected = np.exp(logits - logits.max())
                expected /= expected.sum()
                np.testing.assert_allclose(self.h.head_row(0, head, t), expected, atol=1e-7)
                np.testing.assert_allclose(patterns[head, t, :t + 1], self.h.raw_row(0, head, t), atol=1e-7)

    def test_signed_deltanet_matches_forward_recurrence(self):
        q = np.array([[[.5, .2], [.1, -.7], [-.4, .6]]], dtype="f")
        k = np.array([[[1, 0], [.6, .8], [0, 1]]], dtype="f")
        v = np.array([[[1, 2], [3, -1], [-2, 4]], [[-1, 3], [2, 1], [0, -2]]], dtype="f")
        c = {"kind": "linear", "q": q, "k": k, "rep": 2, "v": v,
             "beta": np.array([[.8, .7, .9], [.3, .5, .6]], dtype="f"),
             "logd": np.array([[-.2, -.1, -.3], [-.5, -.2, -.1]], dtype="f")}
        self.h.heads["q"][0] = c
        patterns = self.h.layer_patterns(0)
        self.assertTrue(np.any(patterns < 0))
        np.testing.assert_allclose(np.abs(patterns).sum(-1), 1, atol=1e-7)
        for head in range(2):
            state = np.zeros((2, 2), dtype="f")
            np.testing.assert_allclose(patterns[head], self.h.head_pattern(0, head), atol=1e-7)
            np.testing.assert_array_equal(np.triu(patterns[head], 1), 0)
            for t in range(3):
                state *= np.exp(c["logd"][head, t])
                state += np.outer(k[0, t], c["beta"][head, t] * (v[head, t] - k[0, t] @ state))
                row = self.h.raw_row(0, head, t)
                np.testing.assert_allclose(row @ v[head, :t + 1], q[0, t] @ state, atol=1e-7)
                np.testing.assert_allclose(patterns[head, t, :t + 1], self.h.head_row(0, head, t), atol=1e-7)
        other = caps()
        other["q"][0] = {**c, "beta": np.zeros_like(c["beta"])}
        np.testing.assert_array_equal(self.h.layer_patterns(0, other), 0)
        np.testing.assert_array_equal(self.h.raw_row(0, 1, 2, other), 0)
        with self.assertRaises(IndexError):
            self.h.head_row(0, 0, -1)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness(tiny_model())
        self.addCleanup(self.h._db.close)

    def test_disabled_hooks_and_batch_steering(self):
        h = self.h
        h._capture_qk(0, None, None)
        h._capture_gnorm(0, None, None)
        x = mx.arange(48, dtype=mx.float32).reshape(2, 3, 8)
        self.assertIs(h._capture_head_output(0, None, x, 4), x)
        h._heads = [(0, None, 2), (0, 1, 0), (0, 2, -0.5), (1, None, 0)]
        out = h._numpy(h._capture_head_output(0, None, x, 4)).reshape(2, 3, 4, 2)
        expected = h._numpy(x).reshape(2, 3, 4, 2) * np.array([2, 0, -1, 2])[None, None, :, None]
        np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(h._numpy(x), np.arange(48).reshape(2, 3, 8))

    def test_fresh_capture_guards(self):
        h = self.h
        h._capture_heads = caps()
        a = h.lm.layers[0].linear_attn
        x = mx.ones((1, 3, 8))
        with self.assertRaisesRegex(ValueError, "cache=None"):
            h._capture_qk(0, a, x, None, object())
        with self.assertRaisesRegex(ValueError, "one nonempty"):
            h._capture_qk(0, a, mx.ones((2, 3, 8)))
        with self.assertRaisesRegex(ValueError, "padding"):
            h._capture_qk(0, a, x, mask=mx.array([[False, True, True]]))
        with self.assertRaisesRegex(ValueError, "padding"):
            h._capture_qk(1, h.lm.layers[1].self_attn, x, mask=mx.ones((3, 3), dtype=mx.bool_))
        with self.assertRaisesRegex(ValueError, "causal mask"):
            h._capture_qk(1, h.lm.layers[1].self_attn, x)
        self.assertFalse(h._capture_heads["q"])
        # Ordinary explicit causal masks are supported as well as 'causal'.
        h._capture_qk(1, h.lm.layers[1].self_attn, x, mask=mx.tril(mx.ones((3, 3), dtype=mx.bool_)))
        self.assertEqual(h._capture_heads["q"][1]["q"].shape, (4, 3, 4))

    def test_real_mlx_captures_and_drive_decomposition(self):
        h = self.h
        h._capture_heads = h.heads
        residual = mx.array(np.random.default_rng(11).normal(size=(1, 4, 8)).astype("f"))
        for i in range(3):
            residual = capture_layer(h, i, residual)
        for i, layer in enumerate(h.lm.layers):
            q = h.heads["q"][i]
            a = layer.self_attn if hasattr(layer, "self_attn") else layer.linear_attn
            W = h._numpy(h.weight(a.o_proj if q["kind"] == "full" else a.out_proj))
            y = h.heads["y"][i]
            self.assertEqual(h.heads["writes"][i].shape, (4, 4))
            for head in range(4):
                kv = head // 2 if q["kind"] == "full" else head
                for t in range(4):
                    sc = q["s"][head, t] if q["kind"] == "full" else h.heads["s"][i][head, t]
                    expected = (h.raw_row(i, head, t) @ q["v"][kv, :t + 1]) * sc
                    np.testing.assert_allclose(y[head, t], expected, rtol=2e-4, atol=2e-6)
                hd = y.shape[-1]
                expected = np.linalg.norm(y[head] @ W[:, head * hd:(head + 1) * hd].T, axis=1)
                np.testing.assert_allclose(h.heads["writes"][i][:, head], expected, rtol=1e-6, atol=1e-7)
        result = h.attn_drive(2, 1, 3, top=12)
        lazy_caps = {**h.heads, "x": {l: mx.array(x) for l, x in h.heads["x"].items()},
                     "rms": {l: mx.array(x) for l, x in h.heads["rms"].items()}}
        self.assertEqual(h.attn_drive(2, 1, 3, caps=lazy_caps, top=12), result)
        self.assertEqual(set(result), {"act", "total", "heads", "src"})
        self.assertEqual(len(result["heads"]), 12)
        for l, head, amount in result["heads"]:
            self.assertAlmostEqual(sum(result["src"][(l, head)]), amount, places=5)
        # Independently differentiate the frozen-RMS neuron with respect to each
        # residual write. Nonunit norm weights catch an erroneous HF-style +1.
        layer = h.lm.layers[2]
        x = h.heads["x"][2][3].astype("d")
        rms = h.heads["rms"][2][3]
        w = h._numpy(layer.post_attention_layernorm.weight).astype("d")
        r = x * rms / w
        wg = h._numpy(layer.mlp.gate_proj.weight)[1].astype("d")
        wu = h._numpy(layer.mlp.up_proj.weight)[1].astype("d")
        def activation(res):
            normalized = res / rms * w
            g, u = wg @ normalized, wu @ normalized
            return g / (1 + np.exp(-g)) * u
        self.assertAlmostEqual(result["act"], activation(r), places=6)
        eps = 1e-4
        self.assertAlmostEqual(result["total"], (activation(r + eps * r) - activation(r - eps * r)) / (2 * eps), places=5)
        for l, head, amount in result["heads"]:
            lay = h.lm.layers[l]
            a = lay.self_attn if hasattr(lay, "self_attn") else lay.linear_attn
            W = h._numpy(h.weight(a.o_proj if hasattr(lay, "self_attn") else a.out_proj))
            y = h.heads["y"][l][head, 3]
            direction = W[:, head * y.size:(head + 1) * y.size] @ y
            finite_diff = (activation(r + eps * direction) - activation(r - eps * direction)) / (2 * eps)
            self.assertAlmostEqual(amount, finite_diff, places=5)
        # Captures describe pre-ablation outputs, as in the original runtime.
        h._heads = [(1, 0, 0)]
        a = h.lm.layers[1].self_attn
        x = mx.ones((1, 2, 16))
        out = h._numpy(h._capture_head_output(1, a.o_proj, x, 4))
        np.testing.assert_array_equal(out[..., :4], 0)
        np.testing.assert_array_equal(h.heads["y"][1][0], 1)

    def test_linear_epsilon_and_decay_rounding(self):
        h = self.h
        h._capture_heads = h.heads
        a = h.lm.layers[0].linear_attn
        a.A_log = a.A_log.astype(mx.bfloat16)
        x = mx.full((1, 3, 8), 1e-4)
        h._capture_qk(0, a, x)
        c = h.heads["q"][0]
        conv = nn.silu(a.conv1d(mx.pad(a.in_proj_qkv(x), [(0, 0), (2, 0), (0, 0)])))
        q, k, _ = mx.split(conv, [a.key_dim, 2 * a.key_dim], axis=-1)
        q = h._numpy(q).reshape(3, 2, 2)
        k = h._numpy(k).reshape(3, 2, 2)
        expected_q = q / np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + 1e-6) / 2
        expected_k = k / np.sqrt(np.mean(k * k, axis=-1, keepdims=True) + 1e-6) / np.sqrt(2)
        np.testing.assert_allclose(c["q"], expected_q.transpose(1, 0, 2), rtol=2e-6, atol=1e-8)
        np.testing.assert_allclose(c["k"], expected_k.transpose(1, 0, 2), rtol=2e-6, atol=1e-8)
        from mlx_lm.models.gated_delta import compute_g
        expected_decay = h._numpy(compute_g(a.A_log, a.in_proj_a(x), a.dt_bias)[0]).T
        np.testing.assert_allclose(np.exp(c["logd"]), expected_decay, rtol=2e-6, atol=1e-8)


class WeightAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness(tiny_model())
        self.addCleanup(self.h._db.close)

    def test_spaces_schemas_directions_and_sqlite_cache(self):
        h = self.h
        V, O, spans = h._head_bases()
        self.assertIs(h._head_bases()[0], V)
        self.assertEqual(len(spans), 12)
        for _, _, a, b in spans:
            np.testing.assert_allclose(V[:, a:b].T @ V[:, a:b], np.eye(b - a), atol=2e-6)
            np.testing.assert_allclose(O[:, a:b].T @ O[:, a:b], np.eye(b - a), atol=2e-6)
        lay = h.lm.layers[1]
        # Value read basis uses actual MLX norm scales, not 1 + scale.
        W = h._numpy(lay.self_attn.v_proj.weight)[:4] * h._numpy(lay.input_layernorm.weight)
        expected = np.linalg.qr(W.T)[0]
        _, _, a, b = spans[4]
        np.testing.assert_allclose(V[:, a:b] @ V[:, a:b].T, expected @ expected.T, atol=2e-6)
        qkb = h._qk_bases()
        self.assertIs(h._qk_bases(), qkb)
        full_keys = [hs for l, hs, _, _ in qkb["k"][1] if l == 1]
        self.assertEqual(full_keys, [[0, 1], [2, 3]])
        links = h.head_links(1, 2, k=40)  # k exceeds tiny candidate count
        self.assertEqual(set(links), {"reads", "writes", "q", "k"})
        self.assertEqual(len(links["reads"]), 6)
        self.assertEqual(len(links["writes"]), 12)
        for side, rows in links.items():
            for row in rows:
                self.assertEqual(set(row), {"layer", "n", "z"})
                self.assertTrue(np.isfinite(row["z"]))
                self.assertTrue(row["layer"] >= 1 if side == "writes" else row["layer"] < 1)
        # A cache hit needs no model, and the key cannot collide with HF entries.
        lm, h.lm = h.lm, None
        self.assertEqual(h.head_links(1, 2, k=40), links)
        h.lm = lm
        self.assertTrue(h._db.execute("SELECT key FROM headcon").fetchone()[0].startswith("mlx-heads-v1:"))
        self.assertEqual(h.head_links(0, 0)["reads"], [])
        scores = h.neuron_heads(1, 2)
        self.assertEqual(set(scores), {0, 1, 2})
        self.assertTrue(all(len(row) == 4 for row in scores.values()))
        roles = h.neuron_head_roles(1, 2, k=2)
        self.assertEqual(set(roles), {"q", "k", "v", "o"})
        for role, rows in roles.items():
            self.assertLessEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(set(row), {"layer", "head", "heads", "z"})
                self.assertEqual(row["head"], min(row["heads"]))
                self.assertTrue(row["layer"] <= 1 if role == "o" else row["layer"] > 1)
        self.assertEqual(h.neuron_head_roles(2, 0)["q"], [])
        self.assertEqual(h.neuron_head_roles(1, 0, k=0), {"q": [], "k": [], "v": [], "o": []})

    def test_quantized_weight_helper_and_zero_rows(self):
        h = self.h
        # Quantized projections must go through the parent's dequantizing helper.
        for lay in h.lm.layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(lay.mlp, name)
                # MLX's minimum quantization block is 32, so test the output hook
                # separately with a block-sized input rather than fake packing.
                self.assertEqual(h._head_weight(mod).dtype, np.float32)
        linear = nn.Linear(32, 8, bias=False).to_quantized(group_size=32, bits=4)
        h._capture_heads = caps()
        x = mx.ones((1, 2, 32))
        h._capture_head_output(0, linear, x, 4)
        W = h._numpy(h.weight(linear)).reshape(8, 4, 8)
        expected = np.stack([np.linalg.norm(np.ones((2, 8)) @ W[:, head].T, axis=1) for head in range(4)], axis=1)
        np.testing.assert_allclose(h._capture_heads["writes"][0], expected, atol=2e-6)
        h.lm.layers[1].mlp.down_proj.weight = mx.zeros_like(h.lm.layers[1].mlp.down_proj.weight)
        result = h.neuron_heads(1, 0)
        self.assertTrue(all(np.isfinite(row).all() for row in result.values()))
        h.lm.layers[1].mlp = SimpleNamespace()
        with self.assertRaisesRegex(NotImplementedError, "dense MLP"):
            h.neuron_heads(1, 0)


if __name__ == "__main__":
    unittest.main()
