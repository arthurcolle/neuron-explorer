"""Offline MLX core checks: tiny randomly initialized LM, no model downloads."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.llama import Model, ModelArgs

# Load only this mixin, not the production runtime/model loader.
_spec = importlib.util.spec_from_file_location("mlx_core_under_test", Path(__file__).resolve().parents[1] / "explorer/runtime_mlx_core.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CoreMixin = _mod.CoreMixin


class Tokenizer:
    eos_token_id = 1000  # Tiny vocabulary cannot emit EOS unless a test requests it.
    unk_token_id = 0

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(c) % 29 + 1 for c in text])

    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": [1] + self("".join(m["content"] for m in messages)).input_ids + [2]}

    def convert_tokens_to_ids(self, token):
        return self.eos_token_id

    def decode(self, ids, **kwargs):
        return "".join(chr(97 + int(t)) for t in ids)


class CaptureMLP(nn.Module):
    def __init__(self, mlp, owner, index):
        super().__init__()
        self.inner = mlp
        # Avoid registering the runtime in the parameter tree.
        object.__setattr__(self, "owner", owner)
        self.index = index

    @property
    def down_proj(self):
        return self.inner.down_proj

    @property
    def gate_proj(self):
        return self.inner.gate_proj

    def __call__(self, x):
        h = nn.silu(self.inner.gate_proj(x)) * self.inner.up_proj(x)
        return self.inner.down_proj(self.owner._capture_mlp(self.index, h))


class CaptureNorm(nn.Module):
    def __init__(self, norm, owner, index):
        super().__init__()
        self.inner = norm
        object.__setattr__(self, "owner", owner)
        self.index = index

    def __call__(self, x):
        y = self.inner(x)
        self.owner._capture_mlp_input(self.index, self.inner, (x,), y)
        return y


class CaptureLayer(nn.Module):
    def __init__(self, layer, owner, index):
        super().__init__()
        self.inner = layer
        object.__setattr__(self, "owner", owner)
        self.index = index

    @property
    def mlp(self):
        return self.inner.mlp

    @property
    def use_sliding(self):
        return self.inner.use_sliding

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        self.owner._capture_residual(self.index, out)
        return out


class RecordingModel(nn.Module):
    def __init__(self, model, owner):
        super().__init__()
        self.inner = model
        object.__setattr__(self, "owner", owner)

    @property
    def layers(self):
        return self.inner.layers

    def __call__(self, ids, cache=None):
        self.owner.calls.append((np.array(ids).tolist(), id(cache), self.owner._push))
        if self.owner.fail_model:
            raise RuntimeError("fixture forward failure")
        return self.inner(ids, cache=cache)


class TinyRuntime(CoreMixin):
    REFERENCE_TEXT = "calibrate"

    def __init__(self, directory):
        self.stats_dir = directory
        self.tok = Tokenizer()
        self.lock = threading.RLock()
        self.resid_p50 = {0: 2.0, 1: 2.0}
        self._push = self._heads = self._read = None
        self._capture_all = self._capture_full = self._capture_heads = None
        self._capture_from = 0
        self._seen = {}
        self.heads = None
        self.status = "ready"
        self.calls = []
        self.fail_model = False
        mx.random.seed(17)
        model = Model(ModelArgs(model_type="llama", hidden_size=8, num_hidden_layers=2,
                                intermediate_size=12, num_attention_heads=2, num_key_value_heads=2,
                                rms_norm_eps=1e-5, vocab_size=32))
        for i, layer in enumerate(model.layers):
            layer.mlp = CaptureMLP(layer.mlp, self, i)
            layer.post_attention_layernorm = CaptureNorm(layer.post_attention_layernorm, self, i)
            model.layers[i] = CaptureLayer(layer, self, i)
        self.lm = model.model
        self.model = RecordingModel(model, self)
        mx.eval(model.parameters())

    def ensure_loaded(self):
        pass

    def weight(self, module):
        if isinstance(module, nn.QuantizedLinear):
            return mx.dequantize(module.weight, module.scales, module.biases,
                                 group_size=module.group_size, bits=module.bits).astype(mx.float32)
        return module.weight.astype(mx.float32)

    def down_col(self, layer, n):
        return self.weight(self.lm.layers[layer].mlp.down_proj)[:, n]

    @staticmethod
    def _numpy(value):
        mx.eval(value)
        return np.array(value.astype(mx.float32))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rt = TinyRuntime(self.tmp.name)

    def assert_clean(self):
        for flag in ("_push", "_heads", "_read", "_capture_all", "_capture_full", "_capture_heads"):
            self.assertIsNone(getattr(self.rt, flag), flag)
        self.assertEqual(self.rt._capture_from, 0)
        self.assertNotIn("want_resid", self.rt._seen)

    def test_capture_is_pre_intervention_and_immutable(self):
        rt = self.rt
        h = mx.array([[[1., 2., 3.], [-1., 4., -3.]]])
        rt._capture_full, rt._capture_all = {}, {}
        rt._read = (0, 1)
        rt._push = [(0, 1, "add", mx.array([3.])), (0, None, "mul", mx.array([2.]))]
        out = rt._capture_mlp(0, h)
        np.testing.assert_array_equal(np.array(out), [[[2, 10, 6], [-2, 14, -6]]])
        np.testing.assert_array_equal(np.array(h), [[[1, 2, 3], [-1, 4, -3]]])
        np.testing.assert_array_equal(np.array(rt._capture_full[0]), np.array(h[0]))
        np.testing.assert_array_equal(np.array(rt._seen["act"]), [2, 4])
        np.testing.assert_array_equal(np.array(rt._capture_all[0][1]), [0, 1, 0])

    def test_norm_hook_accepts_direct_array_and_tuple(self):
        rt = self.rt
        x = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8)
        norm = nn.RMSNorm(8, eps=1e-5)
        rt._capture_heads = {"x": {}, "rms": {}}
        for inp in (x, (x,)):
            rt._capture_mlp_input(0, norm, inp, norm(x))
            self.assertEqual(rt._capture_heads["rms"][0].shape, (3,))
            np.testing.assert_allclose(np.array(rt._capture_heads["rms"][0]),
                                       np.sqrt(np.square(np.array(x[0])).mean(-1) + norm.eps), rtol=1e-6)
        rt._capture_heads = None
        self.assert_clean()

    def test_push_validation_normalization_and_row_signs(self):
        rt = self.rt
        vals = rt._push_entry(0, 2, .5, "add", (1, 0, -1))[3]
        expected = 1. / np.linalg.norm(rt._numpy(rt.down_col(0, 2)))
        np.testing.assert_allclose(np.array(vals), [expected, 0, -expected], rtol=1e-6)
        np.testing.assert_array_equal(np.array(rt._push_entry(0, None, 2, "mul", (1, 0, -1))[3]), [2, 1, -2])
        for l, n, d, m in [(-1, 0, 1, "add"), (2, 0, 1, "add"), (0, 12, 1, "add"),
                            (0, -1, 1, "mul"), (True, 0, 1, "mul"), (0, 0, float("nan"), "add"),
                            (0, 0, float("inf"), "mul"), (0, None, 1, "add"), (0, 0, 1, "other")]:
            with self.subTest(args=(l, n, d, m)), self.assertRaises(ValueError):
                rt._push_entry(l, n, d, m, (1,))
        with patch.object(rt, "down_col", return_value=mx.zeros((8,))):
            with self.assertRaisesRegex(ValueError, "zero"):
                rt._push_entry(0, 0, 1., "add", (1,))
        self.assert_clean()

    def test_read_full_and_extrema_agree(self):
        rt = self.rt
        ids = rt.tok("hello").input_ids
        full = rt.full_activations(ids, heads=True)
        self.assertEqual(full.shape, (2, 5, 12))
        self.assertEqual(full.dtype, np.float32)
        self.assertEqual(rt.heads["T"], 5)
        self.assertEqual(rt.heads["x"][0].shape, (5, 8))
        self.assertEqual(rt.heads["rms"][0].shape, (5,))
        self.assertIsInstance(rt.heads["x"][0], np.ndarray)
        self.assertIsInstance(rt.heads["rms"][0], np.ndarray)
        read = rt.read(0, 3, "hello", chat=False, pushes=[(0, 3, 4, "mul")])
        np.testing.assert_allclose([u["act"] for u in read["units"]], full[0, :, 3], atol=1e-6)
        for skip in (False, True):
            all_acts = rt.all_activations(ids, skip_first=skip)
            for l, (hi, ai, lo, bi) in all_acts.items():
                a = full[l, int(skip):]
                np.testing.assert_allclose(hi, a.max(0), atol=1e-6)
                np.testing.assert_allclose(lo, a.min(0), atol=1e-6)
                np.testing.assert_array_equal(ai, a.argmax(0))
                np.testing.assert_array_equal(bi, a.argmin(0))
                self.assertEqual(ai.dtype, np.int64)
        batched = rt.all_activations([ids, ids], skip_first=True)
        np.testing.assert_allclose(batched[0][0], full[0, 1:].max(0), atol=1e-6)
        self.assert_clean()

    def test_logprobs_normalized_numpy_and_prefix(self):
        rt = self.rt
        base, alt = rt.next_logprobs("hi", chat=False, prefix="!", pushes=[(0, None, 0., "mul")])
        self.assertEqual(base.shape, (32,))
        self.assertEqual(base.dtype, np.float32)
        self.assertEqual(alt.dtype, np.float32)
        self.assertAlmostEqual(float(np.exp(base).sum()), 1., places=5)
        self.assertAlmostEqual(float(np.exp(alt).sum()), 1., places=5)
        self.assertGreater(float(np.max(np.abs(base - alt))), 1e-5)
        self.assertEqual(rt.calls[-1][0][0], rt.tok("hi!").input_ids)
        _, none = rt.next_logprobs("hi", chat=False)
        self.assertIsNone(none)
        self.assert_clean()

    def test_independent_unequal_caches_match_full_prefix_oracle(self):
        rt = self.rt
        rows = [[{"role": "user", "content": text}] for text in ("a", "longer", "xy")]
        pushes = [(0, 2, .25, "add"), (1, 3, 2., "mul")]
        got = list(rt.steer_multi(pushes, rows, max_new=4))
        self.assertEqual([r for r, _ in got], [0, 1, 2] * 4 + [None])
        self.assertEqual(got[-1], (None, None))
        self.assertEqual([len(x[0][0]) for x in rt.calls[:3]], [3, 8, 4])
        cache_ids = [x[1] for x in rt.calls[:3]]
        self.assertEqual(len(set(cache_ids)), 3)
        self.assertEqual([x[1] for x in rt.calls], cache_ids * 4)
        self.assertTrue(all(len(x[0][0]) == 1 for x in rt.calls[3:]))
        # Independent uncached full-prefix oracle detects cache position/padding,
        # stale intervention, and feedback mistakes in the incremental decoder.
        for row, sign in enumerate((1, 0, -1)):
            seq = rt.messages_ids(rows[row])
            out = []
            for _ in range(4):
                rt._push = [rt._push_entry(l, n, d, m, (sign,)) for l, n, d, m in pushes]
                logits = rt.model(mx.array([seq]), cache=None)
                token = int(mx.argmax(logits[0, -1]).item())
                rt._push = None
                out.append(token)
                seq.append(token)
            self.assertEqual("".join(d for r, d in got if r == row), rt.tok.decode(out))
        self.assert_clean()

    def test_cancel_close_eos_and_failure_reset(self):
        rt = self.rt
        rows = [[{"role": "user", "content": "hello"}]] * 3
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(list(rt.steer_multi([], rows, cancel=cancel)), [(None, None)])
        self.assertFalse(rt.calls)
        cancel.clear()
        stream = rt.steer_multi([(0, 0, 2, "mul")], rows, cancel=cancel)
        self.assertEqual(next(stream)[0], 0)
        self.assert_clean()
        cancel.set()
        self.assertEqual(list(stream), [(None, None)])
        stream = rt.steer_multi([(0, 0, 2, "mul")], rows)
        next(stream)
        stream.close()
        self.assert_clean()
        rt.fail_model = True
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            list(rt.steer_multi([(0, 0, 2, "mul")], rows))
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            rt.next_logprobs("hi", pushes=[(0, 0, 2, "mul")])
        self.assert_clean()
        rt.fail_model = False
        # All rows stop immediately at the same predicted EOS.
        lp, _ = rt.next_logprobs("hello")
        rt.tok.eos_token_id = int(lp.argmax())
        self.assertEqual(list(rt.steer_multi([], rows, max_new=3)), [(None, None)])
        self.assert_clean()

    def test_evaluation_happens_under_capture_flags(self):
        rt = self.rt
        original = mx.eval
        seen = set()

        def checked(*args, **kwargs):
            for flag in ("_read", "_capture_all", "_capture_full", "_capture_heads", "_push"):
                if getattr(rt, flag) is not None:
                    seen.add(flag)
            return original(*args, **kwargs)

        with patch.object(mx, "eval", side_effect=checked):
            rt.read(0, 0, "abc", pushes=[(0, 0, 1, "mul")])
            rt.all_activations([1, 2])
            rt.full_activations([1, 2], heads=True)
        self.assertEqual(seen, {"_read", "_capture_all", "_capture_full", "_capture_heads", "_push"})
        self.assert_clean()

    def test_residual_calibration_and_stats_file(self):
        rt = self.rt
        rt.resid_p50.clear()
        value = rt.layer_resid_p50(0)
        self.assertGreater(value, 0.)
        self.assertEqual(rt.layer_resid_p50(0), value)
        np.savez(os.path.join(self.tmp.name, "q35-4b_L1.npz"), resid_norm_q10_50_90=[1., 7., 9.])
        self.assertEqual(rt.layer_resid_p50(1), 7.)
        self.assert_clean()

    def test_capture_failures_and_empty_input(self):
        rt = self.rt
        for ids in ([], [[1], []], [1.2], [-1]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                rt.all_activations(ids)
        with self.assertRaises(ValueError):
            rt.all_activations([1], skip_first=True)
        with patch.object(rt.lm.layers[0].mlp, "owner", SimpleNamespace(_capture_mlp=lambda *a: (_ for _ in ()).throw(RuntimeError("capture")))):
            for method in (lambda: rt.read(0, 0, "abc"), lambda: rt.all_activations([1, 2]),
                           lambda: rt.full_activations([1, 2], heads=True)):
                with self.assertRaisesRegex(RuntimeError, "capture"):
                    method()
                self.assert_clean()

    def test_unicode_units_match_original_semantics(self):
        rt = self.rt
        pieces = [b'A', b'\xf0\x9f', b'\x91\xa9', '\u200d'.encode(), '\U0001f4bb'.encode(),
                  '\ufe0f'.encode(), '\U0001f3fd'.encode(), b'B']
        rt.tok.decode = lambda ids, **kw: b''.join(pieces[i] for i in ids).decode('utf-8', errors='replace')
        units = rt.token_units(list(range(len(pieces))))
        self.assertEqual(units, [('A', [0]), ('\U0001f469\u200d\U0001f4bb\ufe0f\U0001f3fd', [1, 2, 3, 4, 5, 6]), ('B', [7])])

    def test_scale_schema_with_offline_dataset(self):
        rt = self.rt

        class Dataset(list):
            def shuffle(self, seed):
                return self

        fake = SimpleNamespace(load_dataset=lambda *a, **kw: Dataset([{"text": "abcdefghijklmno"}]))
        rt.tok.eos_token_id = None
        path = os.path.join(self.tmp.name, "scale.npz")
        with patch.dict(sys.modules, {"datasets": fake}):
            self.assertEqual(rt.build_act_scale(path, tokens=8, seq=4), path)
        with np.load(path) as data:
            self.assertEqual(set(data.files), {"pos", "neg", "tokens"})
            self.assertEqual(int(data["tokens"]), 8)
            self.assertEqual(data["pos"].shape, (2, 12))
            self.assertEqual(data["pos"].dtype, np.float32)
            self.assertTrue(np.all(data["pos"] >= 0))
            self.assertTrue(np.all(data["neg"] <= 0))
        self.assert_clean()


if __name__ == "__main__":
    unittest.main()
