"""Native MLX runtime for Qwen3.5's neuron and attention-head explorer.

Uses the same unquantized Qwen/Qwen3.5-4B weights as the original runtime.
MLX-LM sanitizes the checkpoint's norm offsets and convolution layout. The
shipped indexes describe that checkpoint, not an interchangeable quantized model.
"""
import functools
import json
import os
import sqlite3
import threading

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from transformers import AutoTokenizer

from runtime_mlx_core import CoreMixin
from runtime_mlx_heads import HeadsMixin

MODEL = os.environ.get("EXPLORER_MODEL", "Qwen/Qwen3.5-4B")
NO_MODEL = os.environ.get("EXPLORER_NO_MODEL", "").lower() not in {"", "0", "false", "no"}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)


def _locked(fn):
    @functools.wraps(fn)
    def locked(self, *args, **kwargs):
        with self.lock:
            return fn(self, *args, **kwargs)
    return locked


class CallHook(nn.Module):
    """An instance-local hook; no global monkey patching or parameter mutation."""
    def __init__(self, module, before=None, after=None):
        super().__init__()
        self.inner = module
        self._before, self._after = before, after

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            inner = self.get("inner")
            if inner is None or name == "inner":
                raise AttributeError(name) from None
            return getattr(inner, name)

    def __call__(self, *args, **kwargs):
        if self._before is not None:
            replacement = self._before(self.inner, args, kwargs)
            if replacement is not None:
                args, kwargs = replacement
        out = self.inner(*args, **kwargs)
        if self._after is not None:
            self._after(self.inner, args, out)
        return out


class Runtime(CoreMixin, HeadsMixin):
    backend = "mlx"
    REFERENCE_TEXT = REFERENCE_TEXT

    def __init__(self, stats_dir):
        self.stats_dir = stats_dir
        os.makedirs(CACHE, exist_ok=True)
        # Separate computed connections from the PyTorch runtime's SQLite cache.
        self._db = sqlite3.connect(os.path.join(CACHE, "connections_mlx.sqlite"), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS con (key TEXT PRIMARY KEY, val TEXT)")
        self._db.execute("CREATE TABLE IF NOT EXISTS headcon (key TEXT PRIMARY KEY, val TEXT)")
        self._db_lock = threading.Lock()
        self.lock = threading.RLock()
        self._con_idx = self._lens_idx = self._steer_idx = None
        self.model = self.tok = self.lm = None
        self.status, self.error, self.dev = "idle", None, "mlx"
        self.resid_p50 = {}
        self._push = self._heads = self._read = None
        self._capture_all = self._capture_full = self._capture_heads = None
        self._capture_from = 0
        self._seen, self._con_cache, self._lens_cache = {}, {}, {}
        self.heads = None

    @staticmethod
    def _numpy(value):
        if isinstance(value, mx.array):
            value = value.astype(mx.float32)
            mx.eval(value)
        return np.asarray(value)

    @staticmethod
    def weight(module):
        while isinstance(module, CallHook):
            module = module.inner
        if hasattr(module, "scales"):
            return mx.dequantize(module.weight, module.scales, module.biases,
                                 group_size=module.group_size, bits=module.bits).astype(mx.float32)
        return module.weight.astype(mx.float32)

    def ensure_loaded(self):
        if self.model is not None:
            return
        if NO_MODEL:
            raise RuntimeError("live model disabled (EXPLORER_NO_MODEL)")
        with self.lock:
            if self.model is not None:
                return
            self.status, self.error = "loading", None
            try:
                model, tokenizer = load(MODEL, tokenizer_config={"trust_remote_code": False})
                text = getattr(model, "language_model", model)
                lm = text.model
                # Cached lenses, layouts and activation ranges are checkpoint-specific.
                args = text.args
                expected = (32, 2560, 9216, 248320)
                actual = (args.num_hidden_layers, args.hidden_size, args.intermediate_size, args.vocab_size)
                if actual != expected or getattr(args, "num_experts", 0):
                    raise ValueError(f"Explorer caches require dense Qwen3.5-4B architecture {expected}; got {actual}")
                if any(hasattr(layer.mlp.down_proj, "scales") for layer in lm.layers):
                    raise ValueError("Use unquantized Qwen3.5-4B: shipped research caches are not calibrated for quantized weights")
                model.eval()
                mx.eval(model.parameters())
                # Special methods are not forwarded by MLX-LM's TokenizerWrapper.
                # Explorer uses the callable HF tokenizer as well as its chat template.
                self.tok, self.lm = getattr(tokenizer, "_tokenizer", tokenizer), lm
                self._install_hooks()
                self.model = model
                self.status = "ready"
            except Exception as exc:
                self.model = self.tok = self.lm = None
                self.status, self.error = "error", str(exc)
                raise

    def _install_hooks(self):
        for i, layer in enumerate(self.lm.layers):
            def mlp_before(mod, args, kwargs, i=i):
                return (self._capture_mlp(i, args[0]), *args[1:]), kwargs
            layer.mlp.down_proj = CallHook(layer.mlp.down_proj, before=mlp_before)
            def norm_after(mod, args, out, i=i):
                self._capture_mlp_input(i, mod, args[0], out)
            layer.post_attention_layernorm = CallHook(layer.post_attention_layernorm, after=norm_after)
            linear = hasattr(layer, "linear_attn")
            attn = layer.linear_attn if linear else layer.self_attn
            n_heads = attn.num_v_heads if linear else attn.num_attention_heads
            def output_before(mod, args, kwargs, i=i, n_heads=n_heads):
                return (self._capture_head_output(i, mod, args[0], n_heads), *args[1:]), kwargs
            output_name = "out_proj" if linear else "o_proj"
            setattr(attn, output_name, CallHook(getattr(attn, output_name), before=output_before))
            if linear:
                def gated_before(mod, args, kwargs, i=i):
                    self._capture_gnorm(i, mod, args)
                attn.norm = CallHook(attn.norm, before=gated_before)
            def qk_before(mod, args, kwargs, i=i):
                self._capture_qk(i, mod, args[0], *args[1:], **kwargs)
            setattr(layer, "linear_attn" if linear else "self_attn", CallHook(attn, before=qk_before))
            def resid_after(mod, args, out, i=i):
                self._capture_residual(i, out)
            self.lm.layers[i] = CallHook(layer, after=resid_after)

    def down_col(self, layer, n):
        return self.weight(self.lm.layers[layer].mlp.down_proj)[:, n]

    def _logits(self, hidden):
        text = getattr(self.model, "language_model", self.model)
        if text.args.tie_word_embeddings:
            return self.lm.embed_tokens.as_linear(hidden)
        return text.lm_head(hidden)

    def con_index(self):
        if self._con_idx is None:
            path = os.path.join(CACHE, "connections.npz")
            if os.path.exists(path):
                with np.load(path) as z:
                    self._con_idx = {key: z[key] for key in z.files}
                self._con_idx["k"] = int(self._con_idx["k"])
            else:
                self._con_idx = False
        return self._con_idx or None

    def lens_index(self):
        if self._lens_idx is None:
            path = os.path.join(CACHE, "lens_index.npz")
            if os.path.exists(path):
                with np.load(path) as z:
                    self._lens_idx = {"up": z["up"], "down": z["down"]}
        return self._lens_idx

    def steer_index(self):
        if self._steer_idx is None:
            path = os.path.join(CACHE, "steer_index.npz")
            if os.path.exists(path):
                with np.load(path) as z:
                    self._steer_idx = dict(z)
            else:
                self._steer_idx = False
        return self._steer_idx or None

    def vocab_strings(self):
        if getattr(self, "_vocab", None) is None:
            path = os.path.join(CACHE, "vocab.json")
            if os.path.exists(path):
                with open(path) as f:
                    self._vocab = json.load(f)
            else:
                if self.tok is None:
                    if NO_MODEL:
                        raise RuntimeError("cache-only mode requires explorer/cache/vocab.json")
                    self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
                self._vocab = [self.tok.decode([i]) for i in range(len(self.tok))]
                with open(path, "w") as f:
                    json.dump(self._vocab, f)
        return self._vocab

    @_locked
    def lens(self, layer, n, k=16):
        idx = self.lens_index()
        if idx is not None and k <= idx["up"].shape[-1]:
            vocab = self.vocab_strings()
            out = {side: [vocab[t] for t in idx[side][layer, n, :k]] for side in ("up", "down")}
            st = self.steer_index()
            if st is not None:
                out.update(steer_up=[vocab[t] for t in st["up"][layer, n, :k]],
                           steer_down=[vocab[t] for t in st["down"][layer, n, :k]],
                           kl_up=round(float(st["up_kl"][layer, n]), 3),
                           kl_down=round(float(st["down_kl"][layer, n]), 3))
            return out
        self.ensure_loaded()
        key = (layer, n, k)
        if key not in self._lens_cache:
            d = self.down_col(layer, n)[None, None]
            out = {}
            for sign, side in ((1, "up"), (-1, "down")):
                logits = self._logits(self.lm.norm(sign * d))[0, 0].astype(mx.float32)
                ids = np.argsort(-self._numpy(logits))[:k]
                out[side] = [self.tok.decode([int(t)]) for t in ids]
            self._lens_cache[key] = out
        return self._lens_cache[key]

    @_locked
    def connections(self, layer, n, k=24, lens_k=5):
        key = (layer, n, k, lens_k)
        if key in self._con_cache:
            return self._con_cache[key]
        ci = self.con_index()
        if ci is not None and k <= ci["k"]:
            out = {side: [{"layer": int(ci[f"{side}_layer"][layer, n, j]),
                           "n": int(ci[f"{side}_n"][layer, n, j]),
                           "z": round(float(ci[f"{side}_z"][layer, n, j]), 2),
                           "sign": int(ci[f"{side}_sign"][layer, n, j])}
                          for j in range(k) if ci[f"{side}_z"][layer, n, j] > -1e4]
                   for side in ("down", "up")}
        else:
            dkey = f"{layer}:{n}:{k}:{lens_k}"
            with self._db_lock:
                row = self._db.execute("SELECT val FROM con WHERE key = ?", (dkey,)).fetchone()
            if row:
                return json.loads(row[0])
            self.ensure_loaded()
            out = {"down": [], "up": []}
            d = self.down_col(layer, n)
            me = self.lm.layers[layer]
            # MLX-LM already shifted the HF norm offset during checkpoint sanitization.
            scale = self.weight(me.post_attention_layernorm)
            rg = self.weight(me.mlp.gate_proj)[n] * scale
            ru = self.weight(me.mlp.up_proj)[n] * scale
            for j, other in enumerate(self.lm.layers):
                if j == layer:
                    continue
                if j > layer:
                    v = d * self.weight(other.post_attention_layernorm)
                    g, u = self.weight(other.mlp.gate_proj) @ v, self.weight(other.mlp.up_proj) @ v
                    side = "down"
                else:
                    w = self.weight(other.mlp.down_proj)
                    g, u = rg @ w, ru @ w
                    side = "up"
                prod = self._numpy(g * u)
                mag = np.abs(prod)
                z = (mag - mag.mean()) / (mag.std(ddof=1) + 1e-8)
                indices = np.argsort(-z)[:k]
                out[side].extend({"layer": j, "n": int(t), "z": round(float(z[t]), 2),
                                  "sign": 1 if prod[t] > 0 else -1} for t in indices)
            out = {side: sorted(items, key=lambda c: -c["z"])[:k] for side, items in out.items()}
        idx = self.steer_index() or self.lens_index()
        if lens_k:
            vocab = self.vocab_strings()
            for c in out["down"] + out["up"]:
                if idx is not None and lens_k <= idx["up"].shape[-1]:
                    c["lens"] = [vocab[t] for t in idx["up"][c["layer"], c["n"], :lens_k]]
                    c["lens_neg"] = [vocab[t] for t in idx["down"][c["layer"], c["n"], :lens_k]]
                    lg = self.lens_index()
                    if lg is not None and lg is not idx:
                        c["logit"] = [vocab[t] for t in lg["up"][c["layer"], c["n"], :lens_k]]
                        c["logit_neg"] = [vocab[t] for t in lg["down"][c["layer"], c["n"], :lens_k]]
                else:
                    ln = self.lens(c["layer"], c["n"], lens_k)
                    c["lens"], c["lens_neg"] = ln["up"], ln["down"]
        if len(self._con_cache) > 2000:
            self._con_cache.clear()
        self._con_cache[key] = out
        return out

    @_locked
    def build_lens_index(self, path, k=12, chunk=64):
        self.ensure_loaded()
        layers, neurons = len(self.lm.layers), self.weight(self.lm.layers[0].mlp.down_proj).shape[1]
        up, down = (np.zeros((layers, neurons, k), np.int32) for _ in range(2))
        for li, layer in enumerate(self.lm.layers):
            self.status = f"indexing lens {li}/{layers}"
            w = self.weight(layer.mlp.down_proj)
            for start in range(0, neurons, chunk):
                d = w[:, start:start + chunk].T[:, None]
                logits = self._numpy(self._logits(self.lm.norm(d))[:, 0])
                up[li, start:start + len(d)] = np.argsort(-logits, axis=-1)[:, :k]
                down[li, start:start + len(d)] = np.argsort(logits, axis=-1)[:, :k]
        np.savez(path, up=up, down=down)
        self.status = "ready"
        return path
