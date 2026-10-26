"""Attention-head analysis for the MLX Qwen3.5 runtime.

The parent owns model loading, hooks, capture lifetime, the reentrant model lock,
SQLite, and weight/dequantization helpers. Captures and analysis live in NumPy
float32; projections, convolution, RoPE and model normalization run in MLX.
Only single-sequence, unpadded, causal, cache-free text captures are supported.
Weight-space analyses describe dense MLPs (not routed MoE experts), just as the
original runtime does. They are geometric associations, not causal attribution.
"""

from functools import wraps
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def _locked(method):
    @wraps(method)
    def locked(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return locked


def _sigmoid(x):
    # Stable for either a scalar or an array, including very negative gates.
    return np.exp(-np.logaddexp(0.0, -np.asarray(x, dtype=np.float32)))


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _unit_rows(a):
    return a / np.maximum(np.abs(a).sum(axis=-1, keepdims=True), 1e-12)


def _basis(a):
    return np.linalg.qr(a, mode="reduced")[0].astype(np.float32, copy=False)


class HeadsMixin:
    """Requires lm.layers, weight, down_col, _numpy, lock and ensure_loaded.

    The parent's pre-MLP norm hook supplies capture fields ``x`` and ``rms``;
    the callbacks here supply ``q``, ``writes``, ``y`` and ``s``. All retain the
    original runtime's axes and keys. The parent must hold the model lock while
    running hooks and clear _capture_heads in a finally block.
    """

    def _head_weight(self, module):
        return np.asarray(self._numpy(self.weight(module)), dtype=np.float32)

    def _head_norm_weight(self, module):
        # MLX sanitization already adds the HF +1. Never add it a second time.
        return np.asarray(self._numpy(module.weight), dtype=np.float32)

    def _capture_qk(self, i, mod, x, *args, **kwargs):
        if self._capture_heads is None:
            return
        mask = kwargs.get("mask", args[0] if args else None)
        cache = kwargs.get("cache", args[1] if len(args) > 1 else None)
        if cache is not None:
            raise ValueError("head capture requires a fresh pass with cache=None")
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1:
            raise ValueError("head capture requires one nonempty, unpadded sequence")
        T = x.shape[1]
        full = hasattr(mod, "q_norm")
        if full and T > 1 and mask is None:
            raise ValueError("full-attention head capture requires a causal mask")
        if isinstance(mask, str):
            if mask != "causal" or not full:
                raise ValueError("unsupported head capture mask")
        elif mask is not None:
            # Accept an explicit ordinary causal mask, but not padding, sliding
            # windows or arbitrary attention biases, which the schema cannot hold.
            m = np.asarray(mask)
            if full:
                want = np.tril(np.ones((T, T), dtype=bool))
                m = np.squeeze(m)
                if m.dtype == np.bool_:
                    valid = np.array_equal(m.reshape((T, T)), want) if m.size == T * T else False
                else:
                    valid = (m.size == T * T and
                             np.all(m.reshape(T, T)[want] == 0) and
                             np.all(np.isneginf(m.reshape(T, T)[~want])))
            else:
                valid = m.dtype == np.bool_ and m.size == T and bool(m.all())
            if not valid:
                raise ValueError("head capture does not support padding or noncausal/custom masks")

        if full:
            hd = mod.head_dim
            q, gate = mx.split(mod.q_proj(x).reshape(1, T, -1, 2 * hd), 2, axis=-1)
            q = mod.rope(mod.q_norm(q).transpose(0, 2, 1, 3))
            k = mod.rope(mod.k_norm(mod.k_proj(x).reshape(1, T, -1, hd)).transpose(0, 2, 1, 3))
            v = mod.v_proj(x).reshape(T, -1, hd).transpose(1, 0, 2)
            c = {"kind": "full", "q": self._numpy(q[0]), "k": self._numpy(k[0]),
                 "scale": float(mod.scale), "v": self._numpy(v),
                 "s": self._numpy(mx.sigmoid(gate[0]).transpose(1, 0, 2))}
        else:
            if not hasattr(mod, "in_proj_qkv"):
                raise NotImplementedError("head capture supports Qwen3.5's separate qkv/a/b projections, not fused Qwen3Next projections")
            qkv = mod.in_proj_qkv(x)
            # MLX Conv1d is [batch, time, channels], with no built-in padding here.
            pad = mx.zeros((1, mod.conv_kernel_size - 1, mod.conv_dim), dtype=qkv.dtype)
            conv = nn.silu(mod.conv1d(mx.concatenate([pad, qkv], axis=1)))
            q, k, v = mx.split(conv, [mod.key_dim, 2 * mod.key_dim], axis=-1)
            q = q.reshape(1, T, mod.num_k_heads, mod.head_k_dim)
            k = k.reshape(1, T, mod.num_k_heads, mod.head_k_dim)
            inv = mod.head_k_dim ** -0.5
            # This is RMS normalization with eps=1e-6, not HF l2norm's epsilon.
            q = (inv ** 2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv * mx.fast.rms_norm(k, None, 1e-6)
            beta = mx.sigmoid(mod.in_proj_b(x))
            # Use the installed implementation, including its final A_log dtype
            # cast: exp(logd) must describe the actual recurrent decay.
            from mlx_lm.models.gated_delta import compute_g
            decay = self._numpy(compute_g(mod.A_log, mod.in_proj_a(x), mod.dt_bias)[0])
            with np.errstate(divide="ignore"):
                logd = np.log(decay).T.copy()
            c = {"kind": "linear", "q": self._numpy(q[0].transpose(1, 0, 2)),
                 "k": self._numpy(k[0].transpose(1, 0, 2)),
                 "beta": self._numpy(beta[0]).T.copy(), "logd": logd,
                 "rep": mod.num_v_heads // mod.num_k_heads,
                 "v": self._numpy(v.reshape(T, mod.num_v_heads, mod.head_v_dim).transpose(1, 0, 2))}
        self._capture_heads["q"][i] = c

    def _capture_gnorm(self, i, mod, inputs):
        if self._capture_heads is None:
            return
        o = self._numpy(inputs[0])
        z = self._numpy(inputs[1]) if len(inputs) > 1 and inputs[1] is not None else None
        sc = self._head_norm_weight(mod) / np.sqrt(np.mean(o * o, axis=-1, keepdims=True) + mod.eps)
        if z is not None:
            sc = sc * z * _sigmoid(z)
        H = self.n_heads(i)
        self._capture_heads["s"][i] = sc.reshape(-1, H, sc.shape[-1]).transpose(1, 0, 2).copy()

    def _capture_head_output(self, i, mod, x, n_heads):
        if self._capture_heads is not None:
            if x.ndim != 3 or x.shape[0] != 1:
                raise ValueError("head capture requires one unpadded sequence")
            xv = self._numpy(x[0]).reshape(x.shape[1], n_heads, -1)
            W = self._head_weight(mod).reshape(-1, n_heads, xv.shape[-1])
            # One head at a time avoids a [T,H,hidden] temporary on long texts.
            writes = np.stack([np.linalg.norm(xv[:, h] @ W[:, h].T, axis=-1)
                               for h in range(n_heads)], axis=1)
            self._capture_heads["writes"][i] = writes
            self._capture_heads["y"][i] = xv.transpose(1, 0, 2).copy()
        mine = [p for p in (getattr(self, "_heads", None) or []) if p[0] == i]
        if not mine:
            return x
        scales = np.ones(n_heads, dtype=np.float32)
        for _, head, scale in mine:
            if head is None:
                scales *= scale
            else:
                scales[head] *= scale
        v = x.reshape(*x.shape[:-1], n_heads, -1)
        return (v * mx.array(scales, dtype=x.dtype)[:, None]).reshape(x.shape)

    @_locked
    def head_row(self, layer, head, pos):
        return self._head_row(layer, head, pos)

    def _head_row(self, layer, head, pos):
        row = self.raw_row(layer, head, pos)
        if self.heads["q"][layer]["kind"] != "full":
            row = _unit_rows(row)
        return row.tolist()

    def raw_row(self, layer, head, pos, caps=None):
        """Unnormalized signed DeltaNet coefficients, or full-attention softmax.

        Called with the parent's model lock held. ``row @ v`` reconstructs the
        pre-output-gate head value; the DeltaNet recurrence starts at zero state.
        """
        c = (self.heads if caps is None else caps)["q"][layer]
        if pos < 0 or pos >= c["q"].shape[1]:
            raise IndexError("query token is outside the captured text")
        if c["kind"] == "full":
            kv = head // (c["q"].shape[0] // c["k"].shape[0])
            return _softmax((c["k"][kv, :pos + 1] @ c["q"][head, pos]) * c["scale"])
        kh = head // c["rep"]
        q, k = c["q"][kh], c["k"][kh]
        beta, decay = c["beta"][head], np.exp(c["logd"][head])
        u, row = q[pos].copy(), np.zeros(pos + 1, dtype=np.float32)
        for j in range(pos, -1, -1):
            dot = u @ k[j]
            row[j] = beta[j] * dot
            u = decay[j] * (u - beta[j] * dot * k[j])
        return row

    @_locked
    def head_pattern(self, layer, head):
        return self._head_pattern(layer, head)

    def _head_pattern(self, layer, head):
        c = self.heads["q"][layer]
        if c["kind"] == "full":
            kv = head // (c["q"].shape[0] // c["k"].shape[0])
            s = (c["q"][head] @ c["k"][kv].T) * c["scale"]
            s[np.triu_indices(s.shape[-1], 1)] = -np.inf
            return _softmax(s)
        kh = head // c["rep"]
        return self._linear_patterns(c["q"][kh:kh + 1], c["k"][kh:kh + 1],
                                     c["beta"][head:head + 1], c["logd"][head:head + 1])[0]

    @staticmethod
    def _linear_patterns(q, k, beta, logd):
        H, T, D = q.shape
        A = np.zeros((H, T, T), dtype=np.float32)
        U = np.zeros((H, T, D), dtype=np.float32)
        decay = np.exp(logd)
        for j in range(T - 1, -1, -1):
            U[:, j] = q[:, j]
            dot = np.einsum("htd,hd->ht", U, k[:, j])
            A[:, :, j] = beta[:, j, None] * dot
            U = decay[:, j, None, None] * (U - A[:, :, j, None] * k[:, j, None, :])
        return _unit_rows(A)

    @_locked
    def layer_patterns(self, layer, caps=None):
        return self._layer_patterns(layer, caps)

    def _layer_patterns(self, layer, caps=None):
        c = (self.heads if caps is None else caps)["q"][layer]
        if c["kind"] == "full":
            keys = np.repeat(c["k"], c["q"].shape[0] // c["k"].shape[0], axis=0)
            s = (c["q"] @ keys.transpose(0, 2, 1)) * c["scale"]
            a, b = np.triu_indices(s.shape[-1], 1)
            s[:, a, b] = -np.inf
            return _softmax(s)
        return self._linear_patterns(np.repeat(c["q"], c["rep"], axis=0),
                                     np.repeat(c["k"], c["rep"], axis=0), c["beta"], c["logd"])

    @_locked
    def attn_drive(self, layer, n, pos, caps=None, top=6):
        return self._attn_drive(layer, n, pos, caps, top)

    @staticmethod
    def _dense_mlp(layer):
        if not all(hasattr(layer.mlp, name) for name in ("gate_proj", "up_proj", "down_proj")):
            raise NotImplementedError("neuron head analysis requires a dense MLP, not routed MoE experts")
        return layer.mlp

    def _attn_drive(self, layer, n, pos, caps=None, top=6):
        c = self.heads if caps is None else caps
        me = self.lm.layers[layer]
        mlp = self._dense_mlp(me)
        x = c["x"][layer][pos]
        # Accept the core's lazy norm captures as well as materialized NumPy.
        x = self._numpy(x) if isinstance(x, mx.array) else np.asarray(x, dtype=np.float32)
        rms = float(c["rms"][layer][pos])
        wg, wu = self._head_weight(mlp.gate_proj)[n], self._head_weight(mlp.up_proj)[n]
        g, u = wg @ x, wu @ x
        sg = _sigmoid(g)
        J = ((sg * (1 + g * (1 - sg))) * u * wg + g * sg * wu) * self._head_norm_weight(me.post_attention_layernorm) / rms
        out, projections = [], {}
        for l in range(layer + 1):
            lay = self.lm.layers[l]
            W = self._head_weight(lay.self_attn.o_proj if hasattr(lay, "self_attn") else lay.linear_attn.out_proj)
            y = c["y"][l][:, pos]
            t = (W.T @ J).reshape(y.shape)
            projections[l] = t
            out.extend((l, h, float(v)) for h, v in enumerate((t * y).sum(axis=-1)))
        src = {}
        for l, h, _ in sorted(out, key=lambda r: -abs(r[2]))[:max(0, top)]:
            q = c["q"][l]
            if q["kind"] == "full":
                kv = h // (q["q"].shape[0] // q["k"].shape[0])
                v, sc = q["v"][kv, :pos + 1], q["s"][h, pos]
            else:
                v, sc = q["v"][h, :pos + 1], c["s"][l][h, pos]
            src[(l, h)] = (self.raw_row(l, h, pos, c) * (v @ (sc * projections[l][h]))).tolist()
        return {"act": float(g * sg * u), "total": float((sg * (1 + g * (1 - sg)) * g + g * sg) * u),
                "heads": out, "src": src}

    def _value_spaces(self, lay):
        g = self._head_norm_weight(lay.input_layernorm)
        if hasattr(lay, "self_attn"):
            a = lay.self_attn
            hd, nq, nkv = a.head_dim, a.num_attention_heads, a.num_key_value_heads
            V, O = self._head_weight(a.v_proj), self._head_weight(a.o_proj)
            for h in range(nq):
                kv = h // (nq // nkv)
                yield V[kv * hd:(kv + 1) * hd] * g, O[:, h * hd:(h + 1) * hd]
        else:
            a = lay.linear_attn
            if not hasattr(a, "in_proj_qkv"):
                raise NotImplementedError("weight analysis requires Qwen3.5 separate qkv projections")
            hd, start = a.head_v_dim, 2 * a.key_dim
            V, O = self._head_weight(a.in_proj_qkv), self._head_weight(a.out_proj)
            norm = self._head_norm_weight(a.norm)
            for h in range(a.num_v_heads):
                yield V[start + h * hd:start + (h + 1) * hd] * g, O[:, h * hd:(h + 1) * hd] * norm

    def _qk_spaces(self, lay, role):
        g = self._head_norm_weight(lay.input_layernorm)
        if hasattr(lay, "self_attn"):
            a = lay.self_attn
            hd, nq, nkv = a.head_dim, a.num_attention_heads, a.num_key_value_heads
            if role == "q":
                W = self._head_weight(a.q_proj).reshape(nq, 2 * hd, -1)[:, :hd]
                for h in range(nq):
                    yield W[h] * g, [h]
            else:
                W = self._head_weight(a.k_proj)
                rep = nq // nkv
                for kv in range(nkv):
                    yield W[kv * hd:(kv + 1) * hd] * g, list(range(kv * rep, (kv + 1) * rep))
        else:
            a = lay.linear_attn
            if not hasattr(a, "in_proj_qkv"):
                raise NotImplementedError("weight analysis requires Qwen3.5 separate qkv projections")
            hd, rep = a.head_k_dim, a.num_v_heads // a.num_k_heads
            W = self._head_weight(a.in_proj_qkv)
            start = 0 if role == "q" else a.key_dim
            for kh in range(a.num_k_heads):
                yield W[start + kh * hd:start + (kh + 1) * hd] * g, list(range(kh * rep, (kh + 1) * rep))

    @_locked
    def head_links(self, layer, head, k=40):
        # Backend-specific key: never reuse HF/another backend's quantized scores.
        key = f"mlx-heads-v1:{layer}:{head}:{k}:qk"
        with self._db_lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS headcon (key TEXT PRIMARY KEY, val TEXT)")
            row = self._db.execute("SELECT val FROM headcon WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])
        self.ensure_loaded()
        lay = self.lm.layers[layer]
        if head < 0 or head >= self.n_heads(layer):
            raise IndexError("attention head is out of range")
        WV, WO = next(pair for h, pair in enumerate(self._value_spaces(lay)) if h == head)
        QV, QO = _basis(WV.T), _basis(WO)
        QQ = _basis(next(W for W, hs in self._qk_spaces(lay, "q") if head in hs).T)
        QK = _basis(next(W for W, hs in self._qk_spaces(lay, "k") if head in hs).T)
        reads, rq, rk, wr = [], [], [], []
        for l in range(layer):
            D = self._head_weight(self._dense_mlp(self.lm.layers[l]).down_proj)
            dn = np.maximum(np.linalg.norm(D, axis=0), 1e-8)
            for scores, Q in ((reads, QV), (rq, QQ), (rk, QK)):
                scores.append(np.linalg.norm(Q.T @ D, axis=0) / dn)
        for lay2 in self.lm.layers[layer:]:
            mlp = self._dense_mlp(lay2)
            w = self._head_norm_weight(lay2.post_attention_layernorm)
            G, U = self._head_weight(mlp.gate_proj) * w, self._head_weight(mlp.up_proj) * w
            def share(R):
                return np.linalg.norm(R @ QO, axis=1) / np.maximum(np.linalg.norm(R, axis=1), 1e-8)
            wr.append(share(G) * share(U))
        out = {}
        for side, scores, l0 in (("reads", reads, 0), ("q", rq, 0), ("k", rk, 0), ("writes", wr, layer)):
            out[side] = []
            if not scores:
                continue
            s = np.concatenate(scores)
            # torch.std's original sample correction, including safe tiny inputs.
            std = s.std(ddof=1) if s.size > 1 else 0.0
            z = (s - s.mean()) / (std + 1e-8)
            order = np.argsort(-z, kind="stable")[:max(0, k)]
            offsets = np.cumsum([0] + [len(v) for v in scores])
            for idx in order:
                li = int(np.searchsorted(offsets, idx, side="right") - 1)
                out[side].append({"layer": l0 + li, "n": int(idx - offsets[li]), "z": round(float(z[idx]), 2)})
        with self._db_lock:
            self._db.execute("INSERT OR REPLACE INTO headcon VALUES (?, ?)", (key, json.dumps(out)))
            self._db.commit()
        return out

    @_locked
    def _head_bases(self):
        if getattr(self, "_hb", None) is not None:
            return self._hb
        self.ensure_loaded()
        qv, qo, spans, at = [], [], [], 0
        for l, lay in enumerate(self.lm.layers):
            for h, (WV, WO) in enumerate(self._value_spaces(lay)):
                V, O = _basis(WV.T), _basis(WO)
                width = V.shape[1]
                if O.shape[1] != width:
                    raise ValueError("head read and write basis dimensions differ")
                qv.append(V)
                qo.append(O)
                spans.append((l, h, at, at + width))
                at += width
        self._hb = (np.concatenate(qv, axis=1), np.concatenate(qo, axis=1), spans)
        return self._hb

    @_locked
    def neuron_heads(self, layer, n):
        QV, QO, spans = self._head_bases()
        me = self.lm.layers[layer]
        mlp = self._dense_mlp(me)
        d = self._numpy(self.down_col(layer, n))
        w = self._head_norm_weight(me.post_attention_layernorm)
        gr, ur = self._head_weight(mlp.gate_proj)[n] * w, self._head_weight(mlp.up_proj)[n] * w
        pr, pg, pu = QV.T @ d, QO.T @ gr, QO.T @ ur
        dn, gn, un = (max(float(np.linalg.norm(a)), 1e-12) for a in (d, gr, ur))
        reads, writes = [], []
        for l, h, a, b in spans:
            base = ((b - a) / d.shape[0]) ** 0.5
            if l > layer:
                reads.append((l, h, float(np.linalg.norm(pr[a:b])) / dn / base))
            else:
                writes.append((l, h, float(np.linalg.norm(pg[a:b])) / gn * float(np.linalg.norm(pu[a:b])) / un / base ** 2))
        out = {}
        for side in (reads, writes):
            if len(side) < 2:
                continue
            v = np.array([x[2] for x in side])
            z = (v - v.mean()) / (v.std() + 1e-12)
            for (l, h, _), zz in zip(side, z):
                out.setdefault(l, {})[h] = round(float(zz), 2)
        return {l: [hs[h] for h in sorted(hs)] for l, hs in out.items()}

    @_locked
    def _qk_bases(self):
        if getattr(self, "_qkb", None) is not None:
            return self._qkb
        self.ensure_loaded()
        out = {}
        for role in ("q", "k"):
            cols, spans, at = [], [], 0
            for l, lay in enumerate(self.lm.layers):
                for W, hs in self._qk_spaces(lay, role):
                    B = _basis(W.T)
                    cols.append(B)
                    spans.append((l, hs, at, at + B.shape[1]))
                    at += B.shape[1]
            out[role] = (np.concatenate(cols, axis=1), spans)
        self._qkb = out
        return out

    @_locked
    def neuron_head_roles(self, layer, n, k=8):
        qk = self._qk_bases()
        zs = self.neuron_heads(layer, n)
        d = self._numpy(self.down_col(layer, n))
        dn = max(float(np.linalg.norm(d)), 1e-12)
        proj = {r: B.T @ d for r, (B, _) in qk.items()}
        roles = {}
        for r, (_, spans) in qk.items():
            side = [(l, hs, float(np.linalg.norm(proj[r][a:b])) / dn / ((b - a) / d.shape[0]) ** 0.5)
                    for l, hs, a, b in spans if l > layer]
            if len(side) < 2:
                roles[r] = []
                continue
            v = np.array([x[2] for x in side])
            z = (v - v.mean()) / (v.std() + 1e-12)
            rows = sorted(((zz, l, hs) for (l, hs, _), zz in zip(side, z)), key=lambda x: -x[0])[:max(0, k)]
            roles[r] = [{"layer": l, "head": hs[0], "heads": hs, "z": round(float(zz), 1)} for zz, l, hs in rows]
        flat = [(z, int(l), h) for l, row in zs.items() for h, z in enumerate(row)]
        for r, keep in (("v", lambda l: l > layer), ("o", lambda l: l <= layer)):
            rows = []
            for z, l, h in sorted((x for x in flat if keep(x[1])), reverse=True):
                if rows and rows[-1]["layer"] == l and rows[-1]["z"] == round(z, 1):
                    rows[-1]["heads"].append(h)
                elif len(rows) < max(0, k):
                    rows.append({"layer": l, "head": h, "heads": [h], "z": round(z, 1)})
                else:
                    break
            for x in rows:
                x["heads"].sort()
                x["head"] = x["heads"][0]
            roles[r] = rows
        return roles

    def n_heads(self, layer):
        lay = self.lm.layers[layer]
        return lay.self_attn.num_attention_heads if hasattr(lay, "self_attn") else lay.linear_attn.num_v_heads
