"""MLX generation and activation capture, independent of model loading/wrapping.

The owning runtime supplies the lock, tokenizer, model/transformer, weight helpers,
and capture flags. Wrappers call the three ``_capture_*`` methods synchronously
while building the graph. Public numeric results and retained head captures are
CPU NumPy arrays, including the norm inputs used by attention analysis.
"""
import math
import os
from numbers import Integral, Real

import mlx.core as mx
import numpy as np


class CoreMixin:
    @staticmethod
    def _core_eval(*trees):
        """Materialize results *and* side captures before their flags are cleared."""
        arrays = []

        def visit(value):
            if isinstance(value, mx.array):
                arrays.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        for tree in trees:
            visit(tree)
        if arrays:
            mx.eval(*arrays)

    @staticmethod
    def _core_int(value, name, minimum=0):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        return int(value)

    def _core_width(self, layer):
        mlp = self.lm.layers[layer].mlp
        down = mlp.down_proj
        width = getattr(down, "input_dims", getattr(down, "in_features", None))
        if width is None:
            # Quantized gate weights still have one output row per neuron; their
            # packed input dimension (and down_proj's packed width) is not usable.
            gate = getattr(mlp, "gate_proj", None)
            width = gate.weight.shape[0] if gate is not None and hasattr(gate, "weight") else self.weight(down).shape[1]
        return int(width)

    def _core_neuron(self, layer, n, whole=False):
        layer = self._core_int(layer, "layer")
        if layer >= len(self.lm.layers):
            raise ValueError("layer is out of range")
        if n is None and whole:
            return layer, None
        n = self._core_int(n, "neuron")
        if n >= self._core_width(layer):
            raise ValueError("neuron is out of range")
        return layer, n

    @staticmethod
    def _core_token_ids(encoded):
        if hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        a = np.asarray(encoded)
        if a.ndim != 1 or not a.size or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("expected a nonempty one-dimensional sequence of token IDs")
        return a.tolist()

    @staticmethod
    def _core_batch_ids(ids):
        a = np.asarray(ids)
        if a.ndim not in (1, 2) or not a.size or a.dtype.kind not in "iu" or np.any(a < 0):
            raise ValueError("expected nonempty token IDs of shape [T] or [B, T]")
        return mx.array(a[None] if a.ndim == 1 else a, dtype=mx.int32)

    def _capture_mlp(self, i, h):
        """Capture pre-intervention down-projection inputs, then return a new array."""
        if self._capture_full is not None:
            self._capture_full[i] = h[0].astype(mx.float32)
        if self._capture_all is not None:
            a = h[:, self._capture_from:].astype(mx.float32).reshape(-1, h.shape[-1])
            if not a.shape[0]:
                raise ValueError("no tokens remain after skip_first")
            self._capture_all[i] = (mx.max(a, axis=0), mx.argmax(a, axis=0),
                                    mx.min(a, axis=0), mx.argmin(a, axis=0))
        if self._read is not None and self._read[0] == i:
            self._seen["act"] = h[0, :, self._read[1]].astype(mx.float32)
        for layer, n, mode, values in self._push or ():
            if layer != i:
                continue
            values = mx.array(values).astype(h.dtype)
            if values.ndim != 1 or values.shape[0] != h.shape[0]:
                raise ValueError("push rows must match the model batch")
            if n is None:
                h = h * values[:, None, None]
            else:
                col = h[:, :, n:n + 1]
                col = col * values[:, None, None] if mode == "mul" else col + values[:, None, None]
                h = mx.concatenate((h[:, :, :n], col, h[:, :, n + 1:]), axis=-1)
        return h

    def _capture_mlp_input(self, i, mod, inp, out):
        if self._capture_heads is not None:
            # Accept the parent's direct array and the original hook's args tuple.
            inp = inp[0] if isinstance(inp, (tuple, list)) else inp
            r = inp[0].astype(mx.float32)
            eps = getattr(mod, "eps", getattr(mod, "variance_epsilon", 1e-6))
            self._capture_heads["x"][i] = out[0].astype(mx.float32)
            self._capture_heads["rms"][i] = mx.sqrt(mx.mean(r * r, axis=-1) + eps)

    def _capture_residual(self, i, out):
        if self._seen.get("want_resid") == i:
            h = out[0] if isinstance(out, tuple) else out
            r = h[0, 1:].astype(mx.float32)
            self._seen["resid"] = mx.sqrt(mx.sum(r * r, axis=-1))

    # Compatibility closures for callers using the original hook-shaped API.
    def _hook(self, i):
        return lambda mod, inp: (self._capture_mlp(i, inp[0]),)

    def _mlp_in_hook(self, i):
        return lambda mod, inp, out: self._capture_mlp_input(i, mod, inp, out)

    def _resid_hook(self, i):
        return lambda mod, inp, out: self._capture_residual(i, out)

    def layer_resid_p50(self, layer):
        self.ensure_loaded()
        with self.lock:
            layer, _ = self._core_neuron(layer, None, whole=True)
            if layer in self.resid_p50:
                v = float(self.resid_p50[layer])
            else:
                path = os.path.join(self.stats_dir, f"q35-4b_L{layer}.npz")
                if os.path.exists(path):
                    with np.load(path) as stats:
                        v = float(stats["resid_norm_q10_50_90"][1])
                else:
                    ids = self._core_token_ids(self.tok(self.REFERENCE_TEXT))
                    if len(ids) < 2:
                        raise ValueError("residual calibration requires at least two tokens")
                    self._seen.pop("resid", None)
                    self._seen["want_resid"] = layer
                    try:
                        out = self.lm(self._core_batch_ids(ids), cache=None)
                        self._core_eval(out, self._seen)
                        if "resid" not in self._seen:
                            raise RuntimeError("residual wrapper did not capture the requested layer")
                        values = self._numpy(self._seen["resid"])
                        # torch.median's original convention is the lower middle
                        # element, not NumPy's average for an even-sized sample.
                        k = (values.size - 1) // 2
                        v = float(np.partition(values, k)[k])
                    finally:
                        self._seen.pop("want_resid", None)
                        self._seen.pop("resid", None)
            if not math.isfinite(v) or v < 0:
                raise ValueError("residual median must be finite and nonnegative")
            self.resid_p50[layer] = v
            return v

    def chat_ids(self, prompt):
        return self.messages_ids([{"role": "user", "content": prompt}])

    def messages_ids(self, messages):
        self.ensure_loaded()
        return self._core_token_ids(self.tok.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False, tokenize=True))

    def _push_entry(self, layer, n, dose, mode, rows):
        """Normalize additive doses by residual median / down-column norm."""
        self.ensure_loaded()
        with self.lock:
            layer, n = self._core_neuron(layer, n, whole=True)
            if mode not in ("add", "mul"):
                raise ValueError("mode must be 'add' or 'mul'")
            if n is None and mode != "mul":
                raise ValueError("a whole-layer MLP push must be mode 'mul'")
            if isinstance(dose, bool) or not isinstance(dose, Real) or not math.isfinite(dose):
                raise ValueError("dose must be a finite number")
            rows = tuple(rows)
            if not rows or any(r not in (-1, 0, 1) for r in rows):
                raise ValueError("rows must contain only +1, 0 or -1")
            dose = float(dose)
            if mode == "mul":
                vals = [dose if r > 0 else 1.0 if r == 0 else -dose for r in rows]
            else:
                col = mx.array(self.down_col(layer, n)).astype(mx.float32)
                norm = float(mx.sqrt(mx.sum(col * col)).item())
                if not math.isfinite(norm) or norm <= 0:
                    raise ValueError("cannot normalize a zero or nonfinite down column")
                unit = self.layer_resid_p50(layer) / norm
                vals = [r * dose * unit for r in rows]
            if any(not math.isfinite(v) or abs(v) > np.finfo(np.float32).max for v in vals):
                raise ValueError("normalized push is not finite float32")
            return layer, n, mode, mx.array(vals, dtype=mx.float32)

    def steer_stream(self, layer, n, rows_messages, dose, mode, max_new=200, cancel=None):
        yield from self.steer_multi([(layer, n, dose, mode)], rows_messages, max_new, cancel)

    def steer_multi(self, pushes, rows_messages, max_new=200, cancel=None):
        """Greedy token-round interleave: rows 0/1/2 are +dose/baseline/-dose.

        Each row owns an unpadded prompt and an independent causal/recurrent cache.
        No intervention or lock is retained across a yield (including close()).
        """
        from mlx_lm.models.cache import make_prompt_cache

        self.ensure_loaded()
        max_new = self._core_int(max_new, "max_new")
        rows_messages = list(rows_messages)
        if len(rows_messages) != 3:
            raise ValueError("steering requires exactly three conversations")
        if cancel is not None and cancel.is_set():
            yield None, None
            return
        with self.lock:
            pushes = list(pushes)
            entries = [self._push_entry(l, n, d, m, (1, 0, -1)) for l, n, d, m in pushes]
            row_pushes = [[(l, n, m, v[r:r + 1]) for l, n, m, v in entries] for r in range(3)]
            seqs = [self.messages_ids(m) for m in rows_messages]
            caches = [make_prompt_cache(self.model) for _ in range(3)]
            eos = set()
            for value in (getattr(self.tok, "eos_token_id", None), getattr(self.tok, "eos_token_ids", None)):
                if value is not None:
                    eos.update(value if isinstance(value, (list, tuple, set)) else [value])
            if hasattr(self.tok, "convert_tokens_to_ids"):
                end = self.tok.convert_tokens_to_ids("<|im_end|>")
                if end is not None and end != getattr(self.tok, "unk_token_id", None):
                    eos.add(end)
        done, out_ids, sent = [False] * 3, [[], [], []], ["", "", ""]
        for _ in range(max_new):
            for r in range(3):
                if cancel is not None and cancel.is_set():
                    yield None, None
                    return
                if done[r]:
                    continue
                with self.lock:
                    self._push = row_pushes[r] or None
                    try:
                        logits = self.model(self._core_batch_ids(seqs[r]), cache=caches[r])
                        nxt = mx.argmax(logits[0, -1], axis=-1)
                        self._core_eval(logits, nxt, [getattr(c, "state", None) for c in caches[r]])
                        token = int(nxt.item())
                    finally:
                        self._push = None
                if token in eos:
                    done[r] = True
                    continue
                seqs[r] = [token]
                out_ids[r].append(token)
                text = self.tok.decode(out_ids[r], skip_special_tokens=True)
                if not text.endswith("\ufffd"):
                    delta, sent[r] = text[len(sent[r]):], text
                    yield r, delta
            if all(done):
                break
        yield None, None

    def token_units(self, ids):
        """Join byte fragments and emoji joiners/selectors into display units."""
        units, pending, prev = [], [], ""
        for i in range(len(ids)):
            full = self.tok.decode(ids[:i + 1])
            pending.append(i)
            if full.endswith("\ufffd") and i < len(ids) - 1:
                continue
            units.append((full[len(prev):] if full.startswith(prev) else self.tok.decode([ids[j] for j in pending]), pending))
            prev, pending = full, []
        glue = {"\u200d", "\ufe0f", "\ufe0e", "\u20e3"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
        merged = []
        for text, idx in units:
            if merged and text and (text[0] in glue or merged[-1][0].endswith("\u200d")):
                merged[-1] = (merged[-1][0] + text, merged[-1][1] + idx)
            else:
                merged.append((text, idx))
        return merged

    def read(self, layer, n, text, chat=True, pushes=None, heads=None):
        self.ensure_loaded()
        with self.lock:
            layer, n = self._core_neuron(layer, n)
            ids = self.chat_ids(text) if chat else self._core_token_ids(self.tok(text))
            push = [self._push_entry(l, nn, d, m, (1,)) for l, nn, d, m in pushes or ()]
            self._seen.pop("act", None)
            self._read, self._push, self._heads = (layer, n), push or None, list(heads) if heads else None
            try:
                out = self.lm(self._core_batch_ids(ids), cache=None)
                self._core_eval(out, self._seen)
                if "act" not in self._seen:
                    raise RuntimeError("MLP wrapper did not capture the requested neuron")
                act = self._numpy(self._seen["act"]).tolist()
            finally:
                self._read = self._push = self._heads = None
                self._seen.pop("act", None)
            units = []
            for unit, idx in self.token_units(ids):
                vals = [act[i] for i in idx]
                units.append({"text": unit, "act": max(vals, key=abs), "pieces": vals})
            return {"units": units}

    def next_logprobs(self, text, chat=True, prefix="", pushes=None, heads=None):
        """Return (baseline, intervened or None), CPU float32 [vocabulary] arrays."""
        self.ensure_loaded()
        with self.lock:
            ids = self.chat_ids(text) if chat else self._core_token_ids(self.tok(text))
            if prefix:
                ids += self._core_token_ids(self.tok(prefix, add_special_tokens=False))
            x = self._core_batch_ids(ids)
            push = [self._push_entry(l, n, d, m, (1,)) for l, n, d, m in pushes or ()]
            hd = list(heads) if heads else None

            def run(active, head_scales):
                self._push, self._heads = active, head_scales
                try:
                    out = self.model(x, cache=None)
                    logits = out[0, -1].astype(mx.float32)
                    lp = logits - mx.logsumexp(logits, axis=-1)
                    self._core_eval(out, lp)
                    return self._numpy(lp)
                finally:
                    self._push = self._heads = None

            base = run(None, None)
            return base, run(push or None, hd) if push or hd else None

    def all_activations(self, ids, skip_first=False):
        """Layer -> (max, argmax, min, argmin), over flattened batch/token axes.

        Indices are relative to the retained tokens, as in the original runtime.
        Values are float32 NumPy arrays; arg indices are integer NumPy arrays.
        """
        self.ensure_loaded()
        with self.lock:
            x = self._core_batch_ids(ids)
            if skip_first and x.shape[1] < 2:
                raise ValueError("no tokens remain after skip_first")
            self._capture_all, self._capture_from = {}, int(bool(skip_first))
            try:
                out = self.lm(x, cache=None)
                self._core_eval(out, self._capture_all)
                if set(self._capture_all) != set(range(len(self.lm.layers))):
                    raise RuntimeError("MLP wrappers did not capture every layer")
                return {l: (self._numpy(hi), np.array(ahi, dtype=np.int64),
                            self._numpy(lo), np.array(alo, dtype=np.int64))
                        for l, (hi, ahi, lo, alo) in self._capture_all.items()}
            finally:
                self._capture_all, self._capture_from = None, 0

    def full_activations(self, ids, heads=False):
        """Return float32 [layers, tokens, neurons]; optionally retain head data."""
        self.ensure_loaded()
        with self.lock:
            ids = self._core_token_ids(ids)
            self._capture_full = {}
            self._capture_heads = {key: {} for key in ("q", "writes", "y", "s", "x", "rms")} if heads else None
            try:
                out = self.lm(self._core_batch_ids(ids), cache=None)
                self._core_eval(out, self._capture_full, self._capture_heads)
                if len(self._capture_full) != len(self.lm.layers):
                    raise RuntimeError("MLP wrappers did not capture every layer")
                result = np.stack([self._numpy(self._capture_full[l]) for l in range(len(self.lm.layers))])
                if heads:
                    def cpu(value):
                        if isinstance(value, mx.array):
                            return self._numpy(value)
                        if isinstance(value, dict):
                            return {key: cpu(item) for key, item in value.items()}
                        if isinstance(value, (list, tuple)):
                            return type(value)(cpu(item) for item in value)
                        return value
                    self.heads = {"T": len(ids), "ids": list(ids), **cpu(self._capture_heads)}
                return result
            finally:
                self._capture_full = self._capture_heads = None

    def build_act_scale(self, path, tokens=10000, seq=512, seed=0):
        """Build the original pos/neg/tokens NPZ schema from streamed web text.

        Dataset access occurs only on an explicit call, never at import/load time.
        """
        self.ensure_loaded()
        tokens = self._core_int(tokens, "tokens")
        seq = self._core_int(seq, "seq", 2)
        seed = self._core_int(seed, "seed")
        layers = len(self.lm.layers)
        width = self._core_width(0)
        if any(self._core_width(l) != width for l in range(layers)):
            raise ValueError("activation scale requires equal MLP widths")
        pos, neg = np.zeros((layers, width), np.float32), np.zeros((layers, width), np.float32)
        buf, seen = [], 0
        previous_status = self.status
        try:
            if tokens:
                from datasets import load_dataset
                data = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True).shuffle(seed=seed)
                for d in data:
                    encoded = self.tok(d["text"], add_special_tokens=False)
                    raw = encoded["input_ids"] if hasattr(encoded, "keys") else encoded.input_ids
                    buf += list(raw)
                    if self.tok.eos_token_id is not None:
                        buf.append(self.tok.eos_token_id)
                    while len(buf) >= seq and seen < tokens:
                        chunk, buf = buf[:seq], buf[seq:]
                        self.status = f"measuring neuron ranges {seen // 1000}k/{tokens // 1000}k tokens"
                        for l, (hi, _, lo, _) in self.all_activations(chunk, skip_first=True).items():
                            pos[l] = np.maximum(pos[l], hi)
                            neg[l] = np.minimum(neg[l], lo)
                        seen += seq
                    if seen >= tokens:
                        break
            np.savez(path, pos=pos, neg=neg, tokens=seen)
        except BaseException:
            self.status = previous_status
            raise
        self.status = "ready"
        return path
