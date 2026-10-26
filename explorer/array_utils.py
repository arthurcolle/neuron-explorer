"""NumPy interchange for the HTTP boundary; neither backend is imported here."""
import numpy as np


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def topk(value, k):
    value = as_numpy(value)
    k = max(0, min(int(k), value.shape[-1]))
    order = np.argsort(-value, axis=-1, kind="stable")[..., :k]
    return np.take_along_axis(value, order, axis=-1), order
