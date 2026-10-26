"""Select a live inference backend without changing the model or the HTTP API."""
import importlib.util
import os
import platform


def select_backend(preference=None):
    requested = (preference or os.environ.get("EXPLORER_BACKEND", "auto")).lower()
    if requested not in {"auto", "mlx", "torch"}:
        raise ValueError("EXPLORER_BACKEND must be auto, mlx, or torch")
    if requested == "auto":
        apple = platform.system() == "Darwin" and platform.machine() == "arm64"
        return "mlx" if apple and all(importlib.util.find_spec(p) for p in ("mlx", "mlx_lm")) else "torch"
    return requested


BACKEND = select_backend()
try:
    if BACKEND == "mlx":
        os.environ.setdefault("USE_TORCH", "0")
        from runtime_mlx import CACHE, MODEL, NO_MODEL, Runtime
    else:
        from runtime import CACHE, MODEL, NO_MODEL, Runtime
except ModuleNotFoundError as exc:
    raise RuntimeError(
        f"The {BACKEND} backend is not installed. Run uv sync --extra {BACKEND} "
        "(MLX requires Apple Silicon), or select EXPLORER_BACKEND=torch."
    ) from exc
