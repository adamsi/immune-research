"""Numpy-only inference for the trained AttentionMIL, so the demo app does not need PyTorch.

Weights come from model.npz, written by each project's train.py. The forward pass mirrors
shared/mil_torch.py exactly (dropout is the identity at inference). The number of heads, classes and
extra bag-level features is read from the weight shapes, so old single-head binary weights still work.
"""
import numpy as np


def load_weights(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k].astype(np.float32) for k in f.files}


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def predict(p: dict, x: np.ndarray, extra: np.ndarray | None = None):
    """x: (N cells, d_in) float32, extra: (d_extra,) or None.
    Returns (probability float if binary else class-probability vector, attention (N,) averaged over heads)."""
    h = np.maximum(x @ p["enc.0.weight"].T + p["enc.0.bias"], 0.0)
    gate = np.tanh(h @ p["V.weight"].T + p["V.bias"]) * _sigmoid(h @ p["U.weight"].T + p["U.bias"])
    s = (gate @ p["w.weight"].T + p["w.bias"]).T                  # (heads, N)
    s = s - s.max(axis=1, keepdims=True)
    a = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
    z = (a @ h).ravel()
    if extra is not None:
        z = np.concatenate([z, np.asarray(extra, dtype=np.float32)])
    out = z @ p["head.weight"].T + p["head.bias"]
    if out.shape[0] == 1:
        prob = float(_sigmoid(out)[0])
    else:
        e = np.exp(out - out.max())
        prob = (e / e.sum()).astype(np.float32)
    return prob, a.mean(0).astype(np.float32)
