"""Numpy-only inference for the trained AttentionMIL, so the demo app does not need PyTorch.

Weights come from model.npz, written by projects/cart_responder/train.py.
The forward pass mirrors shared/mil_torch.py exactly; dropout is the identity at inference.
"""
import numpy as np


def load_weights(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k].astype(np.float32) for k in f.files}


def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def predict(p: dict, x: np.ndarray):
    """x: (N cells, d_in) float32. Returns (probability, attention weights of shape (N,))."""
    h = np.maximum(x @ p["enc.0.weight"].T + p["enc.0.bias"], 0.0)
    gate = np.tanh(h @ p["V.weight"].T + p["V.bias"]) * _sigmoid(h @ p["U.weight"].T + p["U.bias"])
    s = (gate @ p["w.weight"].T + p["w.bias"]).ravel()
    s = s - s.max()
    a = np.exp(s) / np.exp(s).sum()
    z = a @ h
    prob = float(_sigmoid(z @ p["head.weight"].T + p["head.bias"]).ravel()[0])
    return prob, a.astype(np.float32)
