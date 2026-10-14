"""Gated attention-based multiple-instance learning (Ilse et al. 2018), PyTorch.

A patient is a bag of cells, each a d_in-dim vector (e.g. PCA or scVI latent). The model scores every
cell, pools them with softmax attention into one patient vector, and outputs class probabilities.
shared/mil_numpy.py mirrors the forward pass for torch-free inference.

Options (defaults reproduce the original single-head binary model, so CAR-T weights still load):
  n_heads   several attention heads, each pooling its own view of the bag (multi-head gated attention)
  n_classes 1 = binary (sigmoid, returns a scalar); >1 = multi-class (softmax, returns a vector)
  d_extra   bag-level features concatenated to the pooled vector before the head (e.g. cell-type abundance)
"""
import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(self, d_in: int = 50, d: int = 128, d_att: int = 64, p_drop: float = 0.3,
                 n_heads: int = 1, n_classes: int = 1, d_extra: int = 0):
        super().__init__()
        self.n_classes, self.d_extra = n_classes, d_extra
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.V = nn.Linear(d, d_att)
        self.U = nn.Linear(d, d_att)
        self.w = nn.Linear(d_att, n_heads)
        self.head = nn.Linear(n_heads * d + d_extra, n_classes)

    def logits(self, x: torch.Tensor, extra: torch.Tensor | None = None):
        """x: (N cells, d_in), extra: (d_extra,) or None. Returns (logits (n_classes,), attention (n_heads, N))."""
        h = self.enc(x)
        scores = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).T   # (heads, N)
        a = torch.softmax(scores, dim=1)
        z = (a @ h).reshape(-1)                                                # heads * d
        if extra is not None:
            z = torch.cat([z, extra])
        return self.head(z), a

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None):
        """Returns (probability scalar if binary else class-probability vector, attention (N,) averaged over heads)."""
        out, a = self.logits(x, extra)
        p = torch.sigmoid(out).squeeze() if self.n_classes == 1 else torch.softmax(out, dim=-1)
        return p, a.mean(0)
