"""Gated attention-based multiple-instance learning (Ilse et al. 2018), PyTorch.

A patient is a bag of cells, each a d_in-dim vector (e.g. PCA). The model scores
every cell, pools them with softmax attention into one patient vector, and outputs
a probability. shared/mil_numpy.py mirrors the forward pass for torch-free inference.
"""
import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(self, d_in: int = 50, d: int = 128, d_att: int = 64, p_drop: float = 0.3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Dropout(p_drop))
        self.V = nn.Linear(d, d_att)
        self.U = nn.Linear(d, d_att)
        self.w = nn.Linear(d_att, 1)
        self.head = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor):
        """x: (N cells, d_in). Returns (probability scalar, attention weights of shape (N,))."""
        h = self.enc(x)
        scores = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = torch.softmax(scores, dim=0)
        z = (a.unsqueeze(-1) * h).sum(0)
        return torch.sigmoid(self.head(z)).squeeze(), a
