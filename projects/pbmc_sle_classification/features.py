"""Torch-free building blocks shared by training and the demo app.

Everything here runs on numpy/pandas/scikit-learn only, so the Hugging Face Space can embed and classify
a new patient without PyTorch, scvi-tools or CellTypist installed:
  - pseudobulk / log_cpm                       features for the logistic-regression baseline
  - encode_scvi                                the trained scVI encoder (counts -> 30-d latent), exported to numpy
  - predict_celltypes                          the CellTypist classifier, exported to numpy
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def to_dense(X) -> np.ndarray:
    return X.toarray().astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)


def donor_labels(obs: pd.DataFrame, sle_label: str, donor_key: str = "donor_id") -> pd.Series:
    """One label per donor, 1 = SLE and 0 = healthy, indexed by donor id (str)."""
    d = obs[[donor_key, "disease"]].drop_duplicates(donor_key)
    return pd.Series((d["disease"].astype(str) == sle_label).astype(int).values,
                     index=d[donor_key].astype(str).values, name="y")


def pseudobulk(X, groups) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum the rows of X (cells x genes, raw counts) per group.
    Returns (group names, summed matrix groups x genes, cells per group)."""
    codes, names = pd.factorize(np.asarray(groups).astype(str))
    n = len(codes)
    G = sp.csr_matrix((np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(len(names), n))
    return np.asarray(names), to_dense(G @ X), np.bincount(codes)


def log_cpm(X) -> np.ndarray:
    """Counts -> log(1 + counts per million), row-wise. Makes donors with different cell numbers comparable."""
    X = np.asarray(X, dtype=np.float64)
    return np.log1p(X / np.maximum(X.sum(axis=1, keepdims=True), 1) * 1e6)


def log_normalize(X, totals=None, target: float = 1e4) -> np.ndarray:
    """Per-cell normalisation to `target` counts and log1p (what scanpy and CellTypist expect)."""
    X = to_dense(X)
    totals = X.sum(1) if totals is None else np.asarray(totals, dtype=np.float32)
    return np.log1p(X / np.maximum(totals, 1)[:, None] * target)


# ---------------------------------------------------------------- scVI encoder

def export_scvi_encoder(vae) -> dict:
    """Pull the weights needed for z = encoder(counts) out of a trained scvi.model.SCVI.
    Valid for the default architecture used here: 1 hidden layer, batch norm, encode_covariates=False."""
    sd = {k: v.detach().cpu().numpy() for k, v in vae.module.state_dict().items()}
    pre = "z_encoder.encoder.fc_layers.Layer 0."
    return {
        "W1": sd[pre + "0.weight"], "b1": sd[pre + "0.bias"],
        "bn_mean": sd[pre + "1.running_mean"], "bn_var": sd[pre + "1.running_var"],
        "bn_w": sd[pre + "1.weight"], "bn_b": sd[pre + "1.bias"],
        "W_mu": sd["z_encoder.mean_encoder.weight"], "b_mu": sd["z_encoder.mean_encoder.bias"],
    }


def encode_scvi(enc: dict, counts) -> np.ndarray:
    """Raw counts (cells x model genes, in genes.txt order) -> scVI latent mean (cells x n_latent).
    Mirrors scVI inference: log1p input, Linear, BatchNorm (eval, eps 1e-3), ReLU, Linear."""
    x = np.log1p(to_dense(counts))
    h = x @ enc["W1"].T + enc["b1"]
    h = (h - enc["bn_mean"]) / np.sqrt(enc["bn_var"] + 1e-3) * enc["bn_w"] + enc["bn_b"]
    h = np.maximum(h, 0.0)
    return (h @ enc["W_mu"].T + enc["b_mu"]).astype(np.float32)


# ---------------------------------------------------------------- CellTypist

def export_celltypist(model) -> dict:
    """Arrays of a celltypist.models.Model: features, classes, scaler and logistic-regression weights."""
    return {
        "features": np.asarray(model.features).astype(str),
        "classes": np.asarray(model.classifier.classes_).astype(str),
        "mean": model.scaler.mean_.astype(np.float32), "scale": model.scaler.scale_.astype(np.float32),
        "coef": model.classifier.coef_.astype(np.float32), "intercept": model.classifier.intercept_.astype(np.float32),
    }


def predict_celltypes(ct: dict, X_lognorm: np.ndarray, genes) -> np.ndarray:
    """Per-cell CellTypist label ("best match" mode) from log-normalised expression.
    Like CellTypist, model genes absent from the input are dropped rather than zero-filled."""
    col = {g: i for i, g in enumerate(genes)}
    idx = np.array([i for i, g in enumerate(ct["features"]) if g in col], dtype=int)
    if len(idx) == 0:
        return np.full(len(X_lognorm), "unknown", dtype=object)
    x = X_lognorm[:, [col[g] for g in ct["features"][idx]]]
    x = (x - ct["mean"][idx]) / ct["scale"][idx]
    x[x > 10] = 10
    scores = x @ ct["coef"][:, idx].T + ct["intercept"]
    return ct["classes"][scores.argmax(1)]
