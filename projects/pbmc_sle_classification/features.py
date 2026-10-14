"""Torch-free building blocks shared by training and the demo app.

Everything here runs on numpy/pandas/scikit-learn only, so the Hugging Face Space can embed and classify
a new patient without PyTorch, scvi-tools or CellTypist installed:
  - sample_table                               one row per blood sample (donor, SLE, state, batch, ancestry)
  - pseudobulk / log_cpm                       summed counts per sample, normalised
  - clr / composition                          cell-type abundance on a log-ratio scale (MIL head + baseline)
  - baseline_features                          per-lineage pseudobulk + abundance for the elastic-net baseline
  - encode_scvi                                the trained scVI encoder (counts -> 30-d latent), exported to numpy
  - predict_celltypes                          the CellTypist classifier, exported to numpy
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def to_dense(X) -> np.ndarray:
    return X.toarray().astype(np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)


def donor_labels(obs: pd.DataFrame, sle_label: str, donor_key: str = "donor_id") -> pd.Series:
    """One label per donor (or per sample, with donor_key = the sample column), 1 = SLE and 0 = healthy, indexed by id (str)."""
    d = obs[[donor_key, "disease"]].drop_duplicates(donor_key)
    return pd.Series((d["disease"].astype(str) == sle_label).astype(int).values,
                     index=d[donor_key].astype(str).values, name="y")


def sample_table(obs: pd.DataFrame, sample_key: str, donor_key: str, batch_key: str | None, sle_label: str) -> pd.DataFrame:
    """One row per sample: donor, SLE flag, disease state, batch, ancestry and cell count, indexed by sample id (str)."""
    cols = [c for c in (donor_key, "disease", "disease_state", batch_key, "self_reported_ethnicity", "sex") if c and c in obs]
    t = obs[[sample_key] + cols].drop_duplicates(sample_key).set_index(sample_key)
    t.index = t.index.astype(str)
    t = t.rename(columns={donor_key: "donor", batch_key: "batch", "self_reported_ethnicity": "ancestry"} if batch_key else
                 {donor_key: "donor", "self_reported_ethnicity": "ancestry"})
    t["sle"] = (t["disease"].astype(str) == sle_label).astype(int)
    t["n_cells"] = obs[sample_key].astype(str).value_counts().reindex(t.index).values
    for c in t.columns:
        if t[c].dtype.name == "category":
            t[c] = t[c].astype(str)
    return t


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
    Valid for scVI's default encoder (batch norm, encode_covariates=False) with any number of hidden layers."""
    sd = {k: v.detach().cpu().numpy() for k, v in vae.module.state_dict().items()}
    out = {"W_mu": sd["z_encoder.mean_encoder.weight"], "b_mu": sd["z_encoder.mean_encoder.bias"]}
    i = 0
    while f"z_encoder.encoder.fc_layers.Layer {i}.0.weight" in sd:
        pre, sfx = f"z_encoder.encoder.fc_layers.Layer {i}.", "" if i == 0 else str(i + 1)
        out.update({f"W{i + 1}": sd[pre + "0.weight"], f"b{i + 1}": sd[pre + "0.bias"],
                    f"bn_mean{sfx}": sd[pre + "1.running_mean"], f"bn_var{sfx}": sd[pre + "1.running_var"],
                    f"bn_w{sfx}": sd[pre + "1.weight"], f"bn_b{sfx}": sd[pre + "1.bias"]})
        i += 1
    return out


def encode_scvi(enc: dict, counts) -> np.ndarray:
    """Raw counts (cells x model genes, in genes.txt order) -> scVI latent mean (cells x n_latent).
    Mirrors scVI inference: log1p input, then per layer Linear, BatchNorm (eval, eps 1e-3), ReLU; then Linear."""
    h = np.log1p(to_dense(counts))
    i = 1
    while f"W{i}" in enc:
        sfx = "" if i == 1 else str(i)
        h = h @ enc[f"W{i}"].T + enc[f"b{i}"]
        h = (h - enc[f"bn_mean{sfx}"]) / np.sqrt(enc[f"bn_var{sfx}"] + 1e-3) * enc[f"bn_w{sfx}"] + enc[f"bn_b{sfx}"]
        h = np.maximum(h, 0.0)
        i += 1
    return (h @ enc["W_mu"].T + enc["b_mu"]).astype(np.float32)


# ---------------------------------------------------------------- cell-type abundance and baseline features

def clr(frac: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Centred log-ratio of compositions (rows sum to 1): log(fraction) minus the row mean of logs.
    Fractions of one sample are not independent (they sum to 1); CLR puts them on an unconstrained scale."""
    lf = np.log(np.asarray(frac, dtype=np.float64) + eps)
    return (lf - lf.mean(axis=-1, keepdims=True)).astype(np.float32)


def composition(cell_types, classes) -> np.ndarray:
    """CLR abundance vector of one sample over a fixed list of cell types (types outside the list are ignored)."""
    ct = pd.Series(np.asarray(cell_types).astype(str))
    frac = ct.value_counts().reindex(list(classes), fill_value=0).to_numpy(np.float64)
    return clr(frac / max(frac.sum(), 1))


def composition_matrix(cell_types, groups, classes) -> tuple[np.ndarray, np.ndarray]:
    """CLR abundance per group (sample). Returns (group names, groups x types)."""
    df = pd.crosstab(np.asarray(groups).astype(str), np.asarray(cell_types).astype(str)).reindex(columns=list(classes), fill_value=0)
    frac = df.to_numpy(np.float64) / np.maximum(df.sum(1).to_numpy()[:, None], 1)
    return df.index.to_numpy(), clr(frac)


def baseline_features(counts, cell_type, spec: dict) -> np.ndarray:
    """Features of one sample for the elastic-net baseline (Perez-style, plus abundance):
    per lineage, log-CPM of that lineage's pseudobulk over its selected genes; then the CLR cell-type abundance.
    counts: cells x model genes (genes.txt order); cell_type: per-cell fine type; spec: baseline_features.json
    (lineages, gene_idx per lineage, cell_types, lineage_of = fine type -> lineage)."""
    cell_type = np.asarray(cell_type).astype(str)
    lineage = np.array([spec["lineage_of"].get(c, "other") for c in cell_type])
    parts = []
    for lin, gidx in zip(spec["lineages"], spec["gene_idx"]):
        m = lineage == lin
        pb = np.asarray(counts[m].sum(0)).ravel()[None, :] if m.any() else np.zeros((1, counts.shape[1]))
        parts.append(log_cpm(pb)[0, gidx])
    parts.append(composition(cell_type, spec["cell_types"]))
    return np.concatenate(parts).astype(np.float32)


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
