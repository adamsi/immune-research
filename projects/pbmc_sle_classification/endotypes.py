"""Step 3: find SLE subtypes (endotypes) from the patients' blood, to give the classifier a richer target.

SLE is heterogeneous: some patients have a strong interferon response, others mostly lose naive T cells or
expand cytotoxic cells. This step describes each SLE sample by an interpretable profile, clusters the profiles,
and names the clusters. The classifier then predicts healthy vs each subtype (train.py).

Profile of a sample (all computed from its cells):
  - abundance: CLR fraction of each Perez cell type
  - programs: mean z-scored log-CPM of a gene program (interferon, cytotoxic) in each lineage's pseudobulk
Batch handling: each feature is centred on the healthy samples of the same processing cohort, so a cohort
shift is not mistaken for a subtype (cohorts with < 3 healthy samples use all healthy samples).

Input : data/processed/pbmc_rep.h5ad
Output: results/samples.csv            one row per sample: donor, SLE, state, batch, ancestry, endotype, class
        results/endotype_profiles.csv  mean profile z-score per subtype (what defines each one)
        results/endotype_selection.csv silhouette, stability and sizes for each k tried
        results/endotypes.png          heatmap of the profiles
        artifacts/classes.json         class names and descriptions, read by train.py, insights.py and the app
"""
import json
import os

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

from shared.common import load_config, parse_args

from . import DEFAULT_CONFIG, path
from .features import composition_matrix, log_cpm, pseudobulk, sample_table

matplotlib.use("Agg")

PROGRAMS = {"cytotoxic": ["GZMB", "GZMH", "PRF1", "NKG7", "GNLY", "GZMA"]}


def profiles(adata: ad.AnnData, samples: pd.DataFrame, sample_key: str, ifn_genes: list[str],
             min_frac: float = 0.005, min_cells: int = 10) -> pd.DataFrame:
    """Samples x interpretable features (abundance CLR + gene-program scores per lineage)."""
    obs = adata.obs
    ct = obs["cell_type_fine"].astype(str)
    keep_types = ct.value_counts(normalize=True)
    keep_types = keep_types[keep_types >= min_frac].index.tolist()
    names, comp = composition_matrix(ct, obs[sample_key], keep_types)
    feats = {f"{t} abundance": pd.Series(comp[:, i], index=names) for i, t in enumerate(keep_types)}

    programs = {"IFN": ifn_genes, **PROGRAMS}
    var = pd.Index(adata.var_names)
    for lin in obs["lineage"].astype(str).value_counts().index:
        sub = adata[obs["lineage"].astype(str).values == lin]
        ids, counts, n = pseudobulk(sub.layers["counts"], sub.obs[sample_key])
        ok = n >= min_cells
        if ok.sum() < 0.8 * len(samples):
            continue
        lc = pd.DataFrame(log_cpm(counts[ok]), index=ids[ok], columns=var)
        for prog, genes in programs.items():
            g = [x for x in genes if x in var]
            if len(g) < 3 or (prog == "cytotoxic" and lin not in ("T8", "NK", "T4")):
                continue
            z = (lc[g] - lc[g].mean()) / lc[g].std().replace(0, 1)
            feats[f"{prog} in {lin}"] = z.mean(axis=1)
    return pd.DataFrame(feats).reindex(samples.index)


def healthy_centre(F: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    """Subtract the mean of the same cohort's healthy samples from every feature, then scale by the healthy SD."""
    F = F.fillna(F.median()).astype(np.float64)
    healthy = samples["sle"] == 0
    out = F.copy()
    groups = samples["batch"] if "batch" in samples else pd.Series("all", index=samples.index)
    for b in groups.unique():
        m = groups == b
        ref = F[m & healthy] if (m & healthy).sum() >= 3 else F[healthy]
        out[m] = F[m] - ref.mean()
    return out / F[healthy].std().replace(0, 1)


def choose_k(X: np.ndarray, k_range, min_size: int, n_boot: int, seed: int) -> tuple[int, pd.DataFrame]:
    """Try each k: silhouette (separation) and bootstrap ARI (does an 80% resample give the same clusters?).
    Pick the best silhouette among k with all clusters >= min_size and stability >= 0.6."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(X):
            break
        full = KMeans(k, n_init=50, random_state=seed).fit_predict(X)
        aris = []
        for _ in range(n_boot):
            idx = rng.choice(len(X), int(0.8 * len(X)), replace=False)
            part = KMeans(k, n_init=10, random_state=int(rng.integers(1e9))).fit_predict(X[idx])
            aris.append(adjusted_rand_score(full[idx], part))
        rows.append({"k": k, "silhouette": silhouette_score(X, full), "stability_ari": float(np.mean(aris)),
                     "min_size": int(np.bincount(full).min())})
    sel = pd.DataFrame(rows)
    ok = sel[(sel["min_size"] >= min_size) & (sel["stability_ari"] >= 0.6)]
    best = ok if len(ok) else sel
    return int(best.loc[best["silhouette"].idxmax(), "k"]), sel


def describe(profile: pd.Series, n: int = 2) -> str:
    """Name a subtype by its strongest features relative to the other SLE samples, e.g. 'IFN in cM ↑ · T4_naive abundance ↓'."""
    top = profile.reindex(profile.abs().sort_values(ascending=False).index[:n])
    return " · ".join(f"{f} {'↑' if v > 0 else '↓'}" for f, v in top.items())


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    d, e = cfg["data"], cfg["endotypes"]
    art, res = path(cfg["paths"]["artifacts"]), path(cfg["paths"]["results"])
    os.makedirs(art, exist_ok=True)
    os.makedirs(res, exist_ok=True)

    adata = ad.read_h5ad(path(d["represented"]))
    batch_key = adata.uns.get("batch_key") or None
    samples = sample_table(adata.obs, d["sample_key"], d["donor_key"], batch_key, d["sle_label"])
    print(f"[endotypes] {len(samples)} samples, {samples['sle'].sum()} SLE")

    print("[endotypes] 1/3 profiles (abundance + gene programs per lineage), centred on same-cohort healthy samples")
    F = healthy_centre(profiles(adata, samples, d["sample_key"], e["ifn_genes"]), samples)
    sle = samples["sle"] == 1
    Xs = F[sle].to_numpy()
    X = PCA(n_components=min(10, Xs.shape[1], len(Xs) - 1), random_state=0).fit_transform(Xs)

    print("[endotypes] 2/3 clustering SLE samples")
    k_range = e["k_range"]
    k, sel = choose_k(X, k_range, e["min_size"], e["n_boot"], cfg["train"]["seed"])
    if e["k"] != "auto":
        k = int(e["k"])
    sel.to_csv(os.path.join(res, "endotype_selection.csv"), index=False)
    print(sel.round(3).to_string(index=False))
    lab = KMeans(k, n_init=50, random_state=cfg["train"]["seed"]).fit_predict(X)

    prof = F[sle].groupby(lab).mean()
    ifn_cols = [c for c in prof.columns if c.startswith("IFN")]
    order = prof[ifn_cols].mean(axis=1).sort_values(ascending=False).index if ifn_cols else prof.index
    letters = "ABCDEFGH"
    rename = {old: f"SLE-{letters[i]}" for i, old in enumerate(order)}
    prof = prof.loc[order].rename(index=rename)
    samples["endotype"] = "healthy"
    samples.loc[sle, "endotype"] = pd.Series(lab, index=samples.index[sle]).map(rename)
    classes = ["healthy"] + list(prof.index)
    samples["class"] = samples["endotype"].map({c: i for i, c in enumerate(classes)})

    print("[endotypes] 3/3 save")
    samples.to_csv(os.path.join(res, "samples.csv"), index_label="sample")
    prof.to_csv(os.path.join(res, "endotype_profiles.csv"), index_label="endotype")
    sle_mean = F[sle].mean()
    desc = {"healthy": "healthy donor", **{c: describe(prof.loc[c] - sle_mean) for c in prof.index}}
    json.dump({"classes": classes, "descriptions": desc, "k_chosen": k,
               "sizes": samples["endotype"].value_counts().to_dict()},
              open(os.path.join(art, "classes.json"), "w"), indent=1, ensure_ascii=False)
    for c in classes:
        print(f"  {c:8s} n={int((samples['endotype'] == c).sum()):3d}  {desc[c]}")
    if "disease_state" in samples:
        print(pd.crosstab(samples["endotype"], samples["disease_state"]))

    import matplotlib.pyplot as plt
    top = prof.abs().max().sort_values(ascending=False).index[:20]
    fig, ax = plt.subplots(figsize=(0.45 * len(top) + 2, 0.5 * len(prof) + 1.5))
    v = float(np.nanmax(np.abs(prof[top].to_numpy()))) or 1.0
    im = ax.imshow(prof[top].to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(top)), top, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(prof)), prof.index)
    fig.colorbar(im, ax=ax, label="z vs healthy (same cohort)")
    fig.savefig(os.path.join(res, "endotypes.png"), dpi=120, bbox_inches="tight")
    print(f"[endotypes] wrote {res}/samples.csv and {art}/classes.json")


if __name__ == "__main__":
    main()
