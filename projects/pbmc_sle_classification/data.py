"""Step 1: download the Perez 2022 lupus PBMC atlas, subsample, QC, cell labels, normalise, pick HVGs, report batches.

Follows the paper where it states its choices: platelet removal by marker counts, Perez's manual cell-type
labels (markers + CITE-seq; 14 lymphoid subtypes + broad myeloid types), and a larger HVG set.

Input : configs/default.yaml
Output: data/raw/perez2022.h5ad          the original CELLxGENE file (large, downloaded once)
        data/processed/pbmc.h5ad         raw counts in layers["counts"], log-normalised X, HVG flags,
                                         obs["cell_type_fine"] (Perez labels) and obs["lineage"] (broad type)
        results/batch_report.csv         samples per batch x disease state (single-disease batches are confounded)
"""
import gc
import os

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

try:
    from anndata.io import read_elem, sparse_dataset
except ImportError:  # anndata < 0.11
    from anndata.experimental import read_elem, sparse_dataset

from shared.common import load_config, parse_args

from . import DEFAULT_CONFIG, path

BATCH_CANDIDATES = ("Processing_Cohort", "batch_cov", "batch", "pool")


def download(cfg: dict) -> str:
    """Find the Perez 2022 dataset in the CELLxGENE Census by its DOI and download the source .h5ad."""
    dest = path(cfg["data"]["raw"])
    if os.path.exists(dest):
        print(f"  exists, skipping: {dest}")
        return dest
    import cellxgene_census

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with cellxgene_census.open_soma(census_version="stable") as census:
        ds = census["census_info"]["datasets"].read().concat().to_pandas()
    hits = ds[ds["collection_doi"].astype(str).str.contains(cfg["data"]["collection_doi"].split("/")[-1])]
    if hits.empty:
        raise SystemExit(f"No CELLxGENE dataset found for DOI {cfg['data']['collection_doi']}")
    dataset_id = hits.sort_values("dataset_total_cell_count").iloc[-1]["dataset_id"]
    print(f"  downloading CELLxGENE dataset {dataset_id}")
    cellxgene_census.download_source_h5ad(dataset_id, to_path=dest, census_version="stable")
    return dest


def load_subsample(src: str, max_cells: int, seed: int, group_key: str = "sample_uuid", block: int = 25_000) -> ad.AnnData:
    """Keep up to max_cells random cells per sample and load only their raw counts.

    The file is streamed: obs is read first, then the count matrix (raw/X if present, as on
    CELLxGENE, else X) is read `block` rows at a time keeping only the picked rows, so peak RAM
    stays around 1-2 GB however big the file is.
    """
    with h5py.File(src, "r") as f:
        obs = read_elem(f["obs"])
        rng = np.random.default_rng(seed)
        groups = obs.groupby(group_key if group_key in obs else "donor_id", observed=True).indices
        idx = np.sort(np.concatenate([rng.choice(ix, min(len(ix), max_cells), replace=False) for ix in groups.values()]))
        has_raw = "raw" in f
        X = sparse_dataset(f["raw/X"] if has_raw else f["X"])
        var = read_elem(f["raw/var"] if has_raw else f["var"])
        if "feature_name" not in var and "feature_name" in f["var"]:
            var["feature_name"] = read_elem(f["var"])["feature_name"].reindex(var.index)
        parts = []
        for start in range(0, X.shape[0], block):
            rows = idx[(idx >= start) & (idx < start + block)] - start
            if len(rows):
                parts.append(X[start:start + block][rows])
        adata = ad.AnnData(X=sp.vstack(parts).tocsr(), obs=obs.iloc[idx].copy(), var=var)
    del obs, parts
    gc.collect()

    vals = adata.X[:2000].data if hasattr(adata.X, "data") else np.asarray(adata.X[:2000]).ravel()
    if not np.allclose(vals % 1, 0):
        raise SystemExit("Expected raw integer counts in X (or .raw.X); found normalised values.")
    adata.var["ensembl_id"] = adata.var_names
    adata.var_names = adata.var["feature_name"].astype(str).values   # gene symbols, needed by CellTypist
    adata.var_names_make_unique()
    for c in ("donor_id", group_key):
        if c in adata.obs:
            adata.obs[c] = adata.obs[c].astype(str)
    return adata


def pick_batch_key(obs: pd.DataFrame, setting: str) -> str | None:
    if setting != "auto":
        return setting
    found = [c for c in BATCH_CANDIDATES if c in obs and obs[c].nunique() > 1]
    return found[0] if found else None


def qc(adata: ad.AnnData, q: dict) -> ad.AnnData:
    """Drop near-empty cells, likely dying cells (high mitochondrial fraction) and platelet-containing
    droplets (Perez: > 1 count of PF4, PPBP, GNG11, SDPR/CAVIN2), then rare genes."""
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    keep = (adata.obs["n_genes_by_counts"] >= q["min_genes"]) & (adata.obs["pct_counts_mt"] < q["max_pct_mt"])
    plt_genes = [g for g in q.get("platelet_genes", []) if g in adata.var_names]
    if plt_genes:
        plt_counts = np.asarray(adata[:, plt_genes].X.sum(1)).ravel()
        is_plt = plt_counts > q["max_platelet_counts"]
        print(f"  platelet filter ({', '.join(plt_genes)}): {is_plt.sum():,} cells")
        keep &= ~is_plt
    print(f"  QC keeps {keep.sum():,} of {len(keep):,} cells")
    adata._inplace_subset_obs(keep.values)   # in place: avoids a second copy of the matrix
    sc.pp.filter_genes(adata, min_cells=q["min_cells_per_gene"])
    return adata


def fine_labels(obs: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Perez's manual labels: the first non-missing of `columns` per cell (ct_cov has lymphoid subtypes, NaN for
    myeloid cells, where author_cell_type has the broad type). Falls back to CELLxGENE's cell_type."""
    out = pd.Series(np.nan, index=obs.index, dtype=object)
    for c in [c for c in columns if c in obs] + ["cell_type"]:
        v = obs[c].astype(object).where(obs[c].notna())
        v = v.where(~v.astype(str).isin(["nan", "na", "unknown", ""]))
        out = out.fillna(v)
    return out.fillna("unknown").astype(str)


def batch_report(obs: pd.DataFrame, batch_key: str | None, sample_key: str) -> pd.DataFrame:
    """Samples per batch x disease (and disease state). A batch holding only one disease is confounded with the label."""
    if batch_key is None:
        return pd.DataFrame()
    s = obs.drop_duplicates(sample_key)
    table = pd.crosstab(s[batch_key], s["disease"])
    table["single_disease"] = (table > 0).sum(axis=1) == 1
    if "disease_state" in s:
        table = table.join(pd.crosstab(s[batch_key], s["disease_state"]).add_prefix("state_"))
    return table


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    d, q = cfg["data"], cfg["qc"]
    os.makedirs(path(cfg["paths"]["results"]), exist_ok=True)

    print("[data] 1/4 download")
    src = download(cfg)
    print("[data] 2/4 subsample + raw counts")
    sample_key = d["sample_key"]
    adata = load_subsample(src, d["max_cells_per_sample"], cfg["train"]["seed"], sample_key)
    if sample_key not in adata.obs:   # files without a sample column: one sample per donor
        adata.obs[sample_key] = adata.obs[d["donor_key"]].astype(str)
    batch_key = pick_batch_key(adata.obs, d["batch_key"])
    adata.uns["batch_key"] = batch_key or ""
    print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes, {adata.obs[sample_key].nunique()} samples from "
          f"{adata.obs[d['donor_key']].nunique()} donors, batch_key={batch_key}")

    print("[data] 3/4 QC, Perez cell labels, batch report")
    adata = qc(adata, q)
    adata.obs["cell_type_fine"] = fine_labels(adata.obs, cfg["labels"]["fine"])
    adata.obs["lineage"] = (adata.obs["author_cell_type"].astype(str) if "author_cell_type" in adata.obs
                            else adata.obs["cell_type_fine"])
    print(adata.obs["cell_type_fine"].value_counts().to_string())
    report = batch_report(adata.obs, batch_key, sample_key)
    report.to_csv(path(os.path.join(cfg["paths"]["results"], "batch_report.csv")))
    if len(report):
        print(report)
        if report["single_disease"].any():
            print("  WARNING: some batches contain one disease only; batch and label are partly confounded")

    print("[data] 4/4 normalise + HVGs")
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=q["n_top_genes"], flavor="seurat_v3", layer="counts",
                                batch_key=batch_key)
    out = path(d["processed"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    adata.write_h5ad(out)
    print(f"[data] wrote {out}")


if __name__ == "__main__":
    main()
