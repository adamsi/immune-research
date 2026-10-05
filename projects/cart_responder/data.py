"""Step 1: download GSE267097, load the 10x libraries, normalise, select genes, cache to disk.

Input : nothing (downloads from GEO), configs/default.yaml
Output: data/processed/dataset.h5ad  (cells x 2,000 genes, log-normalised, with labels in .obs)

Memory-lean: each library is read, subset to the authors' cells, normalised and log-transformed
before the next one is loaded. Download and extraction are skipped if already done;
Make decides whether the script needs to run at all.
"""
import gc
import glob
import os
import tarfile

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from shared.common import load_config, parse_args
from shared.data import download

from . import DEFAULT_CONFIG, path

GEO_FTP = "https://ftp.ncbi.nlm.nih.gov/geo/series/{series}nnn/{acc}/suppl/"


def load_library(raw_dir: str, lib: str, meta: pd.DataFrame):
    """Read one 10x library, keep only cells present in the authors' metadata, normalise."""
    m = meta[meta["Library"] == lib]
    if len(m) == 0:
        return None
    bc2name = dict(zip(m["BARCODE"], m.index))          # barcode -> authors' cell name
    mtx = glob.glob(os.path.join(raw_dir, f"*_{lib}_matrix.mtx.gz"))
    if not mtx:
        print(f"  WARNING: no matrix file for {lib}")
        return None
    prefix = os.path.basename(mtx[0]).replace("matrix.mtx.gz", "")
    a = sc.read_10x_mtx(raw_dir, prefix=prefix, gex_only=True)
    a = a[a.obs_names.isin(bc2name)].copy()
    a.obs_names = [bc2name[b] for b in a.obs_names]
    a.var_names_make_unique()
    a.obs["total_counts"] = np.asarray(a.X.sum(1)).ravel()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    acc, raw_dir, out = cfg["data"]["geo"], path(cfg["data"]["raw_dir"]), path(cfg["data"]["processed"])
    base = GEO_FTP.format(series=acc[:-3], acc=acc)
    meta_gz, tar_path = os.path.join(raw_dir, f"{acc}_GEO_metadata.tsv.gz"), os.path.join(raw_dir, f"{acc}_RAW.tar")
    print("[data] 1/4 download")
    download(base + f"{acc}_GEO_metadata.tsv.gz", meta_gz)
    download(base + f"{acc}_RAW.tar", tar_path)

    print("[data] 2/4 extract")
    if not glob.glob(os.path.join(raw_dir, "*_matrix.mtx.gz")):
        with tarfile.open(tar_path) as t:
            t.extractall(raw_dir)
    else:
        print("  already extracted")

    print("[data] 3/4 load libraries")
    meta = pd.read_csv(meta_gz, sep="\t", index_col=0)
    meta = meta[meta["status"] == "Patient"]                 # healthy donors are not part of the task
    parts = []
    for lib in sorted(meta["Library"].unique()):
        a = load_library(raw_dir, lib, meta)
        if a is None:
            continue
        parts.append(a)
        print(f"  {lib}: {a.n_obs:,} cells")
        gc.collect()
    adata = ad.concat(parts, join="inner")
    del parts
    gc.collect()

    print("[data] 4/4 labels, gene selection, save")
    adata.obs = adata.obs.join(meta.drop(columns=[c for c in meta.columns if c in adata.obs.columns]))
    adata.obs["y"] = (adata.obs[cfg["data"]["label_column"]] == "R").astype(int)
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg["data"]["n_top_genes"], batch_key="Library")
    adata = adata[:, adata.var["highly_variable"]].copy()
    for col in ("Patient_ID", "annotated_clusters", "Library", "Product"):
        adata.obs[col] = adata.obs[col].astype(str)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    adata.write_h5ad(out)
    n_pat = adata.obs["Patient_ID"].nunique()
    n_resp = adata.obs.groupby("Patient_ID")["y"].first().sum()
    print(f"[data] wrote {out}: {adata.n_obs:,} cells x {adata.n_vars:,} genes, {n_pat} patients, {n_resp} responders")


if __name__ == "__main__":
    main()
