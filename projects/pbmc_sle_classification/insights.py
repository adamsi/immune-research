"""Step 4: what the model looks at, which genes change, and example patients for the demo.

Input : data/processed/pbmc_rep.h5ad, artifacts/, results/oof_predictions.csv
Output: results/attention_by_celltype.csv  mean attention share per cell type, SLE vs healthy donors
        results/de_<cell type>.csv          PyDESeq2 SLE vs healthy on pseudobulk counts, most-attended cell types
        artifacts/examples/<donor>_<label>.h5ad  a few real donors in the format the app reads
"""
import os
import re

import anndata as ad
import numpy as np
import pandas as pd

from shared.common import load_config, parse_args
from shared.mil_numpy import load_weights, predict as mil_predict
from shared.plotting import attention_by_group

from . import DEFAULT_CONFIG, path
from .features import donor_labels, pseudobulk


def attention_table(adata, weights, sle_label: str) -> pd.DataFrame:
    """Attention share per cell type for every donor, then averaged within SLE and within healthy donors."""
    y = donor_labels(adata.obs, sle_label)
    Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    ct = adata.obs["cell_type_pred"].astype(str).to_numpy()
    rows = {}
    for donor, idx in adata.obs.groupby("donor_id", observed=True).indices.items():
        _, att = mil_predict(weights, Z[idx])
        rows[donor] = attention_by_group(att, ct[idx])
    shares = pd.DataFrame(rows).T.fillna(0.0)
    table = shares.groupby(y.loc[shares.index].map({1: "SLE", 0: "healthy"}).values).mean().T
    return table.sort_values("SLE", ascending=False)


def de_one_celltype(adata, cell_type: str, sle_label: str, batch_key: str | None, min_cells: int):
    """PyDESeq2, SLE vs healthy, on per-donor pseudobulk counts of one cell type (all genes).
    Batch is a covariate only when it is not confounded with disease (full-rank design)."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    sub = adata[adata.obs["cell_type_pred"] == cell_type]
    donors, counts, n_cells = pseudobulk(sub.layers["counts"], sub.obs["donor_id"])
    keep = n_cells >= min_cells
    donors, counts = donors[keep], counts[keep]
    y = donor_labels(sub.obs, sle_label).loc[donors]
    meta = pd.DataFrame({"disease": np.where(y.values == 1, "SLE", "healthy")}, index=donors)
    if meta["disease"].nunique() < 2 or meta["disease"].value_counts().min() < 3:
        return None
    design = "~disease"
    if batch_key:
        first = sub.obs.drop_duplicates("donor_id").set_index("donor_id")
        meta["batch"] = first.loc[donors, batch_key].astype(str).values
        X = pd.get_dummies(meta[["batch", "disease"]], drop_first=True).astype(float)
        if meta["batch"].nunique() > 1 and np.linalg.matrix_rank(np.c_[np.ones(len(X)), X.values]) == X.shape[1] + 1:
            design = "~batch + disease"
    df = pd.DataFrame(np.round(counts).astype(int), index=donors, columns=sub.var_names)
    df = df.loc[:, df.sum(axis=0) >= 10]
    dds = DeseqDataSet(counts=df, metadata=meta, design=design, quiet=True)
    dds.deseq2()
    stats = DeseqStats(dds, contrast=["disease", "SLE", "healthy"], quiet=True)
    stats.summary()
    return stats.results_df.sort_values("padj")


def write_examples(adata, cfg, art: str, res: str) -> None:
    """A few correctly predicted donors (out-of-fold), subsampled, with the genes the app needs and raw counts."""
    e = cfg["eval"]
    oof = pd.read_csv(os.path.join(res, "oof_predictions.csv"), index_col=0)
    oof.index = oof.index.astype(str)
    good = oof[(oof["mil"] >= 0.5) == (oof["y"] == 1)]
    n_sle = (e["n_examples"] + 1) // 2
    picks = list(good[good["y"] == 1].nlargest(n_sle, "mil").index) + \
        list(good[good["y"] == 0].nsmallest(e["n_examples"] - n_sle, "mil").index)
    ct = np.load(os.path.join(art, "celltypist.npz"))
    genes = open(os.path.join(art, "genes.txt")).read().split()
    keep_genes = list(dict.fromkeys(genes + [g for g in ct["features"] if g in adata.var_names]))
    out = os.path.join(art, "examples")
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    for donor in picks:
        idx = np.where(adata.obs["donor_id"].values == donor)[0]
        idx = np.sort(rng.choice(idx, min(len(idx), e["example_cells"]), replace=False))
        sub = adata[idx]
        ex = ad.AnnData(X=sub[:, keep_genes].layers["counts"].copy(),
                        obs=pd.DataFrame({"total_counts": np.asarray(sub.layers["counts"].sum(1)).ravel()},
                                         index=sub.obs_names),
                        var=pd.DataFrame(index=keep_genes))
        label = "SLE" if oof.loc[donor, "y"] == 1 else "healthy"
        name = re.sub(r"[^A-Za-z0-9_-]+", "-", donor)
        ex.write_h5ad(os.path.join(out, f"{name}_{label}.h5ad"), compression="gzip")
    print(f"  wrote {len(picks)} examples to {out}")


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    art, res = path(cfg["paths"]["artifacts"]), path(cfg["paths"]["results"])
    sle = cfg["data"]["sle_label"]
    adata = ad.read_h5ad(path(cfg["data"]["represented"]))
    batch_key = adata.uns.get("batch_key") or None

    print("[insights] 1/3 attention per cell type")
    table = attention_table(adata, load_weights(os.path.join(art, "model.npz")), sle)
    table.to_csv(os.path.join(res, "attention_by_celltype.csv"), index_label="cell_type")
    print((table * 100).round(1).head(10))

    print("[insights] 2/3 differential expression in the most-attended cell types")
    for ct in table.index[:cfg["eval"]["de_top_celltypes"]]:
        try:
            de = de_one_celltype(adata, ct, sle, batch_key, cfg["eval"]["min_cells_de"])
        except Exception as e:  # one cell type failing should not stop the rest
            print(f"  {ct}: DE failed ({e})")
            continue
        if de is None:
            print(f"  {ct}: too few donors, skipped")
            continue
        de.to_csv(os.path.join(res, f"de_{re.sub(r'[^A-Za-z0-9]+', '_', ct).strip('_')}.csv"))
        top = de[de["padj"] < 0.05]
        print(f"  {ct}: {len(top)} genes padj<0.05; top up: {', '.join(top[top.log2FoldChange > 0].index[:5])}")

    print("[insights] 3/3 example patients for the demo")
    write_examples(adata, cfg, art, res)


if __name__ == "__main__":
    main()
