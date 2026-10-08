"""Step 4: what the model looks at, which genes change, and example patients for the demo.

Input : data/processed/pbmc_rep.h5ad, artifacts/, results/oof_predictions.csv
Output: results/attention_by_celltype.csv  mean attention share per cell type, SLE vs healthy donors
        results/de_<cell type>.csv          PyDESeq2 SLE vs healthy on pseudobulk counts, most-attended cell types
        results/summary.json                findings shown in the app (attention, genes, composition, IFN score, batches)
        artifacts/examples/<donor>_<label>.h5ad  a few real donors in the format the app reads
"""
import glob
import json
import os
import re

import anndata as ad
import numpy as np
import pandas as pd

from shared.common import load_config, parse_args
from shared.mil_numpy import load_weights, predict as mil_predict
from shared.plotting import attention_by_group

from . import DEFAULT_CONFIG, path
from .features import donor_labels, log_cpm, pseudobulk

# type I interferon-stimulated genes, the best-known lupus blood signature
IFN_GENES = ["IFI27", "IFI44L", "ISG15", "MX1", "RSAD2", "IFI6", "EPSTI1", "IFIT1", "IFIT3", "OAS1"]


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


def write_summary(res: str, adata=None, sle_label: str | None = None) -> dict:
    """Collect the headline findings into results/summary.json (small, committed, read by the app).
    Attention, genes and batches come from the CSVs in results/; composition and the interferon
    score need the cells, so they are added only when adata is given."""
    from scipy.stats import mannwhitneyu, spearmanr
    from sklearn.metrics import roc_auc_score

    table = pd.read_csv(os.path.join(res, "attention_by_celltype.csv"), index_col=0)
    summary = {"attention": [{"cell_type": ct, "SLE": round(float(r["SLE"]), 4), "healthy": round(float(r["healthy"]), 4)}
                             for ct, r in table.head(8).iterrows()]}

    # differential expression: how many genes change per cell type, and the strongest ones
    de = []
    for f in sorted(glob.glob(os.path.join(res, "de_*.csv"))):
        d = pd.read_csv(f, index_col=0).dropna(subset=["padj"])
        sig = d[d["padj"] < 0.05]
        up = sig[sig["log2FoldChange"] > 0].sort_values("padj")
        down = sig[sig["log2FoldChange"] < 0].sort_values("padj")
        slug = os.path.basename(f)[3:-4]
        name = next((ct for ct in table.index if re.sub(r"[^A-Za-z0-9]+", "_", ct).strip("_") == slug), slug.replace("_", " "))
        if len(sig):
            de.append({"cell_type": name, "n_sig": int(len(sig)), "n_up": int(len(up)), "n_down": int(len(down)),
                       "top_up": [[g, round(float(v), 2)] for g, v in up["log2FoldChange"].head(8).items()],
                       "top_down": [[g, round(float(v), 2)] for g, v in down["log2FoldChange"].head(4).items()]})
    summary["de"] = sorted(de, key=lambda r: -r["n_sig"])

    bf = os.path.join(res, "batch_report.csv")
    if os.path.exists(bf):
        bt = pd.read_csv(bf, index_col=0)
        summary["batch_key"] = bt.index.name
        counts = bt.drop(columns="single_disease", errors="ignore")
        sle_col = sle_label if sle_label in counts else counts.columns[0]
        summary["batches"] = [{"batch": str(b).removesuffix(".0"), "SLE": int(r[sle_col]),
                               "healthy": int(r.drop(sle_col).sum())} for b, r in counts.iterrows()]

    if adata is not None:
        obs = adata.obs
        y = donor_labels(obs, sle_label)
        # share of each cell type per donor, SLE vs healthy
        frac = pd.crosstab(obs["donor_id"].astype(str), obs["cell_type_pred"].astype(str), normalize="index").loc[y.index]
        comp = []
        for ct in frac.columns:
            a, b = frac.loc[y == 1, ct], frac.loc[y == 0, ct]
            if max(a.mean(), b.mean()) >= 0.005:
                comp.append({"cell_type": ct, "SLE": round(float(a.mean()), 4), "healthy": round(float(b.mean()), 4),
                             "p": float(mannwhitneyu(a, b).pvalue)})
        summary["composition"] = sorted(comp, key=lambda r: r["p"])[:6]

        # interferon score per donor: mean z-scored log-CPM of the IFN genes, whole-blood pseudobulk
        genes = [g for g in IFN_GENES if g in adata.var_names]
        if genes:
            donors, counts, _ = pseudobulk(adata.layers["counts"], obs["donor_id"])
            lc = pd.DataFrame(log_cpm(counts), index=donors, columns=adata.var_names)[genes].loc[y.index]
            score = ((lc - lc.mean()) / lc.std().replace(0, 1)).mean(axis=1)
            ifn = {"genes": genes, "auroc": round(float(roc_auc_score(y, score)), 3),
                   "SLE": round(float(score[y == 1].mean()), 2), "healthy": round(float(score[y == 0].mean()), 2)}
            oof_f = os.path.join(res, "oof_predictions.csv")
            if os.path.exists(oof_f):
                oof = pd.read_csv(oof_f, index_col=0)
                oof.index = oof.index.astype(str)
                ifn["spearman_vs_mil"] = round(float(spearmanr(score, oof.loc[score.index, "mil"]).statistic), 2)
            summary["ifn"] = ifn

    with open(os.path.join(res, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


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

    print("[insights] 1/4 attention per cell type")
    table = attention_table(adata, load_weights(os.path.join(art, "model.npz")), sle)
    table.to_csv(os.path.join(res, "attention_by_celltype.csv"), index_label="cell_type")
    print((table * 100).round(1).head(10))

    print("[insights] 2/4 differential expression in the most-attended cell types")
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

    print("[insights] 3/4 summary for the app")
    s = write_summary(res, adata, sle)
    if "ifn" in s:
        print(f"  interferon score AUROC {s['ifn']['auroc']:.3f}")

    print("[insights] 4/4 example patients for the demo")
    write_examples(adata, cfg, art, res)


if __name__ == "__main__":
    main()
