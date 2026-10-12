"""Step 5: what the model looks at, which cells and genes change, what treatment reverses, and demo examples.

Input : data/processed/pbmc_rep.h5ad, artifacts/, results/samples.csv, results/oof_predictions.csv
Output: results/attention_by_celltype.csv  mean attention share per cell type, per class (healthy, each subtype, all SLE)
        results/composition.csv            Perez's abundance test per cell type (weighted least squares, Bonferroni)
        results/de_<cell type>.csv          PyDESeq2 SLE vs healthy on pseudobulk counts, most-attended cell types
        results/paired_composition.csv     flare -> after treatment, same patients: abundance change per cell type
        results/paired_de_<lineage>.csv     flare -> after treatment, paired PyDESeq2 (~ donor + state)
        results/summary.json                findings shown in the app
        artifacts/examples/<sample>_<class>.h5ad  a few real samples in the format the app reads
"""
import glob
import json
import os
import re

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

from shared.common import load_config, parse_args
from shared.mil_numpy import load_weights, predict as mil_predict
from shared.plotting import attention_by_group

from . import DEFAULT_CONFIG, path
from .features import composition, log_cpm, pseudobulk


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def one_per_donor(samples: pd.DataFrame) -> pd.DataFrame:
    """One sample per donor for between-person tests: drop after-treatment samples, then keep the first per donor."""
    s = samples[samples.get("disease_state", pd.Series("", index=samples.index)).astype(str) != "treated"]
    return s[~s["donor"].duplicated()]


def attention_table(adata, weights, samples, classes, spec, sample_key) -> pd.DataFrame:
    """Attention share per cell type for every sample, then averaged within each class and within all SLE."""
    Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    ct = adata.obs["cell_type_fine"].astype(str).to_numpy()
    use_extra = weights["head.weight"].shape[1] > weights["w.weight"].shape[0] * weights["enc.0.weight"].shape[0]
    rows = {}
    for s, idx in adata.obs.groupby(sample_key, observed=True).indices.items():
        _, att = mil_predict(weights, Z[idx], composition(ct[idx], spec["cell_types"]) if use_extra else None)
        rows[str(s)] = attention_by_group(att, ct[idx])
    shares = pd.DataFrame(rows).T.fillna(0.0)
    cls = samples.loc[shares.index, "endotype"]
    table = shares.groupby(cls.values).mean().T.reindex(columns=classes)
    table["SLE"] = shares[samples.loc[shares.index, "sle"].values == 1].mean()
    return table.sort_values("SLE", ascending=False)


def composition_test(adata, samples, sample_key) -> pd.DataFrame:
    """Perez's test: per cell type, % of the sample's cells ~ SLE by weighted least squares (weights = cell counts),
    one sample per donor, Bonferroni-adjusted."""
    s = one_per_donor(samples)
    frac = pd.crosstab(adata.obs[sample_key].astype(str), adata.obs["cell_type_fine"].astype(str), normalize="index").loc[s.index]
    w = s["n_cells"].to_numpy(float)
    X = np.c_[np.ones(len(s)), s["sle"].to_numpy(float)]
    rows = []
    for ct in frac.columns:
        yv = frac[ct].to_numpy() * 100
        W = np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(X * W, yv * W[:, 0], rcond=None)
        resid = (yv - X @ beta) * W[:, 0]
        dof = len(yv) - 2
        cov = np.linalg.inv((X * W).T @ (X * W)) * (resid @ resid) / dof
        tval = beta[1] / np.sqrt(cov[1, 1])
        rows.append({"cell_type": ct, "healthy_pct": round(float(frac.loc[s["sle"] == 0, ct].mean() * 100), 2),
                     "SLE_pct": round(float(frac.loc[s["sle"] == 1, ct].mean() * 100), 2),
                     "diff_pct": round(float(beta[1]), 2), "p": float(2 * stats.t.sf(abs(tval), dof))})
    out = pd.DataFrame(rows).set_index("cell_type")
    out["p_bonf"] = np.minimum(out["p"] * len(out), 1.0)
    return out.sort_values("p")


def deseq(counts: pd.DataFrame, meta: pd.DataFrame, design: str, contrast: list[str]) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts = counts.loc[:, counts.sum(axis=0) >= 10]
    dds = DeseqDataSet(counts=counts, metadata=meta, design=design, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=True)
    st.summary()
    return st.results_df.sort_values("padj")


def de_one_celltype(adata, cell_type: str, samples: pd.DataFrame, sample_key: str, min_cells: int):
    """PyDESeq2, SLE vs healthy, on per-sample pseudobulk counts of one cell type (all genes), one sample per donor.
    Batch is a covariate only when it is not confounded with disease (full-rank design)."""
    sub = adata[adata.obs["cell_type_fine"] == cell_type]
    ids, counts, n_cells = pseudobulk(sub.layers["counts"], sub.obs[sample_key])
    keep_ids = set(one_per_donor(samples).index)
    keep = (n_cells >= min_cells) & np.isin(ids, list(keep_ids))
    ids, counts = ids[keep], counts[keep]
    meta = pd.DataFrame({"disease": np.where(samples.loc[ids, "sle"].values == 1, "SLE", "healthy")}, index=ids)
    if meta["disease"].nunique() < 2 or meta["disease"].value_counts().min() < 3:
        return None
    design = "~disease"
    if "batch" in samples:
        meta["batch"] = samples.loc[ids, "batch"].astype(str).values
        X = pd.get_dummies(meta[["batch", "disease"]], drop_first=True).astype(float)
        if meta["batch"].nunique() > 1 and np.linalg.matrix_rank(np.c_[np.ones(len(X)), X.values]) == X.shape[1] + 1:
            design = "~batch + disease"
    df = pd.DataFrame(np.round(counts).astype(int), index=ids, columns=sub.var_names)
    return deseq(df, meta, design, ["disease", "SLE", "healthy"])


def paired_flare_treated(adata, samples, sample_key, res: str, lineages: list[str], min_cells: int) -> dict | None:
    """Same patients at flare and after treatment: abundance change (Wilcoxon signed-rank) and paired DE
    (PyDESeq2, ~ donor + state). No batch confounding: both samples of a pair are in the same cohort."""
    if "disease_state" not in samples:
        return None
    st = samples["disease_state"].astype(str)
    pairs = [d for d, g in samples.groupby("donor") if {"flare", "treated"} <= set(st[g.index])]
    if len(pairs) < 3:
        return None
    sel = samples[samples["donor"].isin(pairs) & st.isin(["flare", "treated"])]
    sel = sel[~sel.duplicated(["donor", "disease_state"])]
    frac = pd.crosstab(adata.obs[sample_key].astype(str), adata.obs["cell_type_fine"].astype(str), normalize="index").loc[sel.index]
    rows = []
    for ct in frac.columns:
        a = frac.loc[sel.index[sel["disease_state"] == "flare"], ct].groupby(sel["donor"]).first()
        b = frac.loc[sel.index[sel["disease_state"] == "treated"], ct].groupby(sel["donor"]).first()
        a, b = a.align(b, join="inner")
        if (a - b).abs().sum() == 0:
            continue
        rows.append({"cell_type": ct, "flare_pct": round(a.mean() * 100, 2), "treated_pct": round(b.mean() * 100, 2),
                     "p": float(stats.wilcoxon(a, b).pvalue)})
    comp = pd.DataFrame(rows).set_index("cell_type").sort_values("p")
    comp.to_csv(os.path.join(res, "paired_composition.csv"))

    de = []
    sub_all = adata[adata.obs[sample_key].astype(str).isin(sel.index)]
    for lin in lineages:
        sub = sub_all[sub_all.obs["lineage"].astype(str) == lin]
        ids, counts, n = pseudobulk(sub.layers["counts"], sub.obs[sample_key])
        ok = n >= min_cells
        ids, counts = ids[ok], counts[ok]
        meta = sel.loc[ids, ["donor", "disease_state"]].rename(columns={"disease_state": "state"}).astype(str)
        both = meta.groupby("donor")["state"].nunique() == 2
        m = meta["donor"].map(both).to_numpy()
        if both.sum() < 3:
            continue
        try:
            r = deseq(pd.DataFrame(np.round(counts[m]).astype(int), index=ids[m], columns=sub.var_names), meta[m],
                      "~donor + state", ["state", "treated", "flare"])
        except Exception as e:  # one lineage failing should not stop the rest
            print(f"  paired DE {lin}: failed ({e})")
            continue
        r.to_csv(os.path.join(res, f"paired_de_{slug(lin)}.csv"))
        sig = r[r["padj"] < 0.05]
        de.append({"lineage": lin, "n_sig": int(len(sig)),
                   "top_down": [[g, round(float(v), 2)] for g, v in sig[sig["log2FoldChange"] < 0]["log2FoldChange"].head(6).items()],
                   "top_up": [[g, round(float(v), 2)] for g, v in sig[sig["log2FoldChange"] > 0]["log2FoldChange"].head(4).items()]})
    return {"n_pairs": len(pairs),
            "composition": [{"cell_type": ct, "flare": float(r["flare_pct"]), "treated": float(r["treated_pct"]), "p": r["p"]}
                            for ct, r in comp.head(6).iterrows()],
            "de": de}


def write_summary(res: str, art: str, adata=None, samples=None, cfg=None, ifn_genes=()) -> dict:
    """Collect the headline findings into results/summary.json (small, committed, read by the app)."""
    from sklearn.metrics import roc_auc_score

    table = pd.read_csv(os.path.join(res, "attention_by_celltype.csv"), index_col=0)
    summary = {"attention": [{"cell_type": ct, "SLE": round(float(r["SLE"]), 4), "healthy": round(float(r["healthy"]), 4)}
                             for ct, r in table.head(8).iterrows()]}
    cj = os.path.join(art, "classes.json")
    if os.path.exists(cj):
        c = json.load(open(cj))
        prof_f = os.path.join(res, "endotype_profiles.csv")
        prof = pd.read_csv(prof_f, index_col=0) if os.path.exists(prof_f) else None
        summary["endotypes"] = [
            {"name": k, "n": int(c["sizes"].get(k, 0)), "description": c["descriptions"].get(k, ""),
             "attention": [[ct, round(float(v), 3)] for ct, v in table[k].sort_values(ascending=False).head(3).items()] if k in table else [],
             "profile": [[f, round(float(v), 2)] for f, v in prof.loc[k].reindex(prof.loc[k].abs().sort_values(ascending=False).index[:5]).items()]
             if prof is not None and k in prof.index else []}
            for k in c["classes"] if k != "healthy"]

    de = []
    for f in sorted(glob.glob(os.path.join(res, "de_*.csv"))):
        d = pd.read_csv(f, index_col=0).dropna(subset=["padj"])
        sig = d[d["padj"] < 0.05]
        up = sig[sig["log2FoldChange"] > 0].sort_values("padj")
        down = sig[sig["log2FoldChange"] < 0].sort_values("padj")
        name = next((ct for ct in table.index if slug(ct) == os.path.basename(f)[3:-4]), os.path.basename(f)[3:-4])
        if len(sig):
            de.append({"cell_type": name, "n_sig": int(len(sig)), "n_up": int(len(up)), "n_down": int(len(down)),
                       "top_up": [[g, round(float(v), 2)] for g, v in up["log2FoldChange"].head(8).items()],
                       "top_down": [[g, round(float(v), 2)] for g, v in down["log2FoldChange"].head(4).items()]})
    summary["de"] = sorted(de, key=lambda r: -r["n_sig"])

    bf = os.path.join(res, "batch_report.csv")
    if os.path.exists(bf):
        bt = pd.read_csv(bf, index_col=0)
        summary["batch_key"] = bt.index.name
        counts = bt.drop(columns=[c for c in bt.columns if c == "single_disease" or c.startswith("state_")])
        sle_col = cfg["data"]["sle_label"] if cfg and cfg["data"]["sle_label"] in counts else counts.columns[-1]
        summary["batches"] = [{"batch": str(b).removesuffix(".0"), "SLE": int(r[sle_col]),
                               "healthy": int(r.drop(sle_col).sum())} for b, r in counts.iterrows()]

    cf = os.path.join(res, "composition.csv")
    if os.path.exists(cf):
        comp = pd.read_csv(cf, index_col=0)
        summary["composition"] = [{"cell_type": ct, "SLE": round(r["SLE_pct"] / 100, 4), "healthy": round(r["healthy_pct"] / 100, 4),
                                   "p": float(r["p_bonf"])} for ct, r in comp.head(6).iterrows()]

    pf = os.path.join(res, "paired.json")
    if os.path.exists(pf):
        summary["paired"] = json.load(open(pf))

    if adata is not None and samples is not None:
        # interferon score per sample: mean z-scored log-CPM of the IFN genes, whole-blood pseudobulk
        genes = [g for g in ifn_genes if g in adata.var_names]
        if genes:
            s = one_per_donor(samples)
            ids, counts, _ = pseudobulk(adata.layers["counts"], adata.obs[cfg["data"]["sample_key"]])
            lc = pd.DataFrame(log_cpm(counts), index=ids, columns=adata.var_names)[genes].loc[s.index]
            score = ((lc - lc.mean()) / lc.std().replace(0, 1)).mean(axis=1)
            summary["ifn"] = {"genes": genes, "auroc": round(float(roc_auc_score(s["sle"], score)), 3),
                              "SLE": round(float(score[s["sle"] == 1].mean()), 2),
                              "healthy": round(float(score[s["sle"] == 0].mean()), 2)}

    with open(os.path.join(res, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    return summary


def write_examples(adata, cfg, art: str, res: str, classes: list[str]) -> None:
    """One or more correctly predicted samples per class (out-of-fold ensemble), subsampled, with the genes the app
    needs, raw counts and Perez's cell labels."""
    e, sample_key = cfg["eval"], cfg["data"]["sample_key"]
    oof = pd.read_csv(os.path.join(res, "oof_predictions.csv"), index_col=0)
    oof.index = oof.index.astype(str)
    P = oof[[f"ensemble:{c}" for c in classes]].to_numpy()
    oof["pred"], oof["conf"] = P.argmax(1), P.max(1)
    good = oof[oof["pred"] == oof["y"]]
    picks = []
    for i in range(len(classes)):
        picks += list(good[good["y"] == i].nlargest(max(1, e["n_examples"] // len(classes)), "conf").index)
    ct = np.load(os.path.join(art, "celltypist.npz"))
    genes = open(os.path.join(art, "genes.txt")).read().split()
    keep_genes = list(dict.fromkeys(genes + [g for g in ct["features"] if g in adata.var_names]))
    out = os.path.join(art, "examples")
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(out, "*.h5ad")):
        os.remove(f)
    rng = np.random.default_rng(0)
    sid = adata.obs[sample_key].astype(str).values
    for s in picks:
        idx = np.where(sid == s)[0]
        idx = np.sort(rng.choice(idx, min(len(idx), e["example_cells"]), replace=False))
        sub = adata[idx]
        ex = ad.AnnData(X=sub[:, keep_genes].layers["counts"].copy(),
                        obs=pd.DataFrame({"total_counts": np.asarray(sub.layers["counts"].sum(1)).ravel(),
                                          "cell_type": sub.obs["cell_type_fine"].astype(str).values}, index=sub.obs_names),
                        var=pd.DataFrame(index=keep_genes))
        ex.write_h5ad(os.path.join(out, f"{slug(s)[:12]}_{classes[int(oof.loc[s, 'y'])]}.h5ad"), compression="gzip")
    print(f"  wrote {len(picks)} examples to {out}")


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    art, res = path(cfg["paths"]["artifacts"]), path(cfg["paths"]["results"])
    sample_key = cfg["data"]["sample_key"]
    adata = ad.read_h5ad(path(cfg["data"]["represented"]))
    samples = pd.read_csv(os.path.join(res, "samples.csv"), index_col=0, dtype={"sample": str})
    samples.index = samples.index.astype(str)
    classes = json.load(open(os.path.join(art, "classes.json")))["classes"]
    spec = json.load(open(os.path.join(art, "baseline_features.json")))

    print("[insights] 1/6 attention per cell type and class")
    table = attention_table(adata, load_weights(os.path.join(art, "model.npz")), samples, classes, spec, sample_key)
    table.to_csv(os.path.join(res, "attention_by_celltype.csv"), index_label="cell_type")
    print((table * 100).round(1).head(10))

    print("[insights] 2/6 cell-type abundance, SLE vs healthy (Perez's weighted least squares)")
    comp = composition_test(adata, samples, sample_key)
    comp.to_csv(os.path.join(res, "composition.csv"))
    print(comp.head(8))

    print("[insights] 3/6 differential expression in the most-attended cell types")
    for f in glob.glob(os.path.join(res, "de_*.csv")):
        os.remove(f)
    for ct in table.index[:cfg["eval"]["de_top_celltypes"]]:
        try:
            de = de_one_celltype(adata, ct, samples, sample_key, cfg["eval"]["min_cells_de"])
        except Exception as e:  # one cell type failing should not stop the rest
            print(f"  {ct}: DE failed ({e})")
            continue
        if de is None:
            print(f"  {ct}: too few samples, skipped")
            continue
        de.to_csv(os.path.join(res, f"de_{slug(ct)}.csv"))
        top = de[de["padj"] < 0.05]
        print(f"  {ct}: {len(top)} genes padj<0.05; top up: {', '.join(top[top.log2FoldChange > 0].index[:5])}")

    print("[insights] 4/6 flare -> after treatment, same patients")
    paired = paired_flare_treated(adata, samples, sample_key, res, cfg["train"]["baseline_lineages"], cfg["eval"]["min_cells_de"])
    if paired:
        json.dump(paired, open(os.path.join(res, "paired.json"), "w"), indent=1)
        print(f"  {paired['n_pairs']} pairs; " + "; ".join(f"{d['lineage']}: {d['n_sig']} genes" for d in paired["de"]))
    else:
        print("  fewer than 3 flare/treated pairs, skipped")

    print("[insights] 5/6 summary for the app")
    s = write_summary(res, art, adata, samples, cfg, cfg["endotypes"]["ifn_genes"])
    if "ifn" in s:
        print(f"  interferon score AUROC {s['ifn']['auroc']:.3f}")

    print("[insights] 6/6 example samples for the demo")
    write_examples(adata, cfg, art, res, classes)


if __name__ == "__main__":
    main()
