"""Step 3: metrics, interpretability, and example patients for the demo.

Input : results/oof_predictions.csv, artifacts/, data/processed/dataset.h5ad
Output: results/metrics.json                 AUC with 95% bootstrap CI
        results/attention_by_celltype.csv    mean attention share per cell type, responders vs non-responders
        examples/<patient>_<label>.csv       a few real patients in the format app/app.py reads
"""
import glob
import json
import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from shared.common import build_bags, load_config, parse_args
from shared.data import dense_float32
from shared.mil_torch import AttentionMIL
from shared.plotting import attention_by_group

from . import DEFAULT_CONFIG, path


def auc_with_ci(y, prob, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        s = rng.integers(0, len(y), len(y))
        if len(set(y[s])) > 1:
            boot.append(roc_auc_score(y[s], prob[s]))
    return roc_auc_score(y, prob), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    art, res, ex = (path(cfg["paths"][k]) for k in ("artifacts", "results", "examples"))
    os.makedirs(res, exist_ok=True)
    os.makedirs(ex, exist_ok=True)

    print("[evaluate] 1/3 AUC from out-of-fold predictions")
    oof = pd.read_csv(os.path.join(res, "oof_predictions.csv"), index_col="patient")
    y = oof["y"].values
    per_repeat = [roc_auc_score(y, oof[c]) for c in oof.columns if c.startswith("repeat_")]
    auc, lo, hi = auc_with_ci(y, oof["prob_mean"].values, cfg["eval"]["n_boot"])
    metrics = {"n_patients": int(len(y)), "n_responders": int(y.sum()), "auc": round(auc, 4),
               "ci95": [round(lo, 4), round(hi, 4)], "auc_per_repeat": [round(a, 4) for a in per_repeat],
               "config": cfg["train"] | cfg["model"]}
    json.dump(metrics, open(os.path.join(res, "metrics.json"), "w"), indent=2)
    print(f"  AUC {auc:.3f}   95% CI {lo:.3f} to {hi:.3f}   (per repeat: {', '.join(f'{a:.2f}' for a in per_repeat)})")

    print("[evaluate] 2/3 attention by cell type")
    adata = ad.read_h5ad(path(cfg["data"]["processed"]))
    pca = pickle.load(open(os.path.join(art, "pca.pkl"), "rb"))
    meta = json.load(open(os.path.join(art, "model_meta.json")))
    net = AttentionMIL(**meta)
    net.load_state_dict(torch.load(os.path.join(art, "model.pt"), map_location="cpu"))
    net.eval()
    X = dense_float32(adata.X)
    Z = pca.transform(X).astype(np.float32)
    patients, bags, yb, ctype = build_bags(adata.obs, Z)
    rows = []
    with torch.no_grad():
        for p in patients:
            _, a = net(torch.tensor(bags[p]))
            rows.append(attention_by_group(a.numpy(), ctype[p]).rename(p))
    att = pd.DataFrame(rows).fillna(0)
    att["y"] = yb
    table = att.groupby("y").mean().T
    table.columns = ["non_responder_mean_share", "responder_mean_share"]
    table["ratio_R_over_NR"] = table["responder_mean_share"] / table["non_responder_mean_share"].clip(1e-9)
    table = table.sort_values("responder_mean_share", ascending=False)
    table.to_csv(os.path.join(res, "attention_by_celltype.csv"), index_label="cell_type")
    print(table.round(3).to_string())

    print("[evaluate] 3/3 export example patients for the demo")
    for f in glob.glob(os.path.join(ex, "*.csv")):
        os.remove(f)
    totals = adata.obs["total_counts"].values.astype(np.float64)
    counts = np.rint(np.expm1(X) * totals[:, None] / 1e4).astype(np.int32)   # invert normalize_total + log1p
    pat = adata.obs["Patient_ID"].values
    resp, nonresp = patients[yb == 1], patients[yb == 0]
    order = [resp[0], nonresp[0]] + list(resp[1:]) + list(nonresp[1:])
    for p in order[: cfg["eval"]["n_examples"]]:
        m = pat == p
        df = pd.DataFrame(counts[m], columns=adata.var_names, index=adata.obs_names[m])
        df.insert(0, "cell_type", adata.obs.loc[m, "annotated_clusters"].values)
        df.insert(1, "total_counts", totals[m].astype(int))
        label = "responder" if yb[list(patients).index(p)] == 1 else "non_responder"
        df.to_csv(os.path.join(ex, f"{p}_{label}.csv"))
        print(f"  {p}_{label}.csv: {m.sum():,} cells")
    print(f"[evaluate] wrote {res}/metrics.json, attention_by_celltype.csv and {ex}/")


if __name__ == "__main__":
    main()
