"""Step 4: classify each blood sample as healthy or one of the SLE subtypes, with two models on identical folds.

Models  : baseline  elastic-net logistic regression (Perez-style) on cell-type-resolved pseudobulk
                    (log-CPM per lineage) + cell-type abundance (CLR)
          MIL       multi-head gated attention-MIL over the sample's cells in scVI space, with the sample's
                    abundance vector fed to the head (shared/mil_torch.py)
          ensemble  mean of the two models' probabilities
Folds   : one bag per sample, folds split by donor (StratifiedGroupKFold), so a patient's flare and
          after-treatment samples are never on both sides. Also leave-one-cohort-out: train on all processing
          cohorts but one, test on it (measures how much the score depends on batch).
Input   : data/processed/pbmc_rep.h5ad, results/samples.csv, artifacts/genes.txt, artifacts/classes.json
Output  : results/oof_predictions.csv   out-of-fold class probabilities per sample for every model
          results/metrics.json          macro AUROC, SLE-vs-healthy AUROC (95% CI), accuracy, confusion, cohort CV
          artifacts/model.npz           final MIL weights as numpy (used by the app), + model.pt, model_meta.json
          artifacts/baseline.joblib     final baseline pipeline (used by the app)
          artifacts/baseline_features.json  how the app builds baseline and abundance features for a new sample
"""
import json
import os
import time

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.common import load_config, parse_args, set_seed
from shared.metrics import auc_with_ci, macro_auc_with_ci
from shared.mil_torch import AttentionMIL

from . import DEFAULT_CONFIG, path
from .features import baseline_features, composition, log_cpm, pseudobulk, to_dense


# ---------------------------------------------------------------- features

def feature_spec(adata, counts_hvg, sample_key: str, t: dict, min_frac: float = 0.001) -> dict:
    """Decide (label-free, on all samples) which cell types and genes the baseline and abundance features use."""
    ct = adata.obs["cell_type_fine"].astype(str)
    freq = ct.value_counts(normalize=True)
    cell_types = sorted(freq[freq >= min_frac].index)
    lin = adata.obs["lineage"].astype(str)
    lineage_of = pd.crosstab(ct, lin).idxmax(axis=1).to_dict()   # fine type -> its majority broad type
    lineages, gene_idx = [], []
    for L in [x for x in t["baseline_lineages"] if (lin == x).any()]:
        m = (lin == L).values
        _, pb, _ = pseudobulk(counts_hvg[m], adata.obs[sample_key].values[m])
        var = log_cpm(pb).var(axis=0)
        lineages.append(L)
        gene_idx.append(np.sort(np.argsort(-var)[:t["baseline_genes_per_lineage"]]).tolist())
    return {"cell_types": cell_types, "lineage_of": lineage_of, "lineages": lineages, "gene_idx": gene_idx}


def make_baseline(t: dict) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=t["baseline_C"], l1_ratio=t["baseline_l1_ratio"],   # 0 < l1_ratio < 1: elastic net
                                  solver="saga", class_weight="balanced", max_iter=3000, tol=1e-3)),
    ])


def proba_full(model, X, n_classes: int) -> np.ndarray:
    """predict_proba with a column for every class, also when a training fold lacked one."""
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = model.predict_proba(X)
    return out


# ---------------------------------------------------------------- MIL

def train_mil(bags, extras, y, train_ids, n_classes: int, cfg, seed):
    """One multi-class AttentionMIL. Each epoch sees a fresh random subsample of each bag's cells; gradients are
    accumulated over 8 bags; AdamW with cosine decay; class-weighted, label-smoothed cross-entropy."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    m, t = cfg["model"], cfg["train"]
    d_extra = extras.shape[1] if m["composition_input"] else 0
    net = AttentionMIL(d_in=bags[0].shape[1], d=m["d"], d_att=m["d_att"], p_drop=m["dropout"],
                       n_heads=m["n_heads"], n_classes=n_classes, d_extra=d_extra)
    opt = torch.optim.AdamW(net.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    counts = np.bincount(y[train_ids], minlength=n_classes).astype(np.float32)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_classes)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=t["label_smoothing"])
    E = torch.tensor(extras)
    for _ in range(t["epochs"]):
        net.train()
        opt.zero_grad()
        for j, i in enumerate(rng.permutation(train_ids), 1):
            b = bags[i]
            idx = rng.choice(len(b), min(t["n_sub"], len(b)), replace=False)
            logits, _ = net.logits(torch.tensor(b[idx]), E[i] if d_extra else None)
            (loss_fn(logits[None], torch.tensor([y[i]])) / 8).backward()
            if j % 8 == 0 or j == len(train_ids):
                opt.step()
                opt.zero_grad()
        sched.step()
    return net.eval()


@torch.no_grad()
def predict_mil(net, bag, extra) -> np.ndarray:
    p, _ = net(torch.tensor(bag), torch.tensor(extra) if net.d_extra else None)
    return p.numpy()


def fit_predict(tr, te, data, cfg):
    """Fit both models on samples tr, return class probabilities for samples te."""
    t, K = cfg["train"], data["K"]
    base = make_baseline(t).fit(data["Xb"][tr], data["y"][tr])
    pb = proba_full(base, data["Xb"][te], K)
    net = train_mil(data["bags"], data["extras"], data["y"], tr, K, cfg, seed=t["seed"] + int(te[0]))
    pm = np.stack([predict_mil(net, data["bags"][i], data["extras"][i]) for i in te])
    return pb, pm


# ---------------------------------------------------------------- metrics

def score(y, prob, sle, n_boot) -> dict:
    """Multi-class metrics plus SLE-vs-healthy AUROC (score = 1 - P(healthy)), each AUROC with a 95% CI."""
    auc, lo, hi = auc_with_ci(sle, 1 - prob[:, 0], n_boot)
    out = {"auroc": round(auc, 3), "ci95": [round(lo, 3), round(hi, 3)]}
    if prob.shape[1] > 2:
        mauc, mlo, mhi = macro_auc_with_ci(y, prob, n_boot)
        pred = prob.argmax(1)
        out.update({"macro_auroc": round(mauc, 3), "macro_ci95": [round(mlo, 3), round(mhi, 3)],
                    "accuracy": round(accuracy_score(y, pred), 3),
                    "balanced_accuracy": round(balanced_accuracy_score(y, pred), 3),
                    "confusion": confusion_matrix(y, pred, labels=range(prob.shape[1])).tolist()})
    return out


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    d, t = cfg["data"], cfg["train"]
    art, res = path(cfg["paths"]["artifacts"]), path(cfg["paths"]["results"])
    os.makedirs(res, exist_ok=True)

    print("[train] 1/5 load")
    adata = ad.read_h5ad(path(d["represented"]))
    genes = open(os.path.join(art, "genes.txt")).read().split()
    classes = json.load(open(os.path.join(art, "classes.json")))["classes"]
    samples = pd.read_csv(path(os.path.join(cfg["paths"]["results"], "samples.csv")), index_col=0, dtype={"sample": str})
    samples.index = samples.index.astype(str)
    K = len(classes)
    counts = adata[:, genes].layers["counts"]
    sid = adata.obs[d["sample_key"]].astype(str).values
    groups = pd.Series(np.arange(len(sid))).groupby(sid).indices
    ids = samples.index.to_numpy()
    ct = adata.obs["cell_type_fine"].astype(str).to_numpy()
    Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)

    print("[train] 2/5 features")
    spec = feature_spec(adata, counts, d["sample_key"], t)
    Xb = np.stack([baseline_features(to_dense(counts[groups[s]]), ct[groups[s]], spec) for s in ids])
    extras = np.stack([composition(ct[groups[s]], spec["cell_types"]) for s in ids])
    data = {"K": K, "y": samples["class"].to_numpy(int), "Xb": Xb, "extras": extras,
            "bags": [Z[groups[s]] for s in ids]}
    y, sle, donors = data["y"], samples["sle"].to_numpy(int), samples["donor"].astype(str).to_numpy()
    print(f"  {len(ids)} samples from {len(set(donors))} donors; classes: "
          + ", ".join(f"{c}={int((y == i).sum())}" for i, c in enumerate(classes)))
    print(f"  baseline: {Xb.shape[1]:,} features ({len(spec['lineages'])} lineages + {len(spec['cell_types'])} abundances); "
          f"MIL: {Z.shape[1]}-d cells + {extras.shape[1]} abundances")

    print(f"[train] 3/5 {t['n_splits']}-fold CV, folds split by donor, same folds for both models")
    oof = {m: np.full((len(y), K), np.nan) for m in ("baseline", "mil")}
    cv = StratifiedGroupKFold(n_splits=t["n_splits"], shuffle=True, random_state=t["seed"])
    t0 = time.time()
    for k, (tr, te) in enumerate(cv.split(ids, y, donors)):
        oof["baseline"][te], oof["mil"][te] = fit_predict(tr, te, data, cfg)
        print(f"  fold {k + 1}/{t['n_splits']} done, {time.time() - t0:.0f}s elapsed")
    oof["ensemble"] = (oof["baseline"] + oof["mil"]) / 2

    metrics = {"task": "healthy vs SLE subtypes", "classes": classes,
               "n_samples": int(len(y)), "n_donors": int(len(set(donors))), "n_sle": int(len(set(donors[sle == 1]))),
               "n_cells": int(Z.shape[0]), "cv": f"{t['n_splits']}-fold, split by donor"}
    for name, p in oof.items():
        metrics[name] = score(y, p, sle, cfg["eval"]["n_boot"])
        mm = metrics[name]
        print(f"  {name:8s} SLE AUROC {mm['auroc']:.3f} | macro AUROC {mm.get('macro_auroc', float('nan')):.3f} | "
              f"balanced acc {mm.get('balanced_accuracy', float('nan')):.3f}")

    if "ancestry" in samples:
        metrics["by_ancestry"] = {}
        for a in samples["ancestry"].unique():
            m = (samples["ancestry"] == a).to_numpy()
            if m.sum() >= 10 and len(set(sle[m])) == 2:
                metrics["by_ancestry"][a] = {n: round(roc_auc_score(sle[m], 1 - p[m, 0]), 3) for n, p in oof.items()}
        print(f"  by ancestry: {metrics['by_ancestry']}")

    if t["cohort_cv"] and "batch" in samples:
        print("[train] 4/5 leave-one-cohort-out")
        metrics["cohort_cv"] = {}
        b = samples["batch"].astype(str).to_numpy()
        for c in sorted(set(b)):
            te, tr = np.where(b == c)[0], np.where(b != c)[0]
            if len(set(sle[te])) < 2:
                print(f"  cohort {c}: one group only, not a test cohort")
                continue
            pb, pm = fit_predict(tr, te, data, cfg)
            r = {"n": int(len(te)), "baseline": round(roc_auc_score(sle[te], 1 - pb[:, 0]), 3),
                 "mil": round(roc_auc_score(sle[te], 1 - pm[:, 0]), 3),
                 "ensemble": round(roc_auc_score(sle[te], 1 - (pb[:, 0] + pm[:, 0]) / 2), 3)}
            metrics["cohort_cv"][c.removesuffix(".0")] = r
            print(f"  held-out cohort {c}: {r}")

    df = pd.DataFrame({"donor": donors, "y": y, "class": [classes[i] for i in y], "sle": sle}, index=ids)
    for name, p in oof.items():
        df[name] = 1 - p[:, 0]                                  # P(SLE), the binary view
        for i, c in enumerate(classes):
            df[f"{name}:{c}"] = p[:, i]
    df.to_csv(os.path.join(res, "oof_predictions.csv"), index_label="sample")
    json.dump(metrics, open(os.path.join(res, "metrics.json"), "w"), indent=2)

    print("[train] 5/5 final models on all samples")
    base = make_baseline(t).fit(Xb, y)
    net = train_mil(data["bags"], extras, y, np.arange(len(y)), K, cfg, seed=t["seed"])
    joblib.dump(base, os.path.join(art, "baseline.joblib"))
    json.dump(spec, open(os.path.join(art, "baseline_features.json"), "w"))
    torch.save(net.state_dict(), os.path.join(art, "model.pt"))
    np.savez(os.path.join(art, "model.npz"), **{k: v.cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()})
    m = cfg["model"]
    json.dump({"d_in": int(Z.shape[1]), "d": m["d"], "d_att": m["d_att"], "n_heads": m["n_heads"], "n_classes": K,
               "d_extra": int(extras.shape[1]) if m["composition_input"] else 0},
              open(os.path.join(art, "model_meta.json"), "w"))
    print(f"[train] wrote {art}/model.npz, baseline.joblib, baseline_features.json and {res}/metrics.json")


if __name__ == "__main__":
    main()
