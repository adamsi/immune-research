"""Step 3: donor-split cross-validation of both models on identical folds, then the final models.

Models  : baseline     pseudobulk (summed HVG counts per donor) -> log-CPM -> logistic regression
          attention-MIL each donor is a bag of cells in scVI space (shared/mil_torch.py)
Input   : data/processed/pbmc_rep.h5ad, artifacts/genes.txt
Output  : results/oof_predictions.csv   out-of-fold P(SLE) per donor for both models
          results/metrics.json          AUROC with 95% bootstrap CI for both models
          artifacts/model.npz           final MIL weights as numpy (used by the app), + model.pt, model_meta.json
          artifacts/baseline.joblib     final baseline pipeline (used by the app)
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from shared.common import load_config, parse_args, set_seed
from shared.metrics import auc_with_ci
from shared.mil_torch import AttentionMIL

from . import DEFAULT_CONFIG, path
from .features import donor_labels, log_cpm, pseudobulk


def make_baseline(C: float) -> Pipeline:
    return Pipeline([
        ("logcpm", FunctionTransformer(log_cpm)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=5000)),
    ])


def train_mil(bags, y, train_ids, cfg, seed):
    """One AttentionMIL on the given donor indices. Each epoch sees a fresh random subsample of each donor's cells."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    m, t = cfg["model"], cfg["train"]
    net = AttentionMIL(d_in=bags[0].shape[1], d=m["d"], d_att=m["d_att"], p_drop=m["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    for _ in range(t["epochs"]):
        net.train()
        for i in rng.permutation(train_ids):
            b = bags[i]
            idx = rng.choice(len(b), min(t["n_sub"], len(b)), replace=False)
            p, _ = net(torch.tensor(b[idx]))
            loss = nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), torch.tensor(float(y[i])))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net.eval()


@torch.no_grad()
def predict_mil(net, bag) -> float:
    return float(net(torch.tensor(bag))[0])


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    t = cfg["train"]
    art, res = path(cfg["paths"]["artifacts"]), path(cfg["paths"]["results"])
    os.makedirs(res, exist_ok=True)

    print("[train] 1/4 load")
    adata = ad.read_h5ad(path(cfg["data"]["represented"]))
    genes = open(os.path.join(art, "genes.txt")).read().split()
    labels = donor_labels(adata.obs, cfg["data"]["sle_label"])
    donors, pb, _ = pseudobulk(adata[:, genes].layers["counts"], adata.obs["donor_id"])
    y = labels.loc[donors].to_numpy()
    groups = adata.obs.groupby("donor_id", observed=True).indices
    Z = np.asarray(adata.obsm["X_scVI"], dtype=np.float32)
    bags = [Z[groups[d]] for d in donors]
    print(f"  {len(donors)} donors: {y.sum()} SLE, {(y == 0).sum()} healthy; {Z.shape[0]:,} cells x {Z.shape[1]} latent dims")

    print(f"[train] 2/4 cross-validation: {t['n_splits']} folds x {t['n_repeats']} repeats, same folds for both models")
    cv = RepeatedStratifiedKFold(n_splits=t["n_splits"], n_repeats=t["n_repeats"], random_state=t["seed"])
    oof_mil = np.full((t["n_repeats"], len(y)), np.nan)
    oof_base = np.full((t["n_repeats"], len(y)), np.nan)
    t0 = time.time()
    for k, (tr, te) in enumerate(cv.split(donors, y)):
        r = k // t["n_splits"]
        base = make_baseline(t["baseline_C"]).fit(pb[tr], y[tr])
        oof_base[r, te] = base.predict_proba(pb[te])[:, 1]
        net = train_mil(bags, y, tr, cfg, seed=t["seed"] + k)
        oof_mil[r, te] = [predict_mil(net, bags[i]) for i in te]
        print(f"  fold {k + 1}/{t['n_splits'] * t['n_repeats']} done, {time.time() - t0:.0f}s elapsed")

    df = pd.DataFrame({"y": y, "baseline": oof_base.mean(0), "mil": oof_mil.mean(0)}, index=donors)
    df.to_csv(os.path.join(res, "oof_predictions.csv"), index_label="donor")
    metrics = {"n_donors": int(len(y)), "n_sle": int(y.sum()), "n_cells": int(Z.shape[0]),
               "cv": f"{t['n_splits']}-fold x {t['n_repeats']}, donor-level"}
    for name in ("baseline", "mil"):
        auc, lo, hi = auc_with_ci(y, df[name].values, cfg["eval"]["n_boot"])
        metrics[name] = {"auroc": round(auc, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        print(f"  {name:8s} AUROC {auc:.3f} (95% CI {lo:.2f} to {hi:.2f})")
    json.dump(metrics, open(os.path.join(res, "metrics.json"), "w"), indent=2)

    print("[train] 3/4 final models on all donors")
    base = make_baseline(t["baseline_C"]).fit(pb, y)
    net = train_mil(bags, y, np.arange(len(y)), cfg, seed=t["seed"])

    print("[train] 4/4 save artifacts")
    joblib.dump(base, os.path.join(art, "baseline.joblib"))
    torch.save(net.state_dict(), os.path.join(art, "model.pt"))
    np.savez(os.path.join(art, "model.npz"), **{k: v.cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()})
    m = cfg["model"]
    json.dump({"d_in": int(Z.shape[1]), "d": m["d"], "d_att": m["d_att"]}, open(os.path.join(art, "model_meta.json"), "w"))
    print(f"[train] wrote {art}/model.npz, baseline.joblib and {res}/metrics.json")


if __name__ == "__main__":
    main()
