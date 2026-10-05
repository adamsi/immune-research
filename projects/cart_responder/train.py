"""Step 2: PCA, repeated patient-level cross-validation, then the final model on all patients.

Input : data/processed/dataset.h5ad, configs/default.yaml
Output: results/oof_predictions.csv   out-of-fold probability per patient and repeat
        artifacts/model.pt            final weights (torch)
        artifacts/model.npz           same weights as numpy arrays, used by the app
        artifacts/pca.pkl             fitted PCA (needed to embed new patients)
        artifacts/genes.txt           the 2,000 gene names, in order
        artifacts/model_meta.json     layer sizes, so the app can rebuild the network
"""
import json
import os
import pickle
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold

from shared.common import build_bags, load_config, parse_args, set_seed
from shared.data import dense_float32
from shared.mil_torch import AttentionMIL

from . import DEFAULT_CONFIG, path


def train_model(bags, patients, y, train_ids, cfg, seed):
    """Train one AttentionMIL on the given patient indices. Each epoch sees a fresh cell subsample."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    m, t = cfg["model"], cfg["train"]
    net = AttentionMIL(d_in=m["n_pcs"], d=m["d"], d_att=m["d_att"], p_drop=m["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    for _ in range(t["epochs"]):
        net.train()
        for i in rng.permutation(train_ids):
            b = bags[patients[i]]
            idx = rng.choice(len(b), min(t["n_sub"], len(b)), replace=False)
            p, _ = net(torch.tensor(b[idx]))
            loss = nn.functional.binary_cross_entropy(p, torch.tensor(float(y[i])))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def predict(net, bag):
    net.eval()
    with torch.no_grad():
        p, a = net(torch.tensor(bag))
    return float(p), a.numpy()


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    art, res = path(cfg["paths"]["artifacts"]), path(cfg["paths"]["results"])
    os.makedirs(art, exist_ok=True)
    os.makedirs(res, exist_ok=True)
    print("[train] 1/4 load + PCA")
    adata = ad.read_h5ad(path(cfg["data"]["processed"]))
    X = dense_float32(adata.X)
    # PCA is fit once on all cells. It is unsupervised (never sees labels), a standard simplification.
    pca = PCA(n_components=cfg["model"]["n_pcs"], random_state=cfg["train"]["seed"]).fit(X)
    Z = pca.transform(X).astype(np.float32)
    del X
    patients, bags, y, _ = build_bags(adata.obs, Z)
    print(f"  {len(patients)} patients, {y.sum()} responders, {Z.shape[0]:,} cells x {Z.shape[1]} PCs")

    t = cfg["train"]
    print(f"[train] 2/4 cross-validation: {t['n_splits']} folds x {t['n_repeats']} repeats")
    cv = RepeatedStratifiedKFold(n_splits=t["n_splits"], n_repeats=t["n_repeats"], random_state=t["seed"])
    oof = np.full((t["n_repeats"], len(y)), np.nan)
    t0 = time.time()
    for k, (tr, te) in enumerate(cv.split(patients, y)):
        net = train_model(bags, patients, y, tr, cfg, seed=t["seed"] + k)
        for i in te:
            oof[k // t["n_splits"], i] = predict(net, bags[patients[i]])[0]
        print(f"  fold {k + 1}/{t['n_splits'] * t['n_repeats']} done, {time.time() - t0:.0f}s elapsed")
    df = pd.DataFrame(oof.T, index=patients, columns=[f"repeat_{r}" for r in range(t["n_repeats"])])
    df.insert(0, "y", y)
    df["prob_mean"] = oof.mean(0)
    df.to_csv(os.path.join(res, "oof_predictions.csv"), index_label="patient")

    print("[train] 3/4 final model on all patients")
    net = train_model(bags, patients, y, np.arange(len(y)), cfg, seed=t["seed"])

    print("[train] 4/4 save artifacts")
    torch.save(net.state_dict(), os.path.join(art, "model.pt"))
    np.savez(os.path.join(art, "model.npz"), **{k: v.cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()})
    pickle.dump(pca, open(os.path.join(art, "pca.pkl"), "wb"))
    open(os.path.join(art, "genes.txt"), "w").write("\n".join(adata.var_names))
    m = cfg["model"]
    json.dump({"d_in": m["n_pcs"], "d": m["d"], "d_att": m["d_att"]}, open(os.path.join(art, "model_meta.json"), "w"))
    print(f"[train] wrote {art}/model.pt, pca.pkl, genes.txt and {res}/oof_predictions.csv")


if __name__ == "__main__":
    main()
