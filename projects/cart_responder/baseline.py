"""Baseline: the paper's own 27 hand-made features (cell-type fractions and ratios), same CV folds.

The Amit lab shipped their per-patient feature table with the paper's code
(github.com/AmitLab/CAR_T, data/250126_Features_for_model.csv) and trained XGBoost on it.
Here the same table is scored with two standard tabular models using exactly the same
RepeatedStratifiedKFold splits as src/train.py, so the AUCs are directly comparable.

Input : data/external/amitlab_features.csv, results/oof_predictions.csv (for patient order and labels)
Output: results/baseline_metrics.json, results/baseline_oof.csv
"""
import json
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared.common import load_config, parse_args

from . import DEFAULT_CONFIG, path
from .evaluate import auc_with_ci

URL = "https://raw.githubusercontent.com/AmitLab/CAR_T/main/data/250126_Features_for_model.csv"
NON_FEATURES = ["response_1m_or_3m", "Bcategory"]   # Patient_ID is the index


def main() -> None:
    args = parse_args(__doc__, DEFAULT_CONFIG)
    cfg = load_config(args.config)
    res = path(cfg["paths"]["results"])
    feat_path = path("data/external/amitlab_features.csv")
    if not os.path.exists(feat_path):
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        urllib.request.urlretrieve(URL, feat_path)

    oof_ours = pd.read_csv(os.path.join(res, "oof_predictions.csv"), index_col="patient")
    patients, y = oof_ours.index.values, oof_ours["y"].values
    feats = pd.read_csv(feat_path, index_col=0).set_index("Patient_ID").loc[patients]
    assert ((feats["response_1m_or_3m"] == "R").astype(int).values == y).all(), "label mismatch with the paper's table"
    X = feats.drop(columns=NON_FEATURES).astype(float).values
    print(f"[baseline] {X.shape[0]} patients x {X.shape[1]} paper features")

    t = cfg["train"]
    models = {
        "logistic_regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                    LogisticRegression(C=0.5, max_iter=2000)),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                  random_state=t["seed"]),
    }
    cv = RepeatedStratifiedKFold(n_splits=t["n_splits"], n_repeats=t["n_repeats"], random_state=t["seed"])
    out, table = {}, pd.DataFrame({"y": y}, index=patients)
    for name, make in models.items():
        oof = np.full((t["n_repeats"], len(y)), np.nan)
        for k, (tr, te) in enumerate(cv.split(patients, y)):
            clf = make().fit(X[tr], y[tr])
            oof[k // t["n_splits"], te] = clf.predict_proba(X[te])[:, 1]
        prob = oof.mean(0)
        auc, lo, hi = auc_with_ci(y, prob, cfg["eval"]["n_boot"])
        out[name] = {"auc": round(auc, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        table[name] = prob
        print(f"  {name:22s} AUC {auc:.3f}   95% CI {lo:.3f} to {hi:.3f}")

    mil = json.load(open(os.path.join(res, "metrics.json")))
    out["attention_mil_cells"] = {"auc": mil["auc"], "ci95": mil["ci95"]}
    json.dump(out, open(os.path.join(res, "baseline_metrics.json"), "w"), indent=2)
    table.to_csv(os.path.join(res, "baseline_oof.csv"), index_label="patient")
    print("[baseline] comparison (same 57 patients, same 25 folds):")
    for k, v in out.items():
        print(f"  {k:22s} {v['auc']:.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")


if __name__ == "__main__":
    main()
