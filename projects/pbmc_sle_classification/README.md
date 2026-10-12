# 🧬 PBMC SLE Classification

Classify a blood sample as **healthy or one of several SLE subtypes** from peripheral blood single-cell RNA-seq,
and explain which cell types and genes drive the call.

Data: Perez et al. 2022 (*Science*, doi 10.1126/science.abf1970), the lupus PBMC atlas on CELLxGENE:
274 blood samples from 261 donors (162 SLE, 99 healthy), 4 processing cohorts. 10 patients were sampled at flare
and again after treatment.

**Why subtypes:** SLE vs healthy is close to solved (AUROC ~0.97), and SLE patients differ a lot from each other
(some have a strong interferon response, others mostly lose naive T cells or expand cytotoxic cells), which
matters for treatment. Subtypes are found by clustering the SLE patients, then the classifier learns to assign a
new sample to healthy or a subtype. Flare vs stable disease would be the obvious clinical target, but every flare
sample is in one processing cohort, so a model could score well by recognising that cohort.

## Layout

```
configs/default.yaml   paths and hyperparameters, read by every script
data.py                download from CELLxGENE, subsample per sample, QC + platelet filter, Perez labels, HVGs -> data/processed/pbmc.h5ad
represent.py           scVI latent (batch removed), CellTypist trained on Perez labels, UMAP; numpy exports   -> data/processed/pbmc_rep.h5ad, artifacts/
endotypes.py           SLE subtypes from abundance + gene programs, centred on same-cohort healthy samples  -> results/samples.csv, artifacts/classes.json
train.py               elastic-net baseline vs multi-head attention-MIL, donor-split + cohort-held-out CV   -> results/metrics.json, artifacts/
insights.py            attention per class, abundance test, DE, flare -> after treatment, demo examples    -> results/, artifacts/examples/
features.py            torch-free helpers shared by training and the app (samples, abundance, baseline features, scVI encoder, CellTypist)
demo.py                the PBMC SLE tab of the hub app (app/app.py)
train_colab.ipynb      runs all five steps on a free Colab/Kaggle GPU, then optionally uploads the model
explore_colab.ipynb    the same steps as functions, cell by cell, with plots and knobs, cached on Google Drive
Makefile               runs the steps in order, skips what is up to date
```

The attention-MIL model is shared with the CAR-T project (`shared/mil_torch.py` for training,
`shared/mil_numpy.py` for inference). Run scripts from the repo root as modules
(`python -m projects.pbmc_sle_classification.train`) or through the Makefile here.

## Run

Easiest: open `train_colab.ipynb` in Colab (T4 GPU) and Run all. Locally:

```bash
cd projects/pbmc_sle_classification
make setup      # once: .venv at the repo root + requirements.txt
make data       # large CELLxGENE download, then QC, Perez labels, normalisation, HVGs
make represent  # scVI (GPU recommended) + CellTypist on Perez labels + UMAP
make endotypes  # SLE subtypes -> the class labels
make train      # 5-fold donor-split CV + leave-one-cohort-out of both models, final fit
make insights   # attention, abundance test, DE, flare vs treated, example samples
make upload     # push artifacts/ to the Hugging Face model repo the app loads from
```

## What the demo needs

`make upload` (or the notebook with `UPLOAD = True`) pushes these to `AdamSion74/pbmc-sle-model`
(override with `PBMC_MODEL_REPO`). They are produced by the pipeline and never committed:

| file | from | used for |
|---|---|---|
| `genes.txt` | represent | the 5,000 HVGs, in model order |
| `scvi_encoder.npz` | represent | embedding a new sample's cells (numpy, no PyTorch) |
| `celltypist.npz` | represent | labelling cells with Perez's cell types (numpy) |
| `classes.json` | endotypes | class names (healthy, SLE-A, ...) and what defines each subtype |
| `model.npz` | train | attention-MIL weights |
| `baseline.joblib` | train | elastic-net baseline |
| `baseline_features.json` | train | cell types, lineages and genes the baseline and abundance features use |
| `examples/*.h5ad` | insights | example samples shown as one-click chips |

Commit `results/metrics.json` and `results/summary.json` to show the scores and findings under the demo.
Without `classes.json` the app runs the earlier binary SLE-vs-healthy model, so the demo keeps working until a
new model is uploaded.

**Demo input:** one donor as `.h5ad` (recommended) with raw counts in `X`, `layers["counts"]` or `.raw`,
gene symbols in `var_names` or `var["feature_name"]`, optional `obs["cell_type"]`. A CSV with one row
per cell and gene columns of raw counts also works.

## Method notes

Design choices, and where they follow Perez et al. 2022:

- **One bag per blood sample, folds split by donor.** A patient's flare and after-treatment samples are separate
  bags but always in the same fold (`StratifiedGroupKFold`), so the model never sees a test patient.
- **QC:** gene and mito filters, plus Perez's platelet filter (> 1 count of PF4, PPBP, GNG11, SDPR/CAVIN2).
  Doublets were already removed by the authors (Freemuxlet + Scrublet).
- **Cell labels:** Perez's manual annotation (markers + CITE-seq): 14 lymphoid subtypes (`ct_cov`) and broad
  myeloid types (`author_cell_type`). A CellTypist model trained on these labels types a new donor's cells in the app.
- **HVGs:** 5,000 (seurat_v3, per batch). The paper used Scanpy 1.6's cutoff-based default and does not give a count.
- **Batch:** Perez used ComBat on pool and regressed out nUMI and % mito. Here scVI takes the cohort and % mito
  in the decoder only (`encode_covariates=False`), so the encoder is covariate-free and embeds a donor from an
  unseen batch. `results/batch_report.csv` shows samples per cohort and disease state; cohort 1 is healthy only.
- **Subtypes:** each SLE sample is described by CLR cell-type abundance and interferon / cytotoxic program scores
  per lineage, each feature centred on the healthy samples of the same cohort. k-means picks k by silhouette among
  stable solutions (bootstrap ARI >= 0.6). Subtypes are found once on all samples without using the classifier's
  folds; they are a description of the cohort, and the classifier's job is to assign new samples to them.
- **Models:** (1) elastic-net logistic regression (saga, l1_ratio 0.5, as in Perez) on per-lineage pseudobulk
  log-CPM plus CLR abundance; (2) attention-MIL with 4 gated attention heads over scVI cell embeddings, the CLR
  abundance vector concatenated before the head, AdamW + cosine schedule, class-weighted label-smoothed
  cross-entropy; (3) their average. The app shows the average.
- **Evaluation:** identical donor-split folds for all models; SLE-vs-healthy AUROC and macro one-vs-rest AUROC with
  95% bootstrap CIs, balanced accuracy, confusion matrix, AUROC per ancestry, and leave-one-cohort-out (train on
  the other cohorts, test on one), which shows how much of the score survives a new lab batch. Perez reported
  AUC 0.84 training on cohort 4 and testing on 1 to 3.
- **Insights:** attention per cell type and class; Perez's abundance test (weighted least squares, weights = cell
  counts, Bonferroni); PyDESeq2 SLE vs healthy per cell type (one sample per donor, batch as covariate where the
  design allows); flare vs after treatment in the same patients (paired PyDESeq2, `~ donor + state`, and Wilcoxon
  on abundance), which has no batch confounding.
- **Demo vs training:** the app types cells with the CellTypist model unless the file has Perez-style labels.
  The example samples were part of training, so their probabilities are in-sample.
