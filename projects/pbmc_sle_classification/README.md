# 🧬 PBMC SLE Classification

Classify systemic lupus erythematosus (SLE) patients from healthy donors using peripheral blood
single-cell RNA-seq, at the patient level, and explain which cell types drive the call.

Data: Perez et al. 2022 (*Science*, doi 10.1126/science.abf1970), the lupus PBMC atlas on CELLxGENE.
One label per donor (SLE or healthy), none per cell.

## Layout

```
configs/default.yaml   paths and hyperparameters, read by every script
data.py                download from CELLxGENE, subsample per donor, QC, normalise, HVGs, batch report -> data/processed/pbmc.h5ad
represent.py           scVI latent (batch removed), CellTypist cell types, UMAP; numpy exports          -> data/processed/pbmc_rep.h5ad, artifacts/
train.py               pseudobulk logistic regression vs attention-MIL, donor-split CV, final fit       -> results/metrics.json, artifacts/
insights.py            attention per cell type, pseudobulk DE (PyDESeq2), demo example donors          -> results/, artifacts/examples/
features.py            torch-free helpers shared by training and the app (pseudobulk, scVI encoder, CellTypist)
demo.py                the PBMC SLE tab of the hub app (app/app.py)
train_colab.ipynb      runs all four steps on a free Colab/Kaggle GPU, then optionally uploads the model
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
make data       # large CELLxGENE download, then QC/normalisation/HVGs
make represent  # scVI (GPU recommended) + CellTypist + UMAP
make train      # 5-fold donor-split CV of both models + final fit
make insights   # attention table, DE per top cell type, example donors
make upload     # push artifacts/ to the Hugging Face model repo the app loads from
```

## What the demo needs

`make upload` (or the notebook with `UPLOAD = True`) pushes these to `AdamSion74/pbmc-sle-model`
(override with `PBMC_MODEL_REPO`). They are produced by the pipeline and never committed:

| file | from | used for |
|---|---|---|
| `genes.txt` | represent | the 2,000 HVGs, in model order |
| `scvi_encoder.npz` | represent | embedding a new donor's cells (numpy, no PyTorch) |
| `celltypist.npz` | represent | labelling cells by type (numpy) |
| `model.npz` | train | attention-MIL weights |
| `baseline.joblib` | train | pseudobulk logistic regression |
| `examples/*.h5ad` | insights | example donors shown as one-click chips |

Commit `results/metrics.json` to show the cross-validated AUROC under the demo.

**Demo input:** one donor as `.h5ad` (recommended) with raw counts in `X`, `layers["counts"]` or `.raw`,
gene symbols in `var_names` or `var["feature_name"]`, optional `obs["cell_type"]`. A CSV with one row
per cell and gene columns of raw counts also works.

## Method notes

- **Batch:** `data.py` writes `results/batch_report.csv` (donors per batch x disease). scVI gets the batch as
  `batch_key` in the decoder only (`encode_covariates=False`), so the encoder is batch-free and embeds a donor
  from an unseen batch. DE adds batch as a covariate only when it is not confounded with disease.
- **Evaluation:** both models are scored on identical donor-level folds; AUROC with a 95% bootstrap CI.
- **Demo vs training:** the app types cells with CellTypist's per-cell "best match"; training used majority voting.
  The example donors were part of training, so their probabilities are in-sample.
