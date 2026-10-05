# CAR-T Responder Classifier

Attention-based multiple-instance learning (MIL) that predicts whether a lymphoma patient will
respond to CD19 CAR-T therapy from pre-treatment blood single-cell RNA-seq, plus a Gradio demo.

Data: [GEO GSE267097](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267097), 55 patients,
Weizmann Institute (Amit lab), *Cancer Research* 2025. One label per patient, none per cell.

## Layout

```
configs/default.yaml     paths and hyperparameters, read by every script
data.py                  download GEO, load 16 libraries, normalise, 2,000 HVGs  -> data/processed/dataset.h5ad
train.py                 PCA, repeated patient-level CV, final fit               -> artifacts/, results/oof_predictions.csv
evaluate.py              AUC with bootstrap CI, attention by cell type, examples -> results/, examples/
baseline.py              the paper's 27 features on identical folds              -> results/baseline_metrics.json
demo.py                  the CAR-T tab of the hub app (app/app.py)
Makefile                 runs the steps in order, skips what is up to date
diagram/architecture.svg architecture figure shown in the app
```

The model itself lives in `shared/` (`mil_torch.py` for training, `mil_numpy.py` for inference).
`data/` and `artifacts/` are git-ignored: they are produced, not authored. Run scripts from the
repo root as modules (`python -m projects.cart_responder.train`) or through the Makefile here.

## Run

```bash
cd projects/cart_responder
make setup      # once: create .venv at the repo root, install requirements.txt
make data       # ~2.4 GB download from GEO, then ~5 min of processing
make train      # 25 CV folds + final model, ~30 min on a laptop CPU
make evaluate   # metrics, attention table, example patients
make baseline   # the paper's feature-based models on identical folds
make app        # hub app at http://localhost:7860
make upload     # push artifacts/ to the Hugging Face model repo the app loads from
```

`make` alone runs data, train, evaluate and baseline, only re-running steps whose inputs changed.
To redo a step, delete its output (for example `rm artifacts/model.pt`) and run `make` again.

## Model

Each patient is a bag of cells. Cells are log-normalised, reduced to 50 PCs, encoded by a small
MLP, scored by gated attention, pooled by weighted sum into one patient vector, and classified.
Trained with cross-entropy on the patient label; evaluated with 5-fold stratified CV repeated
5 times, reported as AUC with a 95% bootstrap CI. Attention weights summed per cell type give
the interpretability table in `results/attention_by_celltype.csv`.

## Results (57 patients, 5-fold CV x 5 repeats, identical folds for every row)

| model | input | AUC | 95% CI |
|---|---|---|---|
| logistic regression | paper's 25 hand-made cell-type features | 0.773 | 0.61 to 0.90 |
| attention MIL (this repo) | ~91k raw cells, 50 PCs each | 0.708 | 0.53 to 0.87 |
| gradient boosting | paper's 25 hand-made cell-type features | 0.665 | 0.48 to 0.82 |

The intervals overlap almost completely: with 57 patients none of the three is distinguishable
from the others. The cell-level model reaches the same range as hand-engineered features without
any biological feature design, which is the point of the comparison.

Attention by cell type (`results/attention_by_celltype.csv`): the model puts 48% of its attention on
B lymphocytes in responders versus under 1% in non-responders, then CD14 monocytes and NK cells.
Caveat: in this cohort circulating B cells are found mostly in responders, so the model may be
leaning on B-cell presence, a coarse feature the paper also encodes. The example patients in the
demo were part of training, so their probabilities are in-sample and look more confident than
a held-out prediction would.

## Demo input format

CSV, one row per cell, first column cell id, remaining columns gene names with raw counts matching
the model's `genes.txt`. Optional `cell_type` (enables the attention chart) and `total_counts`.
`make evaluate` writes three real patients into `examples/` in this format.
