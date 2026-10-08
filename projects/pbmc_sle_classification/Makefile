# Run the PBMC SLE pipeline step by step from this folder. `make` alone runs everything that is out of date.
#   make setup      create .venv at the repo root and install requirements
#   make data       download Perez 2022 from CELLxGENE, QC, normalise, HVGs -> data/processed/pbmc.h5ad
#   make represent  scVI latent, CellTypist cell types, UMAP              -> data/processed/pbmc_rep.h5ad, artifacts/
#   make train      donor-split CV of baseline and attention-MIL, final fit -> results/metrics.json, artifacts/
#   make insights   attention per cell type, pseudobulk DE, demo examples  -> results/, artifacts/examples/
#   make app        launch the hub app (all projects)
#   make upload     push artifacts/ to the Hugging Face model repo the app loads from
ROOT = ../..
PY   = cd $(ROOT) && .venv/bin/python
MOD  = projects.pbmc_sle_classification
CFG  = configs/default.yaml
MODEL_REPO ?= AdamSion74/pbmc-sle-model

.PHONY: all setup data represent train insights app upload clean

all: insights

setup:
	cd $(ROOT) && python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r projects/pbmc_sle_classification/requirements.txt

data/processed/pbmc.h5ad: data.py $(CFG)
	$(PY) -m $(MOD).data

data/processed/pbmc_rep.h5ad: represent.py features.py data/processed/pbmc.h5ad
	$(PY) -m $(MOD).represent

results/metrics.json: train.py features.py $(ROOT)/shared/mil_torch.py data/processed/pbmc_rep.h5ad
	$(PY) -m $(MOD).train

results/attention_by_celltype.csv: insights.py results/metrics.json
	$(PY) -m $(MOD).insights

data:      data/processed/pbmc.h5ad
represent: data/processed/pbmc_rep.h5ad
train:     results/metrics.json
insights:  results/attention_by_celltype.csv

app:
	$(PY) app/app.py

upload:
	$(ROOT)/.venv/bin/hf upload $(MODEL_REPO) artifacts . --repo-type model --exclude "scvi/*"

clean:
	rm -rf artifacts results data/processed
