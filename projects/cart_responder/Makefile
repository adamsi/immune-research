# Run the CAR-T pipeline step by step from this folder. `make` alone runs everything that is out of date.
#   make setup      create .venv at the repo root and install requirements
#   make data       download GEO data and build data/processed/dataset.h5ad
#   make train      cross-validate, then fit the final model -> artifacts/
#   make evaluate   AUC with CI, attention by cell type, example patients -> results/, examples/
#   make baseline   the paper's 27-feature models on identical folds -> results/baseline_metrics.json
#   make app        launch the hub app (all projects)
#   make upload     push artifacts/ to the Hugging Face model repo the app loads from
ROOT = ../..
PY   = cd $(ROOT) && .venv/bin/python
MOD  = projects.cart_responder
CFG  = configs/default.yaml
MODEL_REPO ?= AdamSion74/cart-responder-model

.PHONY: all setup data train evaluate baseline app upload clean

all: evaluate baseline

setup:
	cd $(ROOT) && python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r projects/cart_responder/requirements.txt

data/processed/dataset.h5ad: data.py $(ROOT)/shared/data.py $(CFG)
	$(PY) -m $(MOD).data

artifacts/model.pt: train.py $(ROOT)/shared/mil_torch.py $(ROOT)/shared/common.py data/processed/dataset.h5ad
	$(PY) -m $(MOD).train

results/metrics.json: evaluate.py artifacts/model.pt
	$(PY) -m $(MOD).evaluate

results/baseline_metrics.json: baseline.py results/metrics.json
	$(PY) -m $(MOD).baseline

data:     data/processed/dataset.h5ad
train:    artifacts/model.pt
evaluate: results/metrics.json
baseline: results/baseline_metrics.json

app:
	$(PY) app/app.py

upload:
	$(ROOT)/.venv/bin/hf upload $(MODEL_REPO) artifacts . --repo-type model

clean:
	rm -rf artifacts results data/processed
