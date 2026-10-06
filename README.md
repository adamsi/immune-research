---
title: Immune Research
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app/app.py
pinned: false
license: mit
short_description: Single-cell immunology ML projects in one demo hub
---

# Immune Research

Machine learning on single-cell immune data. One repo, one Gradio app, one Hugging Face Space.

| project | status | folder |
|---|---|---|
| 🧬 CAR-T Responder: predict CD19 CAR-T response from pre-treatment blood scRNA-seq | live demo | [`projects/cart_responder`](projects/cart_responder) |

More projects are in progress.

## Layout

```
shared/                  code reused across projects
  common.py              config, seeding, patient bags
  data.py                downloads, dense float32 matrices
  mil_torch.py           gated attention-MIL (PyTorch, training)
  mil_numpy.py           the same forward pass in numpy (app inference)
  plotting.py            attention share per cell type
projects/<name>/         one folder per project: pipeline, config, README
app/app.py               the Gradio hub: Home + one tab per project
app/requirements.txt     what the Space installs (root requirements.txt points here)
.github/workflows/       sync_to_hf.yml mirrors main to the Space
```

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r app/requirements.txt
cp .env.example .env    # optional: HF_TOKEN for a private model repo, CART_MODEL_REPO to override
.venv/bin/python app/app.py   # http://localhost:7860
```

## Model weights

The CAR-T inference artifacts are small (`model.npz`, `pca.pkl`, `genes.txt`, about 0.5 MB) and are
committed in `projects/cart_responder/artifacts/`; everything else there (`model.pt`, checkpoints) stays
git-ignored. If those files are missing, the app downloads them with `hf_hub_download` from the model
repo in `CART_MODEL_REPO` (default `AdamSion74/cart-responder-model`). To publish weights there instead:

```bash
hf upload AdamSion74/cart-responder-model projects/cart_responder/artifacts . --repo-type model
```

(`huggingface-cli upload ...` with the same arguments on older `huggingface_hub` versions.)
If the repo is private, add `HF_TOKEN` as a Space secret.

## Deploy

The Space runs on the CPU basic hardware (Gradio Spaces on CPU need a Hugging Face PRO account).

Every push to `main` runs `.github/workflows/sync_to_hf.yml`, which uploads the repo to the Space
`AdamSion74/immune-research` with `hf upload` (override with an `HF_SPACE` repository variable). It needs a
Hugging Face write token stored as the `HF_TOKEN` repository secret; without it the job is skipped.
