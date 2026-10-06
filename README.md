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

Machine learning on single-cell immune data, served as one Gradio app.

- `projects/cart_responder`: CAR-T response prediction (live demo)
- `projects/pbmc_sle_classification`: lupus classification from PBMC (planned)
- `shared/`: code reused across projects
- `app/app.py`: the Gradio app

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r app/requirements.txt
.venv/bin/python app/app.py
```

## Deploy

- Hugging Face: pushes to `main` sync to the Space `AdamSion74/immune-research` via
  `.github/workflows/sync_to_hf.yml` (needs the `HF_TOKEN` repo secret).
- Render (free): `render.yaml` blueprint.

CAR-T weights are not in git; the app downloads them from the model repo `AdamSion74/cart-responder-model`
(override with `CART_MODEL_REPO`).
