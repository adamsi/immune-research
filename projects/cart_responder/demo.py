"""CAR-T Responder demo tab: upload a patient's single-cell counts, get P(response) and attention by cell type.

Artifacts (model.npz, pca.pkl, genes.txt) are downloaded from the Hugging Face model repo named by
CART_MODEL_REPO. A local projects/cart_responder/artifacts/ folder (training output) takes precedence.
"""
import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd

from shared.mil_numpy import load_weights, predict as mil_predict
from shared.plotting import attention_by_group
from shared.ui import connect, file_input, github_link

from . import path

MODEL_REPO = os.environ.get("CART_MODEL_REPO", "AdamSion74/cart-responder-model")
ARTIFACT_FILES = ("model.npz", "pca.pkl", "genes.txt")
EX = path("examples")
EXAMPLES = sorted(os.path.join(EX, f) for f in os.listdir(EX) if f.endswith(".csv")) if os.path.isdir(EX) else []
TOP_TYPES = 10


def _artifact_paths() -> dict:
    local = path("artifacts")
    if all(os.path.exists(os.path.join(local, f)) for f in ARTIFACT_FILES):
        return {f: os.path.join(local, f) for f in ARTIFACT_FILES}
    from huggingface_hub import hf_hub_download
    return {f: hf_hub_download(MODEL_REPO, f) for f in ARTIFACT_FILES}


def _load():
    try:
        p = _artifact_paths()
        with open(p["pca.pkl"], "rb") as f:
            pca = pickle.load(f)
        with open(p["genes.txt"]) as f:
            genes = f.read().split()
        return pca, genes, load_weights(p["model.npz"]), None
    except Exception as e:  # keep the rest of the app usable if the model repo is missing
        return None, [], None, f"{type(e).__name__}: {e}"


PCA, GENES, WEIGHTS, LOAD_ERROR = _load()


def predict(file):
    if LOAD_ERROR:
        raise gr.Error(f"Model weights are not available ({LOAD_ERROR}).")
    if file is None:
        raise gr.Error("Upload a CSV or choose an example patient first.")
    path_ = file if isinstance(file, str) else file.name
    header = pd.read_csv(path_, nrows=0, index_col=0).columns
    dtypes = {c: (str if c == "cell_type" else np.float32) for c in header}     # float32 halves memory vs the default
    df = pd.read_csv(path_, index_col=0, dtype=dtypes)

    cell_type = df.pop("cell_type") if "cell_type" in df else None
    totals = df.pop("total_counts").values if "total_counts" in df else None

    present = [g for g in GENES if g in df.columns]
    missing = len(GENES) - len(present)
    if len(present) < 0.5 * len(GENES):
        raise gr.Error(f"Only {len(present)} of {len(GENES)} model genes found in the file. "
                       "Columns must be gene names from the model's genes.txt.")
    counts = df.reindex(columns=GENES, fill_value=0).to_numpy(dtype=np.float32)
    cell_ids, n_cells = df.index, len(df)
    del df
    if totals is None:
        totals = counts.sum(1)
    totals = np.clip(np.asarray(totals, dtype=np.float32), 1, None)

    x = np.log1p(counts / totals[:, None] * 1e4)
    p, att = mil_predict(WEIGHTS, PCA.transform(x).astype(np.float32))

    # ---- probability card
    verdict = "Likely responder ✅" if p >= 0.5 else "Likely non-responder ⚠️"
    color = "#15803d" if p >= 0.5 else "#b91c1c"
    card = f"""
    <div class="card">
      <div class="label">Predicted probability of response</div>
      <div class="big" style="color:{color}">{p:.0%}</div>
      <div class="bar"><div class="fill" style="width:{p*100:.1f}%;background:{color}"></div></div>
      <div class="verdict" style="color:{color}">{verdict}</div>
      <div class="stats">
        <div><span>{n_cells:,}</span>cells analysed</div>
        <div><span>{len(present):,}</span>of {len(GENES):,} genes matched</div>
        <div><span>{missing}</span>genes filled with 0</div>
      </div>
    </div>"""

    # ---- attention by cell type
    if cell_type is not None:
        share = attention_by_group(att, cell_type.values, top=TOP_TYPES)
        plot_df = pd.DataFrame({"cell type": share.index, "attention share (%)": (share.values * 100).round(2)})
    else:
        plot_df = pd.DataFrame({"cell type": ["(no cell_type column in file)"], "attention share (%)": [100.0]})

    # ---- top attended cells
    top = np.argsort(att)[::-1][:10]
    top_df = pd.DataFrame({
        "cell": cell_ids[top],
        "cell type": cell_type.values[top] if cell_type is not None else "?",
        "attention weight": [round(float(v), 5) for v in att[top]],
    })
    return card, gr.BarPlot(value=plot_df, visible=True), gr.Dataframe(value=top_df, visible=True)


IDLE = "<div class='card'><div class='idle'>Upload a patient's file or pick an example.</div></div>"

HOW = """
<p class="note" style="margin-top:0">
  Single-cell RNA-seq is the input. It is the lab measurement that reads gene activity in every individual
  blood cell, so a patient becomes a table of thousands of cells by thousands of genes.
</p>
<div class="steps">
  <div class="step"><div class="n">STEP 1</div><b>🩸 Blood sample</b><span>Thousands of immune cells, each with about 20,000 gene activity values, measured before treatment.</span></div>
  <div class="step"><div class="n">STEP 2</div><b>📉 Compress</b><span>Keep 2,000 informative genes, normalise, and reduce every cell to 50 numbers with PCA.</span></div>
  <div class="step"><div class="n">STEP 3</div><b>👀 Attend</b><span>A small network scores every cell. Softmax turns the scores into weights that sum to one.</span></div>
  <div class="step"><div class="n">STEP 4</div><b>🎯 Predict</b><span>The weighted average of the cells becomes one patient vector, then a probability of response.</span></div>
</div>
<p class="note">
  Trained on 55 lymphoma patients from the public dataset
  <a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE267097" target="_blank">GSE267097</a>
  (Weizmann Institute, Amit lab, <i>Cancer Research</i> 2025). There is one label per patient and none per cell,
  so the model learns on its own which cells matter. This is attention-based multiple-instance learning.
  Performance is measured with patient-level cross-validation and reported as AUC with a bootstrap confidence interval.
</p>
<p class="note muted">⚠️ For research use only. Not a medical device and not intended for clinical decisions.</p>"""

FORMAT = """
<ul class="fmt">
  <li>One row per cell; first column is the cell id.</li>
  <li>Other columns: gene names with raw counts (the model's <code>genes.txt</code>).</li>
  <li>Optional <code>cell_type</code> (enables the attention chart) and <code>total_counts</code>.</li>
</ul>"""


def _example_label(path_: str) -> str:
    name = os.path.basename(path_).removesuffix(".csv")
    patient, _, label = name.partition("_")
    return f"{patient} · {label.replace('_', '-')}"


def _diagram() -> str:
    svg = path("diagram/architecture.svg")
    if not os.path.exists(svg):
        return ""
    with open(svg) as f:
        return "<div class='diagram'>" + f.read() + "</div>"


def build() -> None:
    """Render the CAR-T tab contents inside the current gr.Blocks context."""
    gr.HTML(f"""<div class="page-head"><h2>🧬 CAR-T Responder</h2>
      <p>Predict CD19 CAR-T response from a pre-treatment blood sample, and see which cells drove it.</p>
      {github_link("cart_responder")}</div>""")
    if LOAD_ERROR:
        gr.HTML(f"<div class='banner'>Model weights could not be loaded from <code>{MODEL_REPO}</code>. "
                "Predictions are disabled until they are uploaded (see the README).</div>")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=280):
            gr.HTML("<div class='section'>1 · Patient</div>")
            picker = file_input("Upload a patient's cells", ".csv · raw counts", [".csv"],
                                [(_example_label(e), e) for e in EXAMPLES])
            with gr.Accordion("File format", open=False):
                gr.HTML(FORMAT)
        with gr.Column(scale=7, min_width=280):
            gr.HTML("<div class='section'>2 · Result</div>")
            card = gr.HTML(IDLE)
            plot = gr.BarPlot(x="cell type", y="attention share (%)", label="Attention by cell type", sort="-y",
                              x_label_angle=-35, height=360, visible=False)
            top = gr.Dataframe(label="Top 10 cells", interactive=False, wrap=True, visible=False)
    with gr.Accordion("🧠 How the model works", open=True):
        gr.HTML(_diagram() + HOW)

    connect(picker, predict, [card, plot, top])
