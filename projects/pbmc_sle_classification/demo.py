"""PBMC SLE demo tab: upload one donor's PBMC single-cell counts, get P(SLE) and attention by cell type.

Runs without PyTorch: the scVI encoder, CellTypist and the attention-MIL are numpy exports (features.py,
shared/mil_numpy.py). Artifacts come from the Hugging Face model repo named by PBMC_MODEL_REPO; a local
projects/pbmc_sle_classification/artifacts/ folder (training output) takes precedence.
"""
import json
import os

import gradio as gr
import numpy as np
import pandas as pd

from shared.mil_numpy import load_weights, predict as mil_predict
from shared.plotting import attention_by_group
from shared.ui import connect, file_input

from . import path
from .features import encode_scvi, log_normalize, predict_celltypes, to_dense

MODEL_REPO = os.environ.get("PBMC_MODEL_REPO", "AdamSion74/pbmc-sle-model")
ARTIFACT_FILES = ("model.npz", "baseline.joblib", "scvi_encoder.npz", "celltypist.npz", "genes.txt")
TOP_TYPES = 10


def _artifact_dir() -> str:
    local = path("artifacts")
    if all(os.path.exists(os.path.join(local, f)) for f in ARTIFACT_FILES):
        return local
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_REPO, allow_patterns=[*ARTIFACT_FILES, "examples/*.h5ad"])


def _load():
    try:
        import joblib
        d = _artifact_dir()
        art = {
            "dir": d,
            "mil": load_weights(os.path.join(d, "model.npz")),
            "baseline": joblib.load(os.path.join(d, "baseline.joblib")),
            "scvi": dict(np.load(os.path.join(d, "scvi_encoder.npz"))),
            "celltypist": dict(np.load(os.path.join(d, "celltypist.npz"))),
            "genes": open(os.path.join(d, "genes.txt")).read().split(),
        }
        return art, None
    except Exception as e:  # keep the rest of the app usable until the model is trained and uploaded
        return None, f"{type(e).__name__}: {e}"


ART, LOAD_ERROR = _load()
EX = os.path.join(ART["dir"], "examples") if ART else ""
EXAMPLES = sorted(os.path.join(EX, f) for f in os.listdir(EX) if f.endswith(".h5ad")) if EX and os.path.isdir(EX) else []


def _read(file_path: str):
    """Return (counts cells x genes, gene names, per-cell totals, optional cell types, cell ids)."""
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, index_col=0)
        cell_type = df.pop("cell_type").astype(str).values if "cell_type" in df else None
        totals = df.pop("total_counts").values if "total_counts" in df else None
        return df.to_numpy(np.float32), list(df.columns), totals, cell_type, df.index
    import anndata as ad
    a = ad.read_h5ad(file_path)
    if "counts" in a.layers:
        X = a.layers["counts"]
    elif a.raw is not None:
        X, a = a.raw.X, a.raw.to_adata()
    else:
        X = a.X
    genes = a.var["feature_name"].astype(str).tolist() if "feature_name" in a.var else list(a.var_names)
    totals = a.obs["total_counts"].values if "total_counts" in a.obs else None
    ct_col = next((c for c in ("cell_type_pred", "cell_type") if c in a.obs), None)
    return X, genes, totals, (a.obs[ct_col].astype(str).values if ct_col else None), a.obs_names


def predict(file_path):
    if LOAD_ERROR:
        raise gr.Error("The PBMC model is not available yet (see the banner above).")
    if not file_path:
        raise gr.Error("Upload a file or pick an example first.")
    X, genes, totals, cell_type, cell_ids = _read(file_path)
    X = to_dense(X)
    if X.size and not np.allclose(X[:200] % 1, 0):
        raise gr.Error("Expected raw integer counts, but the values look normalised.")
    col = {g: i for i, g in enumerate(genes)}
    model_genes = ART["genes"]
    present = [g for g in model_genes if g in col]
    if len(present) < 0.5 * len(model_genes):
        raise gr.Error(f"Only {len(present)} of {len(model_genes)} model genes found. "
                       "Genes must be symbols (e.g. CD3E), in var_names or var['feature_name'].")
    counts = np.zeros((len(X), len(model_genes)), dtype=np.float32)
    idx = [i for i, g in enumerate(model_genes) if g in col]
    counts[:, idx] = X[:, [col[model_genes[i]] for i in idx]]

    if cell_type is None:
        cell_type = predict_celltypes(ART["celltypist"], log_normalize(X, totals), genes)
    z = encode_scvi(ART["scvi"], counts)
    p, att = mil_predict(ART["mil"], z)
    p_base = float(ART["baseline"].predict_proba(counts.sum(0, keepdims=True))[0, 1])

    sle = p >= 0.5
    color = "#b91c1c" if sle else "#15803d"
    verdict = "Lupus-like profile" if sle else "Healthy-like profile"
    card = f"""
    <div class="card">
      <div class="label">Probability of SLE</div>
      <div class="big" style="color:{color}">{p:.0%}</div>
      <div class="bar"><div class="fill" style="width:{p * 100:.1f}%;background:{color}"></div></div>
      <div class="verdict" style="color:{color}">{verdict}</div>
      <div class="stats">
        <div><span>{len(X):,}</span>cells</div>
        <div><span>{len(present):,}/{len(model_genes):,}</span>model genes</div>
        <div><span>{p_base:.0%}</span>baseline model</div>
      </div>
    </div>"""
    share = attention_by_group(att, cell_type, top=TOP_TYPES)
    plot_df = pd.DataFrame({"cell type": share.index, "attention share (%)": (share.values * 100).round(2)})
    return card, gr.BarPlot(value=plot_df, visible=True)


IDLE = "<div class='card'><div class='idle'>Upload a donor's file or pick an example.</div></div>"

FLOW = [
    ("🩸", "Data", "Perez et al. 2022 lupus PBMC atlas (CELLxGENE)", "≈260 donors · SLE vs healthy"),
    ("🧹", "Preprocess", "QC, normalisation, 2,000 HVGs, batch report", "clean counts per cell"),
    ("🧭", "Represent", "scVI latent space (batch removed), CellTypist, UMAP", "30 numbers + a type per cell"),
    ("🎯", "Model", "Attention-MIL on each donor's cells vs pseudobulk logistic regression", "P(SLE) per donor · AUROC"),
    ("🔍", "Insights", "Attention per cell type, pseudobulk DE (PyDESeq2)", "which cells and genes"),
]


def _metrics_html() -> str:
    f = path("results/metrics.json")
    if not os.path.exists(f):
        return ""
    m = json.load(open(f))
    tile = lambda name, k: (f"<div class='tile'><span>{m[k]['auroc']:.2f}</span>{name}"
                            f"<small>95% CI {m[k]['ci95'][0]:.2f}–{m[k]['ci95'][1]:.2f}</small></div>")
    return (f"<div class='tiles'>{tile('Attention-MIL AUROC', 'mil')}{tile('Baseline AUROC', 'baseline')}"
            f"<div class='tile'><span>{m['n_donors']}</span>donors<small>{m['n_sle']} SLE · {m['cv']}</small></div></div>")


def _architecture_html() -> str:
    nodes = "<div class='arrow'>→</div>".join(
        f"<div class='node'><div class='n'>STEP {i}</div><div class='ico'>{ic}</div><b>{t}</b>"
        f"<span>{d}</span><em>{o}</em></div>" for i, (ic, t, d, o) in enumerate(FLOW, 1))
    return f"""
    <div class="flow">{nodes}</div>
    {_metrics_html()}
    <p class="note">Each donor is a bag of thousands of blood cells with a single label (SLE or healthy).
      scVI compresses every cell to 30 numbers while removing lab-batch effects; the attention-MIL model scores
      each cell, averages the cells by those scores and outputs P(SLE). The attention weights show which cell
      types drove the call. A pseudobulk logistic regression on summed gene counts is the baseline, scored on the
      same donor-split folds.</p>
    <p class="note muted">⚠️ For research use only. Not a medical device and not intended for clinical decisions.</p>"""


FORMAT = """
<ul class="fmt">
  <li><b>.h5ad</b> (recommended): one donor, cells × genes, <b>raw counts</b> in <code>X</code>,
      <code>layers["counts"]</code> or <code>.raw</code>.</li>
  <li>Gene symbols (e.g. <code>CD3E</code>) as <code>var_names</code> or <code>var["feature_name"]</code>.</li>
  <li>Optional <code>obs["cell_type"]</code>; otherwise cells are typed with CellTypist.</li>
  <li><b>.csv</b> also works: one row per cell, first column the cell id, gene columns with raw counts.</li>
</ul>"""


def _example_label(p: str) -> str:
    donor, _, label = os.path.basename(p).removesuffix(".h5ad").rpartition("_")
    return f"{donor} · {label}"


def build() -> None:
    """Render the PBMC SLE tab inside the current gr.Blocks context."""
    status = "Live demo" if not LOAD_ERROR else "Training"
    pill = "live" if not LOAD_ERROR else "soon"
    gr.HTML(f"""<div class="page-head"><h2>🧬 PBMC SLE Classification</h2>
      <p>Classify a donor as lupus (SLE) or healthy from blood single-cell RNA-seq, and see which cells drove it.
      <span class="pill {pill}">{status}</span></p></div>""")
    if LOAD_ERROR:
        gr.HTML(f"<div class='banner'>The model has not been uploaded to <code>{MODEL_REPO}</code> yet, so "
                "predictions are off. Train it with the steps in the project README, then <code>make upload</code>.</div>")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=280):
            gr.HTML("<div class='section'>1 · Donor</div>")
            picker = file_input("Upload a donor's cells", ".h5ad or .csv · raw counts", [".h5ad", ".csv"],
                                [(_example_label(e), e) for e in EXAMPLES])
            with gr.Accordion("File format", open=False):
                gr.HTML(FORMAT)
        with gr.Column(scale=7, min_width=280):
            gr.HTML("<div class='section'>2 · Result</div>")
            card = gr.HTML(IDLE)
            plot = gr.BarPlot(x="cell type", y="attention share (%)", label="Attention by cell type", sort="-y",
                              x_label_angle=-35, height=340, visible=False)
    gr.HTML("<div class='section arch-title'>How it works</div>" + _architecture_html())
    connect(picker, predict, [card, plot])
