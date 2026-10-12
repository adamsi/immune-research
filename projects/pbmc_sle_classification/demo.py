"""PBMC SLE demo tab: upload one blood sample's single-cell counts, get healthy vs SLE subtype, P(SLE) and
attention by cell type.

Runs without PyTorch: the scVI encoder, CellTypist and the attention-MIL are numpy exports (features.py,
shared/mil_numpy.py). Artifacts come from the Hugging Face model repo named by PBMC_MODEL_REPO; a local
projects/pbmc_sle_classification/artifacts/ folder (training output) takes precedence.
Artifacts without classes.json are the earlier binary SLE-vs-healthy model, which still works.
"""
import html
import json
import os

import gradio as gr
import numpy as np
import pandas as pd

from shared.mil_numpy import load_weights, predict as mil_predict
from shared.plotting import attention_by_group
from shared.ui import connect, file_input, github_link

from . import path
from .features import baseline_features, composition, encode_scvi, log_normalize, predict_celltypes, to_dense

MODEL_REPO = os.environ.get("PBMC_MODEL_REPO", "AdamSion74/pbmc-sle-model")
ARTIFACT_FILES = ("model.npz", "baseline.joblib", "scvi_encoder.npz", "celltypist.npz", "genes.txt")
OPTIONAL_FILES = ("classes.json", "baseline_features.json")   # present for the multi-class (subtype) model
TOP_TYPES = 10


def _artifact_dir() -> str:
    local = path("artifacts")
    if all(os.path.exists(os.path.join(local, f)) for f in ARTIFACT_FILES):
        return local
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_REPO, allow_patterns=[*ARTIFACT_FILES, *OPTIONAL_FILES, "examples/*.h5ad"])


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
        if all(os.path.exists(os.path.join(d, f)) for f in OPTIONAL_FILES):
            art["classes"] = json.load(open(os.path.join(d, "classes.json")))
            art["spec"] = json.load(open(os.path.join(d, "baseline_features.json")))
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
    stats = f"""<div class="stats"><div><span>{len(X):,}</span>cells</div>
        <div><span>{len(present):,}/{len(model_genes):,}</span>model genes</div>"""
    if "classes" in ART:
        if np.isin(cell_type, ART["spec"]["cell_types"]).mean() < 0.5:   # the file's labels use other names
            cell_type = predict_celltypes(ART["celltypist"], log_normalize(X, totals), genes)
        card, att = _predict_subtype(counts, cell_type, z, stats)
    else:
        card, att = _predict_binary(counts, z, stats)
    share = attention_by_group(att, cell_type, top=TOP_TYPES)
    plot_df = pd.DataFrame({"cell type": share.index, "attention share (%)": (share.values * 100).round(2)})
    return card, gr.BarPlot(value=plot_df, visible=True)


def _predict_binary(counts, z, stats):
    """The earlier model: one probability of SLE."""
    p, att = mil_predict(ART["mil"], z)
    p_base = float(ART["baseline"].predict_proba(counts.sum(0, keepdims=True))[0, 1])
    color = "#b91c1c" if p >= 0.5 else "#15803d"
    verdict = "Lupus-like profile" if p >= 0.5 else "Healthy-like profile"
    card = f"""
    <div class="card">
      <div class="label">Probability of SLE</div>
      <div class="big" style="color:{color}">{p:.0%}</div>
      <div class="bar"><div class="fill" style="width:{p * 100:.1f}%;background:{color}"></div></div>
      <div class="verdict" style="color:{color}">{verdict}</div>
      {stats}<div><span>{p_base:.0%}</span>baseline model</div></div>
    </div>"""
    return card, att


def _predict_subtype(counts, cell_type, z, stats):
    """The subtype model: healthy or one SLE subtype, from the MIL and the elastic-net baseline averaged."""
    classes, desc, spec = ART["classes"]["classes"], ART["classes"]["descriptions"], ART["spec"]
    w = ART["mil"]
    use_extra = w["head.weight"].shape[1] > w["w.weight"].shape[0] * w["enc.0.weight"].shape[0]
    p_mil, att = mil_predict(w, z, composition(cell_type, spec["cell_types"]) if use_extra else None)
    base = ART["baseline"]
    p_base = np.zeros(len(classes))
    p_base[base.classes_] = base.predict_proba(baseline_features(counts, cell_type, spec)[None])[0]
    p = (np.asarray(p_mil) + p_base) / 2
    top = int(p.argmax())
    p_sle = 1 - p[0]
    color = "#15803d" if top == 0 else "#b91c1c"
    verdict = "Healthy-like profile" if top == 0 else f"Lupus-like profile · subtype {classes[top]}"
    rows = "".join(
        f"<div class='bp'><div class='bp-name'>{html.escape(c)} <small>{html.escape(desc.get(c, ''))}</small></div>"
        f"<div class='bp-row'><i class='{'hc' if i == 0 else 'sle'}' style='width:{max(p[i] * 100, 0.6):.1f}%'></i>"
        f"<b>{p[i]:.0%}</b></div></div>" for i, c in enumerate(classes))
    card = f"""
    <div class="card">
      <div class="label">Probability of SLE</div>
      <div class="big" style="color:{color}">{p_sle:.0%}</div>
      <div class="bar"><div class="fill" style="width:{p_sle * 100:.1f}%;background:{color}"></div></div>
      <div class="verdict" style="color:{color}">{html.escape(verdict)}</div>
      <div class="classes">{rows}</div>
      {stats}<div><span>{1 - p_base[0]:.0%}</span>baseline P(SLE)</div></div>
    </div>"""
    return card, att


IDLE = "<div class='card'><div class='idle'>Upload a donor's file or pick an example.</div></div>"

FLOW = [
    ("🩸", "Data", "Perez et al. 2022 lupus PBMC atlas (CELLxGENE)", "274 samples · 261 donors"),
    ("🧹", "Preprocess", "QC + platelet filter, Perez cell labels, 5,000 HVGs, batch report", "clean counts per cell"),
    ("🧭", "Represent", "scVI latent space (batch removed), CellTypist on Perez labels, UMAP", "30 numbers + a type per cell"),
    ("🧩", "Subtypes", "Cluster SLE patients by cell abundance + interferon/cytotoxic programs", "healthy or SLE-A/B/…"),
    ("🎯", "Model", "Multi-head attention-MIL + abundance vs elastic-net pseudobulk, donor-split and cohort-held-out CV", "class per sample · AUROC"),
    ("🔍", "Insights", "Attention per type, abundance test, DE, flare vs after treatment", "which cells and genes"),
]


def _metrics_html() -> str:
    f = path("results/metrics.json")
    if not os.path.exists(f):
        return ""
    m = json.load(open(f))
    tile = lambda name, k: (f"<div class='tile'><span>{m[k]['auroc']:.2f}</span>{name}"
                            f"<small>95% CI {m[k]['ci95'][0]:.2f}–{m[k]['ci95'][1]:.2f}</small></div>")
    ens = tile("Ensemble SLE AUROC", "ensemble") if "ensemble" in m else ""
    return (f"<div class='tiles'>{ens}{tile('Attention-MIL SLE AUROC', 'mil')}{tile('Baseline SLE AUROC', 'baseline')}"
            f"<div class='tile'><span>{m['n_donors']}</span>donors<small>{m['n_sle']} SLE · {m['cv']}</small></div></div>")


def _load_json(name: str) -> dict | None:
    f = path(f"results/{name}")
    return json.load(open(f)) if os.path.exists(f) else None


def _bar_pair(name: str, a: float, b: float, scale: float, fmt=lambda v: f"{v:.0%}") -> str:
    w = lambda v: f"{max(v / scale * 100, 0.6):.1f}%"
    return (f"<div class='bp'><div class='bp-name'>{html.escape(name)}</div>"
            f"<div class='bp-row'><i class='sle' style='width:{w(a)}'></i><b>{fmt(a)}</b></div>"
            f"<div class='bp-row'><i class='hc' style='width:{w(b)}'></i><b>{fmt(b)}</b></div></div>")


def _chips(genes, cls="") -> str:
    return "".join(f"<span class='chip {cls}' title='log2 fold change {v:+.2f}'>{html.escape(g)}</span>" for g, v in genes)


def _findings_html() -> str:
    """Key results of the training run, from results/summary.json and results/metrics.json."""
    s, m = _load_json("summary.json"), _load_json("metrics.json")
    if not s:
        return ""
    legend = "<div class='legend'><span><i class='sle'></i>SLE donors</span><span><i class='hc'></i>healthy donors</span></div>"
    att = s["attention"][:6]
    top = att[0]
    fold = top["SLE"] / max(top["healthy"], 1e-6)
    de = s.get("de", [])
    ifn = s.get("ifn")
    n_genes = sum(d["n_sig"] for d in de)

    tiles = []
    if m:
        tiles.append((f"{m['mil']['auroc']:.2f}", "AUROC on unseen donors", f"{m['n_donors']} donors · {m['n_sle']} SLE"))
    tiles.append((f"{fold:.1f}×", f"more attention on {top['cell_type'].lower()}", "in SLE vs healthy donors"))
    if de:
        tiles.append((f"{n_genes:,}", "gene changes (padj < 0.05)", f"across {len(de)} cell types"))
    if m and "macro_auroc" in m.get("ensemble", {}):
        tiles.append((f"{m['ensemble']['macro_auroc']:.2f}", "macro AUROC, healthy vs subtypes", f"{len(m['classes'])} classes"))
    if m and m.get("cohort_cv"):
        cc = m["cohort_cv"]
        tiles.append((f"{np.mean([r['ensemble'] for r in cc.values()]):.2f}", "AUROC on a held-out lab cohort",
                      "cohorts " + ", ".join(cc)))
    if ifn:
        tiles.append((f"{ifn['auroc']:.2f}", "AUROC of an interferon score", f"{len(ifn['genes'])} genes, no ML"))
    tiles_html = "".join(f"<div class='tile'><span>{v}</span>{t}<small>{sub}</small></div>" for v, t, sub in tiles)

    scale = max(max(r["SLE"], r["healthy"]) for r in att)
    attention = "".join(_bar_pair(r["cell_type"], r["SLE"], r["healthy"], scale) for r in att)

    genes = "".join(
        f"<div class='de'><div class='de-head'><b>{html.escape(d['cell_type'])}</b>"
        f"<span>{d['n_sig']:,} genes · {d['n_up']:,} up · {d['n_down']:,} down</span></div>"
        f"<div class='chips'>{_chips(d['top_up'][:6], 'up')}{_chips(d['top_down'][:2], 'down')}</div></div>"
        for d in de[:4])

    ifn_html = ""
    if ifn:
        ifn_html = f"""<div class="fcard"><h4>💉 The interferon signature</h4>
          <p>A simple score, the average of {len(ifn['genes'])} interferon-response genes per donor, separates SLE from
          healthy with AUROC <b>{ifn['auroc']:.2f}</b>. These genes lead the up-regulated list in every cell type tested.</p>
          <div class='chips'>{_chips([(g, 0) for g in ifn['genes']], 'up')}</div>
          <p class="small">Mean score (z-units): SLE <b>{ifn['SLE']:+.2f}</b> · healthy <b>{ifn['healthy']:+.2f}</b></p></div>"""

    batch_html = ""
    if s.get("batches"):
        bscale = max(b["SLE"] + b["healthy"] for b in s["batches"])
        rows = "".join(
            f"<div class='bt'><div class='bt-name'>Cohort {html.escape(b['batch'])}"
            f"{' <em>healthy only</em>' if b['SLE'] == 0 or b['healthy'] == 0 else ''}</div>"
            f"<div class='bt-bar'><i class='sle' style='width:{b['SLE'] / bscale * 100:.1f}%'></i>"
            f"<i class='hc' style='width:{b['healthy'] / bscale * 100:.1f}%'></i></div>"
            f"<div class='bt-n'>{b['SLE']} / {b['healthy']}</div></div>" for b in s["batches"])
        confounded = [b["batch"] for b in s["batches"] if b["SLE"] == 0 or b["healthy"] == 0]
        warn = (f"Cohort {', '.join(confounded)} holds one group only, so lab batch and disease are partly mixed. "
                "scVI removes the batch from the cell embeddings, and DE adds batch as a covariate where the design allows."
                if confounded else "Every cohort holds both groups, so batch is not confounded with disease.")
        batch_html = f"""<div class="fcard"><h4>🧪 Batch check</h4>
          <p>{'Samples' if s.get('endotypes') else 'Donors'} per processing cohort (SLE / healthy).</p>{rows}<p class="small">{warn}</p></div>"""

    base = (f"The simple pseudobulk baseline scores about the same (AUROC {m['baseline']['auroc']:.2f}), so the MIL "
            "model's added value is the per-cell explanation, not accuracy." if m else "")
    conclusions = [
        f"Lupus is clearly visible in blood: the attention-MIL model reaches AUROC {m['mil']['auroc']:.2f} on donors it never saw. {base}" if m else "",
        f"Monocytes carry the signal. {top['cell_type']} get {top['SLE']:.0%} of the model's attention in SLE donors "
        f"vs {top['healthy']:.0%} in healthy donors, where attention goes mostly to {max(att, key=lambda r: r['healthy'])['cell_type']}.",
        "One program dominates: type I interferon response (IFI27, ISG15, EPSTI1, IFI44L) is up in monocytes, T and B cells "
        "alike, matching the known lupus interferon signature." if de else "",
        "Down-regulated genes are mostly ribosomal (RPL/RPS), a sign of shifted cell state rather than a single pathway." if de else "",
    ]
    concl = "".join(f"<li>{c}</li>" for c in conclusions if c)

    return f"""
    <div class="section">Findings</div>
    <div class="headline">🔎 <b>Key result:</b> SLE donors' monocytes are switched into an interferon-response state,
      and that is what the model reads to call lupus.</div>
    <div class="tiles">{tiles_html}</div>
    <div class="fgrid">
      <div class="fcard"><h4>🎯 Where the model looks</h4><p>Share of attention per cell type, averaged over donors.</p>
        {legend}{attention}</div>
      <div class="fcard"><h4>🧬 Genes changed in SLE</h4><p>Pseudobulk DESeq2 per cell type. Strongest genes up
        <span class='chip up'>gene</span> and down <span class='chip down'>gene</span>.</p>{genes}</div>
      {_endotypes_html(s)}{_composition_html(s)}{_paired_html(s)}{ifn_html}{batch_html}
    </div>
    <div class="fcard concl"><h4>✅ Conclusions</h4><ol>{concl}</ol></div>"""


def _ct_chip(cell_type: str, share: float) -> str:
    return f'<span class="chip ct">{html.escape(cell_type)} {share:.0%}</span>'


def _endotypes_html(s: dict) -> str:
    if not s.get("endotypes"):
        return ""
    rows = "".join(
        f"<div class='ecard'><div class='de-head'><b>{html.escape(e['name'])}</b><span>{e['n']} samples</span></div>"
        f"<p class='small'>{html.escape(e['description'])}</p>"
        f"<div class='chips'>{''.join(_ct_chip(ct, v) for ct, v in e['attention'])}</div></div>"
        for e in s["endotypes"])
    return f"""<div class="fcard"><h4>🧩 SLE subtypes</h4><p>SLE samples clustered by cell-type abundance and gene
      programs (interferon, cytotoxic) per lineage, each centred on healthy donors of the same lab cohort. Chips: where
      the model looks for that subtype.</p>{rows}</div>"""


def _composition_html(s: dict) -> str:
    comp = s.get("composition")
    if not comp:
        return ""
    legend = "<div class='legend'><span><i class='sle'></i>SLE donors</span><span><i class='hc'></i>healthy donors</span></div>"
    scale = max(max(r["SLE"], r["healthy"]) for r in comp)
    bars = "".join(_bar_pair(f"{r['cell_type']}  (p={r['p']:.0e})", r["SLE"], r["healthy"], scale, lambda v: f"{v:.1%}") for r in comp)
    return f"""<div class="fcard"><h4>📊 Cell abundance</h4><p>Mean share of a donor's cells; Perez's weighted
      least squares test, Bonferroni-adjusted p.</p>{legend}{bars}</div>"""


def _paired_html(s: dict) -> str:
    pr = s.get("paired")
    if not pr:
        return ""
    de = "".join(
        f"<div class='de'><div class='de-head'><b>{html.escape(d['lineage'])}</b><span>{d['n_sig']:,} genes change</span></div>"
        f"<div class='chips'>{_chips(d['top_down'][:6], 'down')}{_chips(d['top_up'][:2], 'up')}</div></div>"
        for d in pr["de"] if d["n_sig"])
    comp = ", ".join(f"{html.escape(r['cell_type'])} {r['flare']:.1f}% → {r['treated']:.1f}%" for r in pr["composition"][:3])
    return f"""<div class="fcard"><h4>💊 Flare → after treatment</h4><p>The same {pr['n_pairs']} patients sampled at
      flare and after treatment (same lab cohort, so no batch effect). Genes going down after treatment
      <span class='chip down'>gene</span>.</p>{de}<p class="small">Largest abundance shifts: {comp}</p></div>"""


def _architecture_html() -> str:
    nodes = "<div class='arrow'>→</div>".join(
        f"<div class='node'><div class='n'>STEP {i}</div><div class='ico'>{ic}</div><b>{t}</b>"
        f"<span>{d}</span><em>{o}</em></div>" for i, (ic, t, d, o) in enumerate(FLOW, 1))
    return f"""
    <div class="flow">{nodes}</div>
    {_metrics_html()}
    <p class="note">Each blood sample is a bag of cells with one label: healthy, or one of the SLE subtypes found by
      clustering patients. scVI compresses every cell to 30 numbers while removing lab-batch effects; the multi-head
      attention-MIL model scores each cell, pools the cells by those scores, adds the sample's cell-type abundance
      and outputs class probabilities. An elastic-net regression on per-lineage pseudobulk plus abundance (as in
      Perez et al.) is the baseline; the demo averages both. Folds are split by donor, and a second test holds out a
      whole lab cohort.</p>
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
    sample, _, label = os.path.basename(p).removesuffix(".h5ad").rpartition("_")
    return f"{sample} · {label}"


def build() -> None:
    """Render the PBMC SLE tab inside the current gr.Blocks context."""
    status = "Live demo" if not LOAD_ERROR else "Training"
    pill = "live" if not LOAD_ERROR else "soon"
    gr.HTML(f"""<div class="page-head"><h2>🧬 PBMC SLE Classification</h2>
      <p>Classify a blood sample as healthy or a lupus (SLE) subtype from single-cell RNA-seq, and see which cells drove it.
      <span class="pill {pill}">{status}</span></p>{github_link("pbmc_sle_classification")}</div>""")
    if LOAD_ERROR:
        gr.HTML(f"<div class='banner'>The model has not been uploaded to <code>{MODEL_REPO}</code> yet, so "
                "predictions are off. Train it with the steps in the project README, then <code>make upload</code>.</div>")
    findings = _findings_html()
    if findings:
        gr.HTML(findings)
        gr.HTML("<div class='section demo-title'>Try it</div>")
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
