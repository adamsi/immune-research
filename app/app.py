"""AIMMUNE Research: one Gradio app with a tab per project (deployed as a single Hugging Face Space)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # so relative paths and Gradio's file cache work from any launch directory

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass

import gradio as gr  # noqa: E402

from projects.cart_responder import demo as cart  # noqa: E402

PROJECTS = [
    {"id": "cart", "icon": "🧬", "title": "CAR-T Responder", "status": "Live demo", "cta": "Open demo →",
     "blurb": "Predict response to CD19 CAR-T therapy from a pre-treatment blood single-cell sample, "
              "with attention showing which immune cells mattered."},
    {"id": "sle", "icon": "🦋", "title": "PBMC SLE Classification", "status": "Coming soon", "cta": "View plan →",
     "blurb": "Classify lupus patients from healthy donors using PBMC single-cell RNA-seq, "
              "and find the cell states that separate them."},
    {"id": "fm", "icon": "🌐", "title": "Foundation Model", "status": "Coming soon", "cta": "Learn more →",
     "blurb": "A pretrained single-cell immune foundation model to power every downstream task in this hub."},
]

SLE_PLAN = """
<div class="page-head"><h2>🦋 PBMC SLE Classification</h2>
<p>Patient-level lupus classification from blood single-cell RNA-seq. <span class="pill soon">Coming soon</span></p></div>
<ol class="plan">
  <li><b>Data</b><span>Perez et al. 2022 (<i>Science</i>) lupus PBMC atlas, downloaded from CELLxGENE.</span></li>
  <li><b>QC</b><span>scanpy quality control, normalisation and highly variable genes.</span></li>
  <li><b>Latent space</b><span>scANVI embedding and harmonised cell-type labels.</span></li>
  <li><b>Baseline</b><span>Pseudobulk per patient and cell type, then logistic regression.</span></li>
  <li><b>Attention-MIL</b><span>Each patient is a bag of cells, shared with the CAR-T project.</span></li>
  <li><b>Evaluation</b><span>Patient-split cross-validation, reported as AUROC.</span></li>
  <li><b>Insights</b><span>Attention by cell type plus pseudobulk differential expression.</span></li>
</ol>"""

FM_TEXT = """
<div class="page-head"><h2>🌐 Foundation Model</h2>
<p>A single-cell foundation model for immunology, pretrained once and fine-tuned for each project here.
<span class="pill soon">Coming soon</span></p></div>"""

FOOT = "<div id='foot'>AIMMUNE Research · single-cell immunology and machine learning · for research use only</div>"

CSS = """
.gradio-container { width: 100% !important; max-width: 1120px !important; margin: 0 auto !important; font-size: 18px; }
#hero { padding: clamp(28px, 6vw, 56px) 0 clamp(18px, 3vw, 28px); }
#hero .tag { display: inline-block; font-size: 14px; letter-spacing: .06em; text-transform: uppercase;
             color: var(--primary-600); background: var(--primary-50); border-radius: 8px; padding: 5px 12px; margin-bottom: 16px; }
#hero h1 { font-size: clamp(34px, 7vw, 56px); font-weight: 700; letter-spacing: -0.025em; margin: 0 0 12px; line-height: 1.08; }
#hero p  { font-size: clamp(18px, 2.6vw, 22px); color: var(--body-text-color-subdued); margin: 0; max-width: 760px; line-height: 1.55; }

/* tabs: big and thumb-friendly; Gradio folds overflow into its own "..." menu */
button[role="tab"] { font-size: clamp(16px, 2.2vw, 18px) !important; padding: 12px 16px !important; }

/* project cards */
.cards { gap: clamp(14px, 2.5vw, 22px) !important; }
.pcard { border-radius: 18px !important; padding: clamp(20px, 3.5vw, 28px) !important; gap: 14px !important;
         border: 1px solid var(--border-color-primary) !important; background: var(--block-background-fill) !important; }
.pcard .icon { font-size: 44px; line-height: 1; width: 72px; height: 72px; display: grid; place-items: center;
               border-radius: 18px; background: var(--primary-50); margin-bottom: 14px; }
.pcard h3 { font-size: clamp(22px, 3vw, 26px); font-weight: 650; margin: 0 0 8px; letter-spacing: -0.01em; }
.pcard p  { font-size: 17px; line-height: 1.55; color: var(--body-text-color-subdued); margin: 10px 0 0; }
.pill { display: inline-block; font-size: 13px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
        border-radius: 99px; padding: 4px 11px; vertical-align: middle; }
.pill.live { color: #15803d; background: #dcfce7; }
.pill.soon { color: var(--body-text-color-subdued); background: var(--border-color-primary); }
.pcard button { font-size: 17px !important; min-height: 50px; margin-top: auto; }

/* project pages */
.page-head { padding: clamp(18px, 4vw, 30px) 0 clamp(14px, 2.5vw, 22px); }
.page-head h2 { font-size: clamp(28px, 5vw, 40px); font-weight: 700; letter-spacing: -0.02em; margin: 0 0 10px; line-height: 1.15; }
.page-head p  { font-size: clamp(17px, 2.3vw, 20px); color: var(--body-text-color-subdued); margin: 0; max-width: 760px; line-height: 1.55; }
.plan { list-style: none; counter-reset: s; padding: 0; margin: 0; display: grid; gap: 14px;
        grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr)); }
.plan li { counter-increment: s; padding: 20px; border-radius: 14px; border: 1px solid var(--border-color-primary);
           background: var(--block-background-fill); }
.plan li::before { content: "STEP " counter(s); display: block; font-size: 13px; font-weight: 600; letter-spacing: .06em; color: var(--primary-600); }
.plan b { display: block; font-size: 19px; margin: 6px 0; }
.plan span { font-size: 16px; line-height: 1.55; color: var(--body-text-color-subdued); }

/* CAR-T demo */
.section { font-size: 14px; letter-spacing: .06em; text-transform: uppercase; color: var(--body-text-color-subdued); margin: 4px 0 2px; }
.card { text-align: center; padding: 30px 20px 22px; border-radius: 16px; background: var(--block-background-fill);
        border: 1px solid var(--border-color-primary); }
.card .label { font-size: 16px; color: var(--body-text-color-subdued); }
.card .big   { font-size: clamp(60px, 12vw, 84px); font-weight: 700; letter-spacing: -0.03em; line-height: 1.05; margin: 6px 0 10px; }
.card .bar   { height: 10px; border-radius: 99px; background: var(--border-color-primary); overflow: hidden; margin: 0 auto 14px; max-width: 420px; }
.card .fill  { height: 100%; border-radius: 99px; transition: width .5s ease; }
.card .verdict { font-size: clamp(20px, 3vw, 24px); font-weight: 600; margin-bottom: 18px; }
.card .stats { display: flex; justify-content: center; gap: 28px; flex-wrap: wrap; border-top: 1px solid var(--border-color-primary); padding-top: 16px; }
.card .stats div { font-size: 14px; color: var(--body-text-color-subdued); }
.card .stats span { display: block; font-size: 22px; font-weight: 600; color: var(--body-text-color); }
.card .idle { font-size: 18px; color: var(--body-text-color-subdued); padding: 34px 0; }
.banner { background: var(--primary-50); color: var(--primary-700); border: 1px solid var(--primary-200);
          border-radius: 12px; padding: 12px 16px; margin-bottom: 12px; font-size: 16px; }
.diagram { padding: 6px 0 18px; overflow-x: auto; contain: inline-size; }
.diagram svg { min-width: 640px; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 220px), 1fr)); gap: 14px; margin-top: 4px; }
.step { padding: 18px; border-radius: 12px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.step .n { font-size: 13px; font-weight: 600; color: var(--primary-600); letter-spacing: .06em; }
.step b { display: block; margin: 6px 0; font-size: 18px; font-weight: 600; }
.step span { font-size: 16px; line-height: 1.5; color: var(--body-text-color-subdued); }
.note { font-size: 17px; line-height: 1.6; margin-top: 16px; }
.note.muted { font-size: 15px; color: var(--body-text-color-subdued); }

@media (max-width: 640px) {
  .pcard { padding: 20px !important; }
  .card .stats { gap: 18px; }
}

footer { display: none !important; }
#foot { color: var(--body-text-color-subdued); font-size: 14px; padding: 28px 0 12px; border-top: 1px solid var(--border-color-primary); margin-top: 32px; }
"""

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", font=gr.themes.GoogleFont("Inter"),
                       radius_size=gr.themes.sizes.radius_lg, text_size=gr.themes.sizes.text_lg)


def _card_html(p: dict) -> str:
    pill = "live" if p["status"] == "Live demo" else "soon"
    return (f"<div class='icon'>{p['icon']}</div><h3>{p['title']}</h3>"
            f"<span class='pill {pill}'>{p['status']}</span><p>{p['blurb']}</p>")


with gr.Blocks(title="AIMMUNE Research") as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("🏠 Home", id="home"):
            gr.HTML("""<div id="hero"><div class="tag">🔬 Research hub</div>
              <h1>AIMMUNE Research</h1>
              <p>Machine learning on single-cell immune data: predicting therapy response and disease
                 from the cells in a blood sample.</p></div>""")
            with gr.Row(elem_classes="cards"):
                for p in PROJECTS:
                    with gr.Column(min_width=280, elem_classes="pcard"):
                        gr.HTML(_card_html(p))
                        btn = gr.Button(p["cta"], variant="primary" if p["id"] == "cart" else "secondary")
                        btn.click(lambda tab=p["id"]: gr.Tabs(selected=tab), outputs=tabs)
        with gr.Tab("🧬 CAR-T Responder", id="cart"):
            cart.build()
        with gr.Tab("🦋 PBMC SLE", id="sle"):
            gr.HTML(SLE_PLAN)
        with gr.Tab("🌐 Foundation Model", id="fm"):
            gr.HTML(FM_TEXT)
    gr.HTML(FOOT)

if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, allowed_paths=[cart.EX], server_name="0.0.0.0")
