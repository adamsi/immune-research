"""Immune Research: one Gradio app with a tab per project (deployed as a single Hugging Face Space)."""
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
from projects.pbmc_sle_classification import demo as pbmc  # noqa: E402

PROJECTS = [
    {"id": "cart-responder", "icon": "🧬", "title": "CAR-T Responder", "status": "Live demo", "cta": "Open demo →",
     "blurb": "Predict CAR-T response from blood cells."},
    {"id": "pbmc-sle", "icon": "🧬", "title": "PBMC SLE", "status": "Training" if pbmc.LOAD_ERROR else "Live demo",
     "cta": "Open demo →", "blurb": "Lupus vs healthy from PBMC single-cell data."},
    {"id": "foundation-model", "icon": "🌐", "title": "Foundation Model", "status": "Coming soon", "cta": "Learn more →",
     "blurb": "A pretrained single-cell model for immunology."},
]

FM_TEXT = """
<div class="page-head"><h2>🌐 Foundation Model</h2>
<p>A single-cell foundation model for immunology, pretrained once and fine-tuned for each project here.
<span class="pill soon">Coming soon</span></p></div>"""

FOOT = "<div id='foot'>Immune Research · single-cell immunology and machine learning · for research use only</div>"

CSS = """
.gradio-container { width: 100% !important; max-width: 1120px !important; margin: 0 auto !important; font-size: 18px; }
#hero { padding: clamp(18px, 4vw, 40px) 0 clamp(14px, 2.5vw, 26px); }
#hero h1 { font-size: clamp(30px, 6vw, 48px); font-weight: 700; letter-spacing: -0.025em; margin: 0 0 8px; line-height: 1.1; }
#hero p  { font-size: clamp(17px, 2.4vw, 21px); color: var(--body-text-color-subdued); margin: 0; line-height: 1.45; }

/* tabs: big and thumb-friendly; Gradio folds overflow into its own "..." menu */
button[role="tab"] { font-size: clamp(16px, 2.2vw, 18px) !important; padding: 12px 16px !important; }

/* project cards: compact so all of them fit on one screen */
.cards { display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: clamp(12px, 2vw, 20px) !important; align-items: stretch !important; }
.cards > .pcard { min-width: 0 !important; width: auto !important; height: 100%; }
.pcard .pill { margin-top: 6px; }
.pcard { border-radius: 18px !important; padding: clamp(16px, 2.5vw, 24px) !important; gap: 12px !important;
         border: 1px solid var(--border-color-primary) !important; background: var(--block-background-fill) !important; }
.pcard { position: relative; }
.pcard .head { display: flex; align-items: flex-start; gap: 14px; }
.pcard .txt { flex: 1; min-width: 0; }
.pcard .go { display: none; }
.pcard .icon { font-size: 30px; line-height: 1; width: 54px; height: 54px; flex: none; display: grid; place-items: center;
               border-radius: 14px; background: var(--primary-50); }
.pcard h3 { font-size: clamp(19px, 2.4vw, 22px); font-weight: 650; margin: 0; letter-spacing: -0.01em; line-height: 1.3; }
.pcard p  { font-size: 16px; line-height: 1.45; color: var(--body-text-color-subdued); margin: 6px 0 0; }
.pill { display: inline-block; font-size: 12px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
        border-radius: 99px; padding: 3px 10px; vertical-align: middle; }
.pill.live { color: #15803d; background: #dcfce7; }
.pill.soon { color: var(--body-text-color-subdued); background: var(--border-color-primary); }
.pcard button { font-size: 16px !important; min-height: 46px; margin-top: auto; }

/* project pages */
.page-head { padding: clamp(18px, 4vw, 30px) 0 clamp(14px, 2.5vw, 22px); }
.page-head h2 { font-size: clamp(28px, 5vw, 40px); font-weight: 700; letter-spacing: -0.02em; margin: 0 0 10px; line-height: 1.15; }
.page-head p  { font-size: clamp(17px, 2.3vw, 20px); color: var(--body-text-color-subdued); margin: 0; max-width: 760px; line-height: 1.55; }
/* uploader: one calm row, our own labels (no browser-language strings), then example chips */
.uploader { align-items: center !important; gap: 12px !important; padding: 14px 16px !important; flex-wrap: nowrap !important;
            border: 1.5px dashed var(--border-color-primary) !important; border-radius: 14px !important;
            background: var(--block-background-fill) !important; }
.uploader:hover { border-color: var(--primary-400) !important; }
.uploader .upl-status { flex: 1 1 auto !important; min-width: 0 !important; }
.uploader .upl-status .html-container { padding: 0 !important; }
.upl-empty, .upl-file { display: flex; align-items: center; gap: 12px; min-width: 0; }
.upl-icon { width: 40px; height: 40px; flex: none; display: grid; place-items: center; border-radius: 10px;
            background: var(--primary-50); color: var(--primary-600); }
.upl-title, .upl-name { font-size: 16px; font-weight: 600; line-height: 1.3; }
.upl-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.upl-meta { min-width: 0; }
.upl-hint { font-size: 14px; color: var(--body-text-color-subdued); line-height: 1.3; }
.upl-ok { margin-left: auto; color: #15803d; font-weight: 700; }
.uploader .upl-btn { flex: none !important; width: auto !important; min-width: 0 !important; white-space: nowrap; }
.chips { gap: 8px !important; flex-wrap: wrap !important; justify-content: flex-start !important; margin-top: -4px; }
.chips-label { font-size: 13px; color: var(--body-text-color-subdued); margin-top: 2px; }
.chips .chip { flex: none !important; width: auto !important; min-width: 0 !important; border-radius: 99px !important;
               font-size: 14px !important; padding: 4px 12px !important; }

/* architecture: step flow, then result tiles */
.arch-title { margin-top: 28px; }
.flow { display: flex; align-items: stretch; gap: 8px; margin: 8px 0 4px; }
.flow .node { flex: 1 1 0; min-width: 0; padding: 14px; border-radius: 14px; border: 1px solid var(--border-color-primary);
              background: var(--block-background-fill); display: flex; flex-direction: column; gap: 4px; }
.flow .n { font-size: 12px; font-weight: 600; letter-spacing: .06em; color: var(--primary-600); }
.flow .ico { font-size: 22px; line-height: 1.2; }
.flow b { font-size: 17px; }
.flow span { font-size: 14px; line-height: 1.45; color: var(--body-text-color-subdued); }
.flow em { font-style: normal; font-size: 13px; margin-top: auto; padding-top: 6px; color: var(--primary-700); }
.flow .arrow { align-self: center; color: var(--body-text-color-subdued); font-size: 18px; flex: none; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 200px), 1fr)); gap: 12px; margin-top: 14px; }
.tile { padding: 14px 16px; border-radius: 14px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
        font-size: 14px; color: var(--body-text-color-subdued); }
.tile span { display: block; font-size: 28px; font-weight: 700; color: var(--body-text-color); letter-spacing: -0.02em; }
.tile small { display: block; font-size: 12px; margin-top: 2px; }
@media (max-width: 900px) {
  .cards { grid-template-columns: 1fr; }
  .flow { flex-direction: column; }
  .flow .arrow { transform: rotate(90deg); margin: -2px 0; }
}
.fmt { margin: 0; padding-left: 20px; font-size: 15px; line-height: 1.6; }

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

/* GitHub link in page headers */
.gh { display: inline-flex; align-items: center; gap: 8px; margin-top: 14px; padding: 8px 14px; border-radius: 99px;
      font-size: 15px; font-weight: 550; text-decoration: none !important; color: var(--body-text-color) !important;
      border: 1px solid var(--border-color-primary); background: var(--block-background-fill); }
.gh:hover { border-color: var(--primary-500); color: var(--primary-600) !important; }

/* PBMC findings */
.headline { font-size: clamp(17px, 2.3vw, 20px); line-height: 1.5; padding: 16px 20px; border-radius: 16px; margin-top: 6px;
            background: var(--primary-50); color: var(--primary-800, var(--body-text-color)); border: 1px solid var(--primary-200); }
.fgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 420px), 1fr)); gap: 14px; margin-top: 14px; }
.fcard { padding: 18px 20px; border-radius: 16px; border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
         min-width: 0; }
.fcard h4 { font-size: 19px; font-weight: 650; margin: 0 0 4px; }
.fcard p { font-size: 15px; line-height: 1.5; color: var(--body-text-color-subdued); margin: 0 0 12px; }
.fcard p.small { font-size: 14px; margin: 12px 0 0; }
.fcard.concl { margin-top: 14px; }
.fcard ol { margin: 6px 0 0; padding-left: 22px; font-size: 16px; line-height: 1.6; }
.fcard li + li { margin-top: 6px; }
i.sle, .legend i.sle { background: #e11d48; }
i.hc, .legend i.hc { background: #10b981; }
.legend { display: flex; gap: 16px; font-size: 13px; color: var(--body-text-color-subdued); margin-bottom: 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }
.bp { margin-top: 10px; }
.bp-name { font-size: 14px; font-weight: 550; margin-bottom: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bp-row { display: flex; align-items: center; gap: 8px; height: 14px; margin: 2px 0; }
.bp-row i { display: block; height: 9px; border-radius: 99px; flex: none; max-width: calc(100% - 48px); }
.bp-row b { font-size: 12px; font-weight: 600; color: var(--body-text-color-subdued); }
.bp-name small { font-weight: 400; color: var(--body-text-color-subdued); margin-left: 6px; }
.card .classes { text-align: left; margin: 0 auto 16px; max-width: 420px; }
.ecard + .ecard { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-color-primary); }
.de { padding: 10px 0; border-top: 1px solid var(--border-color-primary); }
.de:first-of-type { border-top: 0; padding-top: 0; }
.de-head { display: flex; justify-content: space-between; gap: 10px; flex-wrap: wrap; align-items: baseline; margin-bottom: 6px; }
.de-head b { font-size: 15px; font-weight: 600; }
.de-head span { font-size: 13px; color: var(--body-text-color-subdued); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; font-weight: 600; padding: 3px 9px;
        border-radius: 8px; white-space: nowrap; }
.chip.up { color: #be123c; background: #ffe4e6; }
.chip.down { color: #475569; background: #e2e8f0; }
.chip.ct { color: #1d4ed8; background: #dbeafe; font-family: inherit; }
.bt { display: grid; grid-template-columns: minmax(0, 150px) 1fr auto; align-items: center; gap: 10px; margin: 8px 0; }
.bt-name { font-size: 14px; font-weight: 550; }
.bt-name em { font-style: normal; font-size: 11px; font-weight: 600; color: #b45309; background: #fef3c7; border-radius: 6px; padding: 1px 6px; margin-left: 4px; white-space: nowrap; }
.bt-bar { display: flex; height: 12px; border-radius: 99px; overflow: hidden; background: var(--border-color-primary); }
.bt-bar i { display: block; height: 100%; }
.bt-n { font-size: 13px; color: var(--body-text-color-subdued); font-variant-numeric: tabular-nums; }
.demo-title { margin-top: 30px !important; }

@media (max-width: 640px) {
  #hero { padding: 6px 0 4px; }
  #hero h1 { font-size: 30px; margin-bottom: 4px; }
  #hero p { font-size: 16px; }
  .pcard .html-container { padding: 0 !important; }
  .cards { grid-template-columns: 1fr; gap: 10px !important; }
  .pcard { padding: 14px 16px !important; }
  .pcard .head { align-items: center; }
  .pcard .icon { width: 44px; height: 44px; font-size: 24px; border-radius: 12px; }
  .pcard h3 { font-size: 18px; }
  .pcard .pill { font-size: 11px; padding: 2px 8px; }
  .pcard p { font-size: 15px; margin-top: 3px; line-height: 1.4; }
  .pcard .go { display: block; font-size: 30px; line-height: 1; color: var(--body-text-color-subdued); }
  /* on phones the whole card is the tap target: stretch the button over it, invisibly */
  .pcard button { position: absolute !important; inset: 0; opacity: 0; min-height: 0; margin: 0; }
  .card .stats { gap: 18px; }
  .fcard { padding: 14px 16px; }
  .tiles { grid-template-columns: 1fr 1fr; gap: 10px; }
  .tile { padding: 12px; }
  .tile span { font-size: 24px; }
  .bt { grid-template-columns: minmax(0, 120px) 1fr auto; }
}

footer { display: none !important; }
#foot { color: var(--body-text-color-subdued); font-size: 14px; padding: 28px 0 12px; border-top: 1px solid var(--border-color-primary); margin-top: 32px; }
"""

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate", font=gr.themes.GoogleFont("Inter"),
                       radius_size=gr.themes.sizes.radius_lg, text_size=gr.themes.sizes.text_lg)


def _card_html(p: dict) -> str:
    pill = "live" if p["status"] == "Live demo" else "soon"
    return (f"<div class='head'><div class='icon'>{p['icon']}</div><div class='txt'>"
            f"<h3>{p['title']}</h3><span class='pill {pill}'>{p['status']}</span><p>{p['blurb']}</p>"
            f"</div><span class='go'>›</span></div>")


TABS = ("home", "cart-responder", "pbmc-sle", "foundation-model")


# Client-side: put ?tab=<slug> in the address bar so the current tab can be shared as a link (Home clears it).
# On huggingface.co/spaces/... the app is an iframe, so also ask the Space page to update its own URL
# (Hugging Face listens for this postMessage and forwards its query string to the app on load).
SET_URL_JS = """(slug) => { const u = new URL(window.location.href);
  if (slug === "home") u.searchParams.delete("tab"); else u.searchParams.set("tab", slug);
  history.replaceState(null, "", u);
  if (window.parent !== window) window.parent.postMessage({ queryString: u.searchParams.toString() }, "https://huggingface.co"); }"""


def _url_js(slug: str) -> str:
    return f"() => ({SET_URL_JS})('{slug}')"


def _picked_slug(evt: gr.SelectData) -> str:
    return evt.value if evt.value in TABS else TABS[evt.index]


def _tab_from_url(request: gr.Request):
    slug = request.query_params.get("tab", "home")
    return gr.Tabs(selected=slug if slug in TABS else "home")


with gr.Blocks(title="Immune Research") as demo:
    with gr.Tabs() as tabs:
        with gr.Tab("Home", id="home"):
            gr.HTML("""<div id="hero"><h1>Immune Research</h1>
              <p>Machine learning on single-cell immune data.</p></div>""")
            with gr.Row(elem_classes="cards"):
                for p in PROJECTS:
                    with gr.Column(min_width=260, elem_classes="pcard"):
                        gr.HTML(_card_html(p))
                        btn = gr.Button(p["cta"], variant="primary")
                        btn.click(lambda tab=p["id"]: gr.Tabs(selected=tab), outputs=tabs).then(None, js=_url_js(p["id"]))
        with gr.Tab("🧬 CAR-T Responder", id="cart-responder"):
            cart.build()
        with gr.Tab("🧬 PBMC SLE", id="pbmc-sle"):
            pbmc.build()
        with gr.Tab("🌐 Foundation Model", id="foundation-model"):
            gr.HTML(FM_TEXT)
    gr.HTML(FOOT)

    # mirror the picked tab in ?tab= (Home clears it), so the address bar is always shareable
    slug_box = gr.Textbox(visible=False)
    tabs.select(_picked_slug, outputs=slug_box).then(None, inputs=slug_box, js=SET_URL_JS)
    demo.load(_tab_from_url, outputs=tabs)  # deep links: ?tab=pbmc-sle opens that tab

if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME, allowed_paths=[cart.EX] + ([pbmc.EX] if pbmc.EX else []), server_name="0.0.0.0")
