"""Reusable Gradio pieces for the project tabs."""
import html
import os

import gradio as gr

UPLOAD_ICON = ("<svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' "
               "stroke-linecap='round' stroke-linejoin='round'><path d='M12 16V4'/><path d='m7 9 5-5 5 5'/>"
               "<path d='M20 16v3a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1v-3'/></svg>")
FILE_ICON = ("<svg width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' "
             "stroke-linecap='round' stroke-linejoin='round'><path d='M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z'/>"
             "<path d='M14 3v5h5'/></svg>")


def _size(path: str) -> str:
    n = os.path.getsize(path)
    return f"{n / 1e6:.1f} MB" if n >= 1e5 else f"{n / 1e3:.0f} KB"


def _empty_html(title: str, hint: str) -> str:
    return (f"<div class='upl-empty'><div class='upl-icon'>{UPLOAD_ICON}</div>"
            f"<div><div class='upl-title'>{title}</div><div class='upl-hint'>{hint}</div></div></div>")


def _file_html(path: str, tag: str) -> str:
    name = html.escape(os.path.basename(path))
    return (f"<div class='upl-file'><div class='upl-icon'>{FILE_ICON}</div><div class='upl-meta'>"
            f"<div class='upl-name'>{name}</div><div class='upl-hint'>{_size(path)} · {tag}</div></div>"
            f"<span class='upl-ok'>✓</span></div>")


def file_input(title: str, hint: str, file_types: list[str], examples: list[tuple[str, str]] = ()):
    """A compact uploader: one row with an icon, title/hint (or the chosen file) and a "Choose file" button,
    plus optional example chips. Uses our own labels, so it reads the same in every browser language.
    Returns (state holding the file path, list of (event trigger, value fn) to wire with `connect`)."""
    state = gr.State(None)
    with gr.Row(elem_classes="uploader", equal_height=True):
        status = gr.HTML(_empty_html(title, hint), elem_classes="upl-status")
        button = gr.UploadButton("Choose file", file_types=file_types, file_count="single",
                                 size="md", variant="secondary", elem_classes="upl-btn")
    chips = []
    if examples:
        gr.HTML("<div class='chips-label'>Or try an example</div>")
        with gr.Row(elem_classes="chips"):
            for label, p in examples:
                chips.append((gr.Button(label, size="sm", elem_classes="chip"), p))
    return {"state": state, "status": status, "button": button, "chips": chips}


def connect(widget: dict, fn, outputs: list) -> None:
    """Run `fn(path)` whenever a file is uploaded or an example chip is clicked."""
    state, status = widget["state"], widget["status"]
    widget["button"].upload(lambda f: (f, _file_html(f, "uploaded")), widget["button"], [state, status]) \
        .then(fn, state, outputs)
    for btn, p in widget["chips"]:
        btn.click(lambda p=p: (p, _file_html(p, "example")), None, [state, status]).then(fn, state, outputs)
