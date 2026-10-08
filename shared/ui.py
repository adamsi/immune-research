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


GITHUB_ICON = ("<svg width='18' height='18' viewBox='0 0 16 16' fill='currentColor' aria-hidden='true'><path d='M8 0C3.58 "
               "0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94"
               "-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87"
               ".51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 "
               "1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 "
               "3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 "
               "8c0-4.42-3.58-8-8-8z'/></svg>")
REPO_URL = "https://github.com/adamsi/immune-research/tree/main/projects/"


def github_link(project: str) -> str:
    """A small 'View code' link to the project's folder on GitHub, for the page header."""
    return f"<a class='gh' href='{REPO_URL}{project}' target='_blank' rel='noopener'>{GITHUB_ICON}View code on GitHub</a>"
