# gradio_app.py

import gradio as gr

def launch_ui():
    with gr.Blocks() as app:
        gr.Markdown("# ClinicalMind")
    return app
