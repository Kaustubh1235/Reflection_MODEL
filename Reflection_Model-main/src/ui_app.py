import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from src.qa_wrapper import generate_response

def process_input(q):
    return generate_response(q)

with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gr-button {font-size: 16px !important; font-weight: 600;}
    .gr-textbox textarea {font-family: 'Fira Code', monospace;}
    footer {display: none !important;}
""") as demo:

    gr.Markdown("""
    <div style='text-align:center; font-size: 28px; font-weight: bold;'>🤖 Reflective QA System</div>
    <div style='text-align:center; font-size: 16px; margin-top: -10px;'>Get high-quality answers with self-critiquing AI</div>
    """)

    with gr.Row():
        question = gr.Textbox(
            label="🔍 Enter Your Question",
            placeholder="e.g., What is XSS?",
            lines=2
        )
        with gr.Column(scale=0.3):
            submit = gr.Button("🚀 Generate", variant="primary")
            clear = gr.Button("🧹 Clear")

    with gr.Accordion("🧠 LLM Answer", open=True):
        llm_out = gr.Markdown()

    with gr.Accordion("🧪 T5 Reflection", open=False):
        reflection_out = gr.Markdown()

    with gr.Accordion("✨ Improved Answer", open=True):
        improved_out = gr.Markdown()

    with gr.Row():
        status = gr.Textbox(label="📦 System Status", interactive=False, max_lines=1)

    submit.click(process_input, inputs=question, outputs=[llm_out, reflection_out, improved_out, status])
    clear.click(lambda: ("", "", "", ""), None, [llm_out, reflection_out, improved_out, status])


demo.launch(inbrowser=True, server_name="127.0.0.1", server_port=5000)

