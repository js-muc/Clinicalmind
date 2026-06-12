import gradio as gr
from handlers.router_handler import route_query
from core.config import (
    model

)
from rag.loader import (
    load_document,  
)
 
# -------------------------------
# 🔧 ANSWER FUNCTION
# -------------------------------
def answer_question(file, question):
    if file is None:
        return "Please upload a PDF first."

    load_document(file, model)

    return route_query(
        question,
        model
    )
    

# -------------------------------
# UI
# -------------------------------
app = gr.Interface(
    fn=answer_question,
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a question")],
    outputs=gr.Textbox(
        lines=25,
        max_lines=40,
        label="ClinicalMind Output"
    ),
    title="📄 Ask Your PDF ClinicalMind",
    description="Upload a document and ask questions"
)

if __name__ == "__main__":
    app.launch()