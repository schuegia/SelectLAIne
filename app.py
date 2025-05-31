import os
import gradio as gr
from main import process_query

# .env / HF Secret muss lediglich OPENAI_API_KEY enthalten, wenn du OpenAI-Embeddings nutzt
os.environ["OPENAI_API_KEY"] = os.getenv("HF_OPENAI_API_KEY")


def answer(q):
    ans, sources = process_query(q)
    src_str = "\n".join(f"{d.metadata['source']}#chunk{d.metadata['chunk']}"
                        for d in sources)
    return ans, src_str


demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(lines=2, placeholder="Frage eingeben..."),
    outputs=[gr.Textbox(label="Antwort"), gr.Textbox(label="Quellen")],
    title="SelectLine ERP RAG",
    description="Stelle Fragen zu deinem SelectLine-Handbuch"
)

if __name__ == "__main__":
    demo.launch()