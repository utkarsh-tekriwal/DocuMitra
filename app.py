import os
import fitz
import gradio as gr
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline

# Pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file_path):
    if file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        return "".join(page.get_text() for page in doc)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def build_vector_store(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def summarize_document(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    summaries = []
    for chunk in chunks[:3]:  # Limit to ~3000 tokens total
        input_text = (
            "stick strictly to the document uploaded and don't generate anything out of that document.\n\n" + chunk
        )
        summary = summarizer(input_text, max_length=100, min_length=60, do_sample=False)
        summaries.append(summary[0]["summary_text"])
    combined_summary = " ".join(summaries)
    return combined_summary.strip()

def retrieve_context(query, chunks, embeddings, index, return_indices=False):
    query_vec = model.encode([query])
    _, I = index.search(np.array(query_vec), k=3)
    context = "\n".join([chunks[i] for i in I[0]])
    if return_indices:
        return context, I[0]
    return context

def ask_question(question, chunks, index):
    context, indices = retrieve_context(question, chunks, model.encode(chunks), index, return_indices=True)
    result = qa_pipeline(question=question, context=context)
    answer = result["answer"]
    reference = f"(Based on paragraph(s): {', '.join([f'#{i+1}' for i in indices])})"
    return f"{answer}\n\n{reference}"

def generate_challenge_questions(chunks):
    questions = []
    for i, chunk in enumerate(chunks[:5]):
        if len(questions) >= 3:
            break
        result = summarizer("stick strictly to the document uploaded and don't generate anything out of that document.\n\n" + chunk[:1024], max_length=50, min_length=25, do_sample=False)
        summary = result[0]["summary_text"]
        question = f"Q{i+1}: What does the document say about — {summary.lower()}?"
        questions.append(question)
    return "\n".join(questions)

def grade_answer(context, question, user_answer):
    expected = qa_pipeline(question=question, context=context)["answer"]
    correct_words = set(expected.lower().split())
    user_words = set(user_answer.lower().split())
    match = correct_words & user_words
    score = len(match) / max(len(correct_words), 1)
    justification = f"The answer was deduced based on the document context:\n\n{context[:300]}"
    if score >= 0.6:
        return f"✅ Correct (Score: {score:.2f})\nExpected: {expected}\n\n{justification}"
    else:
        return f"❌ Incorrect (Score: {score:.2f})\nExpected: {expected}\n\n{justification}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📘 DocuMitra (Smart Assistant for Research Summarization)")

    with gr.Row():
        file_input = gr.File(label="📚 Upload PDF or TXT")
        summary_output = gr.Textbox(label="📌 Summary (≤150 words)", lines=6)

    with gr.Row():
        question_input = gr.Textbox(label="💬 Ask Anything Mode")
        ask_button = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer with Source Reference", lines=4)

    # Challenge Me section
    with gr.Row():
        generate_btn = gr.Button("🧠 Generate Challenge Questions")

    with gr.Column():
        challenge_box = gr.Textbox(label="📝 Challenge Questions (Auto-Generated)", lines=6, interactive=False)
        user_answer = gr.Textbox(label="✍️ Your Answer", placeholder="Type your answer here...", lines=3)
        grade_button = gr.Button("✅ Submit Answer")
        grade_output = gr.Textbox(label="📊 Evaluation with Justification", lines=4, interactive=False)

    state = gr.State()

    def handle_file(file):
        if file is None:
            return "Please upload a file."
        text = extract_text(file.name)
        chunks = chunk_text(text)
        index, _ = build_vector_store(chunks)
        state.value = (chunks, index)
        return summarize_document(text)

    def handle_question(q):
        chunks, index = state.value
        return ask_question(q, chunks, index)

    def handle_generate():
        chunks, _ = state.value
        return generate_challenge_questions(chunks)

    def handle_grade(user_answer, challenge_q):
        chunks, index = state.value
        context = "\n".join(chunks[:5])
        return grade_answer(context, challenge_q, user_answer)

    file_input.change(fn=handle_file, inputs=file_input, outputs=summary_output)
    ask_button.click(fn=handle_question, inputs=question_input, outputs=answer_output)
    generate_btn.click(fn=handle_generate, outputs=challenge_box)
    grade_button.click(fn=handle_grade, inputs=[user_answer, challenge_box], outputs=grade_output)

demo.launch()
