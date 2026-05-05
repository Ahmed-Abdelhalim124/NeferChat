import os, io, re, unicodedata
import fitz, cv2, torch, pytesseract, camelot
import numpy as np
from PIL import Image
from tqdm import tqdm
import gradio as gr
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from doclayout_yolo import YOLOv10
from transformers import TableTransformerForObjectDetection, DetrImageProcessor
from groq import Groq
from huggingface_hub import login, hf_hub_download

from utils import clean_text, ocr_image, table_to_markdown, table_to_latex, extract_table_with_camelot

# ---------------- CONFIG ----------------
HF_TOKEN = "your_hf_token_here"
GROQ_API_KEY = "your_groq_api_here"
MODEL_REPO = "juliozhao/DocLayout-YOLO-DocStructBench"
MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
LOCAL_DIR = "./doclayout_models"
IMAGE_DIR = "./pdf_images"
PREVIEW_DIR = "./pdf_previews"
TABLE_DIR = "./tables_vis"
WORKDIR = "./workdir"
os.makedirs(LOCAL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(WORKDIR, exist_ok=True)

client = Groq(api_key=GROQ_API_KEY)

# ---------------- MODEL LOADING ----------------
login(HF_TOKEN)
MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir=LOCAL_DIR)
detector = YOLOv10(MODEL_PATH)
table_processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-structure-recognition")
table_model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition"
).to("cpu")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

print("✅ All models loaded")

# ---------------- GLOBAL ----------------
docs, chunks, bm25, index = [], [], None, None

# ---------------- CHUNKING ----------------
def split_sentences(text):
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        return nltk.sent_tokenize(text)
    except Exception:
        return [text]

def agentic_chunking(docs, max_chars=700, min_chars=80):
    chunks, cid = [], 0
    for d in docs:
        ctype = d.get("type","text").lower()
        content = d.get("content","").strip()
        if not content:
            continue

        if ctype == "table":
            chunks.append({
                "id": f"chunk_{cid}",
                "page": d["page"],
                "type": "table",
                "content": content
            })
            cid += 1
            continue

        if ctype in ["text", "paragraph", "figure", "image"]:
            sents = split_sentences(content)
            buf, L = [], 0
            for s in sents:
                if L + len(s) > max_chars and buf:
                    chunks.append({
                        "id": f"chunk_{cid}",
                        "page": d["page"],
                        "type": "text",
                        "content": " ".join(buf)
                    })
                    cid += 1
                    buf, L = [], 0
                buf.append(s); L += len(s)
            if buf:
                if L < min_chars and len(chunks) > 0:
                    chunks[-1]["content"] += " " + " ".join(buf)
                else:
                    chunks.append({
                        "id": f"chunk_{cid}",
                        "page": d["page"],
                        "type": "text",
                        "content": " ".join(buf)
                    })
                    cid += 1
            continue

        chunks.append({
            "id": f"chunk_{cid}",
            "page": d["page"],
            "type": ctype,
            "content": content
        })
        cid += 1
    return chunks

# ---------------- RETRIEVAL + QA ----------------
def hybrid_search(query, top_k=5, alpha=0.6):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k*5)
    faiss_hits = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i != -1]

    bm25_scores = bm25.get_scores(query.split())
    bm25_arr = np.array(bm25_scores)
    if bm25_arr.max() - bm25_arr.min() == 0:
        bm25_norm = bm25_arr
    else:
        bm25_norm = (bm25_arr - bm25_arr.min()) / (bm25_arr.max() - bm25_arr.min() + 1e-9)

    combined = {}
    for i, s in faiss_hits:
        combined[i] = combined.get(i, 0) + alpha * s
    for idx, val in enumerate(bm25_norm):
        combined[idx] = combined.get(idx, 0) + (1-alpha) * val

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        results.append({
            "page": chunks[idx]["page"],
            "type": chunks[idx]["type"],
            "score": float(score),
            "content": chunks[idx]["content"]
        })
    return results

def groq_answer(question, top_k=8, model="openai/gpt-oss-120b"):
    results = hybrid_search(question, top_k=top_k)
    context = "\n".join([f"- {r['content']}" for r in results])
    prompt = f"""You are a medical guideline assistant.
Use ONLY the context provided.

Question:
{question}

Context:
{context}

Answer format:
### 🩺 Guideline-based Answer
- Provide explicit recommendations or facts.
- End with a 📌 plain-language summary."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content
    return answer, results

# ---------------- PIPELINE ----------------
def process_pdf(pdf_file, progress=gr.Progress()):
    global docs, chunks, bm25, index
    pdf_path = pdf_file if isinstance(pdf_file, str) else pdf_file.name
    print(f"📄 Processing PDF: {pdf_path}")

    from utils import extract_with_yolo_and_table
    docs = extract_with_yolo_and_table(pdf_path, detector, max_pages=None, dpi=200, table_conf=0.5)

    progress(0.4, desc="Chunking into passages...")
    chunks = agentic_chunking(docs, max_chars=700, min_chars=80)

    progress(0.7, desc="Building FAISS + BM25 index...")
    texts = [c["content"] for c in chunks]
    bm25 = BM25Okapi([t.split() for t in texts])
    embs = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    dim = embs.shape[1]
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    progress(1, desc="Done!")
    return (
        f"✅ PDF processed with {len(docs)} blocks and {len(chunks)} chunks.",
        gr.update(interactive=True, variant="primary"),
        gr.update(value="")
    )

def chatbot_interface(question):
    if not docs or not chunks or index is None or bm25 is None:
        return "⚠️ Please upload and preprocess a PDF first.", ""
    try:
        answer, refs = groq_answer(question, top_k=5)
        refs_text = "\n".join(
            [f"- Page {r['page']} | {r.get('type','?')} | score={r['score']:.3f}" for r in refs]
        )
        return answer + "\n\n📚 **References:**\n" + refs_text, ""
    except Exception as e:
        return f"⚠️ Error: {str(e)}", ""

# ---------------- UI ----------------
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Nefer Guideline Chatbot")
    pdf_input = gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"])
    status_box = gr.Textbox(label="Status", interactive=False)
    question = gr.Textbox(lines=2, placeholder="Ask a question about the guideline...")
    ask_btn = gr.Button("Ask", interactive=False, variant="secondary")
    answer_box = gr.Markdown()
    pdf_input.upload(fn=process_pdf, inputs=pdf_input, outputs=[status_box, ask_btn, answer_box])
    ask_btn.click(fn=chatbot_interface, inputs=question, outputs=[answer_box, question])

demo.launch()
