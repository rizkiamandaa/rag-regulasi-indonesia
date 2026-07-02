import torch.nn as nn
import sys
sys.modules['nn'] = nn

import os
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Document,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import chromadb
import pytesseract
from pdf2image import convert_from_path

# ── OCR Configuration ────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\Release-26.02.0-0\poppler-26.02.0\Library\bin"

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Use local embedding — no API key needed ──────
Settings.embed_model = HuggingFaceEmbedding(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
# Turn off LLM default LlamaIndex
Settings.llm = None


# ── Detect whether the PDF text is from a scan or original ──
def is_text_garbage(text: str) -> bool:
    """Cek apakah teks hasil extract terlihat seperti garbage/scan."""
    if len(text.strip()) == 0:
        return True
    words = text.split()
    if len(words) == 0:
        return True
    readable_words = [
        w for w in words
        if len(w) > 0 and sum(c.isalpha() for c in w) / len(w) > 0.6
    ]
    ratio = len(readable_words) / len(words)
    return ratio < 0.4  # kurang dari 40% kata yang "terlihat normal" → garbage


def ocr_pdf(filepath: str) -> str:
    """Ekstrak teks dari PDF hasil scan menggunakan OCR."""
    pages = convert_from_path(filepath, poppler_path=POPPLER_PATH)
    full_text = []
    for page in pages:
        text = pytesseract.image_to_string(page, lang="ind")
        full_text.append(text)
    return "\n\n".join(full_text)


def load_documents_with_ocr_fallback(docs_folder: str):
    """Load semua PDF; kalau terdeteksi scan, proses ulang pakai OCR."""
    docs = SimpleDirectoryReader(docs_folder).load_data()

    # First, group by filename (a single file can be split into multiple documents / pages)
    by_file = {}
    for d in docs:
        fname = d.metadata.get("file_name", "unknown")
        by_file.setdefault(fname, []).append(d)

    final_docs = []
    for fname, parts in by_file.items():
        combined_text = "\n".join(p.text for p in parts)

        if is_text_garbage(combined_text):
            filepath = os.path.join(docs_folder, fname)
            ocr_text = ocr_pdf(filepath)
            final_docs.append(
                Document(text=ocr_text, metadata={"file_name": fname, "ocr": True})
            )
        else:
            for p in parts:
                final_docs.append(p)

    return final_docs


# ── Gemini via REST API, with automation retry ────────
def ask_gemini(prompt: str, max_retries: int = 3) -> str:
    url = (
        "https://generativelanguage.googleapis.com"
        "/v1beta/models/gemini-flash-latest:generateContent"
        f"?key={GEMINI_API_KEY}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            last_error = e
            # 503/timeout = server is busy, Try again after a brief pause.
            if attempt < max_retries:
                time.sleep(2 * attempt)  # 2s, 4s, ...
                continue
            raise last_error


# ── Load documents and create index (or load an existing index) ──
@st.cache_resource(show_spinner="Memuat dokumen regulasi (PDF hasil scan akan di-OCR, mohon tunggu)...")
def load_index():
    db = chromadb.PersistentClient(path="./chroma_db")
    col = db.get_or_create_collection("regulasi")
    store = ChromaVectorStore(chroma_collection=col)
    ctx = StorageContext.from_defaults(vector_store=store)

    if col.count() > 0:
        # The index has already been created—it loads directly and is not reprocessed
        index = VectorStoreIndex.from_vector_store(
            vector_store=store, storage_context=ctx
        )
    else:
        # Initial run — process all PDFs, using OCR for scanned documents
        docs = load_documents_with_ocr_fallback("docs")
        index = VectorStoreIndex.from_documents(docs, storage_context=ctx)

    return index, col.count()


# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="RAG Regulasi Indonesia",
    page_icon="🏛️",
    layout="centered",
)

# ── Custom styling ───────────────────────────────────────
st.markdown(
    """
    <style>
    .source-card {
        background-color: rgba(120, 120, 120, 0.08);
        border-left: 3px solid #4a90d9;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
    .source-card-ocr {
        border-left: 3px solid #d9a14a;
    }
    .relevance-badge {
        display: inline-block;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 10px;
        background-color: rgba(74, 144, 217, 0.15);
        margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar — document info ──────────────────────────────
with st.sidebar:
    st.markdown("### 🏛️ RAG Regulasi Indonesia")
    st.caption("Tanya jawab regulasi Indonesia berbasis AI")
    st.divider()

    docs_folder = "docs"
    if os.path.exists(docs_folder):
        pdf_files = [f for f in os.listdir(docs_folder) if f.endswith(".pdf")]
        st.markdown(f"**📄 Dokumen termuat:** {len(pdf_files)}")
        with st.expander("Lihat daftar dokumen"):
            for f in pdf_files:
                st.write(f"- {f}")
    else:
        st.warning("Folder 'docs' tidak ditemukan")

    st.divider()
    st.markdown("**ℹ️ Cara kerja sistem**")
    st.caption(
        "Setiap dokumen otomatis dicek: jika teksnya tidak terbaca normal "
        "(misal hasil scan), sistem akan menjalankan OCR sebelum diproses."
    )

    st.divider()
    if st.button("🗑️ Hapus riwayat chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Model: Gemini Flash · Embedding: multilingual-MiniLM · OCR: Tesseract")

def render_sources(nodes_or_sources, is_node=True):
    """Render daftar sumber dokumen dengan styling card."""
    for i, item in enumerate(nodes_or_sources):
        if is_node:
            file_name = item.metadata.get("file_name", "Unknown")
            is_ocr = item.metadata.get("ocr", False)
            score = item.score
            snippet = item.text[:350] + "..."
        else:
            file_name = item["file"]
            is_ocr = item.get("ocr", False)
            score = item["score"]
            snippet = item["text"]

        card_class = "source-card source-card-ocr" if is_ocr else "source-card"
        ocr_label = " 🔍 *(hasil OCR)*" if is_ocr else ""

        st.markdown(
            f"""
            <div class="{card_class}">
                <b>Sumber {i + 1}</b> — <i>{file_name}</i>{ocr_label}
                <span class="relevance-badge">relevansi: {score:.3f}</span>
                <p style="margin-top:8px; font-size:0.9rem; opacity:0.85;">{snippet}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Inisialisasi index & session state ──────────────────
index, n_chunks = load_index()
retriever = index.as_retriever(similarity_top_k=3)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Header ───────────────────────────────────────────────
st.title("🏛️ RAG Regulasi Indonesia")
st.caption("Tanyakan apapun tentang regulasi yang sudah dimuat")

# ── Contoh pertanyaan (tampil hanya kalau belum ada chat) ──
example_query = None
if len(st.session_state.messages) == 0:
    st.markdown("**💡 Contoh pertanyaan:**")
    col1, col2 = st.columns(2)
    examples = [
        "Apa itu informasi elektronik?",
        "Apa kewajiban penyelenggara sistem elektronik?",
        "Apa sanksi melanggar UU ITE?",
        "Apa itu tanda tangan elektronik?",
    ]
    for i, ex in enumerate(examples):
        col = col1 if i % 2 == 0 else col2
        if col.button(ex, use_container_width=True, key=f"example_{i}"):
            example_query = ex

# ── Tampilkan riwayat chat ───────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Lihat sumber dokumen"):
                render_sources(msg["sources"], is_node=False)

# ── Input chat baru ──────────────────────────────────────
query = st.chat_input("Tulis pertanyaan kamu di sini...") or example_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            nodes = retriever.retrieve(query)
            context = "\n\n".join([n.text for n in nodes])

            prompt = f"""Berdasarkan dokumen berikut, jawab pertanyaan ini.

Dokumen:
{context}

Pertanyaan: {query}

Jawab dalam Bahasa Indonesia yang jelas dan informatif."""

            try:
                answer = ask_gemini(prompt)
            except Exception:
                answer = (
                    "⚠️ Server Gemini sedang sibuk atau lambat merespons. "
                    "Coba tanyakan lagi dalam beberapa saat."
                )
            st.markdown(answer)

            sources = []
            with st.expander("📚 Lihat sumber dokumen"):
                render_sources(nodes, is_node=True)
                for node in nodes:
                    file_name = node.metadata.get("file_name", "Unknown")
                    sources.append(
                        {
                            "file": file_name,
                            "score": node.score,
                            "text": node.text[:350] + "...",
                            "ocr": node.metadata.get("ocr", False),
                        }
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
