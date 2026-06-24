# 🏛️ RAG Regulasi Indonesia

An AI-powered Q&A system for Indonesian government regulations. Users can ask questions in natural language and get answers grounded directly in official regulatory documents (laws, government regulations, ministerial decrees, etc.), complete with source citations.

> Note: the app interface is in Bahasa Indonesia by design, since the end users (people looking up Indonesian regulations) are Indonesian speakers. This documentation is in English for a broader technical audience.

## Why this project

Indonesian government regulations are typically long PDF documents that are hard to search through quickly. This project builds a **RAG (Retrieval-Augmented Generation)** pipeline that enables semantic search over document content, then generates natural answers grounded in the retrieved context — rather than generic answers from the LLM's general knowledge.

## Demo

![demo interface](docs_assets/demo.png)

## Architecture

```
Regulation PDFs (docs/)
      │
      ▼
┌─────────────────────┐
│  Text quality check  │  → detect whether extracted text is readable or garbled
└─────────────────────┘
      │
   ┌──┴──┐
   │     │
native   scanned
 text    PDF
   │        │
   │        ▼
   │   OCR (Tesseract)
   │        │
   └────┬───┘
        ▼
┌─────────────────────┐
│ Chunking & Embedding │  → HuggingFace multilingual-MiniLM
└─────────────────────┘
        ▼
┌─────────────────────┐
│   ChromaDB (vector)  │  → persistent local storage
└─────────────────────┘
        ▼
   User query
        │
        ▼
┌─────────────────────┐
│  Similarity Search   │  → retrieve top-3 most relevant chunks
└─────────────────────┘
        ▼
┌─────────────────────┐
│   Gemini Flash API   │  → generate answer from retrieved context
└─────────────────────┘
        ▼
   Answer + sources
```

## Tech stack & rationale

| Component | Choice | Reason |
|---|---|---|
| RAG orchestration | LlamaIndex | Ready-made abstractions for chunking, indexing, retrieval |
| Vector store | ChromaDB | Lightweight, local persistent storage, no separate server needed |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` | Supports Bahasa Indonesia, lightweight enough for CPU |
| LLM | Gemini Flash (REST API) | Free tier sufficient for low volume, fast response for QA |
| OCR | Tesseract + Poppler | Many official regulation PDFs are scanned images, not native text |
| Interface | Streamlit | Fast to prototype data/AI applications |

## Technical challenges & solutions

This section is written in detail because the debugging process was the most valuable part of building this project.

### 1. Some regulation PDFs are scanned images, not native text

Initial text extraction produced random characters (`♦☻♂↨`, etc.) for several documents. Investigation showed these documents were actually scanned images embedded in the PDF structure (`CCITTFaxDecode`, etc.), not text that could be extracted directly.

**Solution:** the system automatically detects the ratio of "normal-looking words" in the extracted text. If the ratio is too low, the document is reprocessed through Tesseract OCR (converted to per-page images via Poppler, then OCR'd with Indonesian language support).

### 2. Slow loading on every app restart

`VectorStoreIndex.from_documents()` always reprocesses all documents from scratch, even though ChromaDB already has persisted data from a previous run.

**Solution:** check `collection.count()` first to decide whether to build the index from scratch or simply load the existing index (`from_vector_store`).

### 3. Dependency version conflicts (`torch`, `transformers`, `sentence-transformers`)

Upgrading one library triggered `ModuleNotFoundError` in another due to incompatible versions — including a missing `torchvision` dependency that wasn't installed.

**Solution:** pinned specific versions in `requirements.txt` that are confirmed to work together.

### 4. Gemini API isn't always stable (503/timeout errors)

**Solution:** automatic retry with backoff (2s, 4s, ...) up to 3 attempts, with a graceful fallback message to the user if all attempts fail.

## Getting started

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd rag-regulasi-indonesia
pip install -r requirements.txt
```

### 2. Install OCR engine (Windows)

- **Tesseract OCR**: download from [github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki), install it, then download [`ind.traineddata`](https://github.com/tesseract-ocr/tessdata/raw/main/ind.traineddata) and place it in Tesseract's `tessdata` folder.
- **Poppler**: download from [github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases), extract to a fixed location.
- Update the Tesseract and Poppler paths at the top of `app.py` to match your installation.

### 3. Set up your API key

Create a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 4. Add your documents

Place regulation PDF files in the `docs/` folder. The system supports both native-text and scanned PDFs.

### 5. Run

```bash
streamlit run app.py
```

## Project structure

```
rag-regulasi-indonesia/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
├── .env                # API key (not committed)
├── docs/               # Regulation PDF documents
└── chroma_db/          # Vector store (auto-generated, not committed)
```

## Current limitations

- Document coverage is limited to regulations collected manually
- OCR significantly increases initial processing time for scanned documents
- No automatic re-indexing mechanism when new documents are added without restarting the app

## Potential improvements

- Automated document sourcing via scraping official JDIH (legal documentation) websites
- Re-ranking retrieval results for higher accuracy
- RAG evaluation metrics to measure answer quality
