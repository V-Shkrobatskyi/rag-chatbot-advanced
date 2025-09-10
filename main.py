from fastapi import FastAPI, UploadFile
import zipfile, os
from embeddings import get_embedding
import faiss, numpy as np
from pydantic import BaseModel
import shutil
from llm_utils import ask_openai

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="RAG Chatbot Advanced")

# Global variables for project data
documents = []
embeddings = None
faiss_index = None
global_summary = None  # Global summary of the entire project
tmp_dir = os.path.join("tmp_uploads", "working")
max_blocks_per_file = 10

class Question(BaseModel):
    user_question: str


STORED_BLOCKS_DIR = "stored_blocks"
GLOBAL_SUMMARY_FILE = "global_summary.txt"


def clear_stored_txt():
    """Clear stored blocks and temporary directories."""
    shutil.rmtree(STORED_BLOCKS_DIR, ignore_errors=True)
    os.makedirs(STORED_BLOCKS_DIR, exist_ok=True)

    if os.path.exists(GLOBAL_SUMMARY_FILE):
        os.remove(GLOBAL_SUMMARY_FILE)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)


def save_blocks_to_txt(documents, include_summary=True):
    """Save individual text blocks to .txt files."""
    for idx, block in enumerate(documents):
        file_path = os.path.join(STORED_BLOCKS_DIR, f"block_{idx}.txt")
        path, text = block[:2]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Path: {path}\n")
            if include_summary and len(block) > 2:
                f.write(f"Summary: {block[2]}\n")
            f.write(f"Text:\n{text}")


def save_global_summary(summary):
    """Save global summary to a file."""
    with open(GLOBAL_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)


async def extract_text_blocks(file: UploadFile, extensions=('.py', '.txt', '.md')):
    """Extract text blocks from uploaded ZIP file asynchronously."""
    tmp_path = os.path.join("tmp", file.filename)
    # Read file asynchronously
    data = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(data)

    # Extract ZIP contents
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    # Process all files with allowed extensions
    blocks = []
    for root, _, files in os.walk(tmp_dir):
        for f_name in files:
            if f_name.endswith(extensions):
                blocks.extend(process_file(os.path.join(root, f_name)))
    return blocks


def process_file(path, max_blocks=max_blocks_per_file):
    """
    Read a text file and split it into dynamic number of blocks,
    ensuring no more than max_blocks per file.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # Determine chunk size per block
    chunk_size = max(1, len(text) // max_blocks)

    # Create text blocks
    blocks = [(path, text[i:i + chunk_size].strip()) for i in range(0, len(text), chunk_size) if
              text[i:i + chunk_size].strip()]

    return blocks


def build_faiss_index(documents):
    """Create embeddings and build a FAISS index."""
    embeddings = np.array([get_embedding(d[1]) for d in documents], dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return embeddings, index


def search_blocks(query, k=3):
    """Search for relevant blocks using FAISS."""
    if not documents or faiss_index is None:
        return []

    embedding = get_embedding(query).astype(np.float32).reshape(1, -1)
    D, I = faiss_index.search(embedding, k)

    # Ignore invalid indices (-1 or out of range)
    idxs = [i for i in I[0] if i >= 0 and i < len(documents)]
    return [documents[i] for i in idxs]


async def load_project(file: UploadFile, debug=False):
    """
    Load project from uploaded ZIP file:
    - Extract blocks
    - Build embeddings and FAISS index
    - Optionally generate summaries and global summary
    """
    global documents, embeddings, faiss_index, global_summary

    # Reset globals
    documents = []
    embeddings = None
    faiss_index = None
    global_summary = None

    clear_stored_txt()

    # Extract text blocks
    documents = await extract_text_blocks(file)

    if debug:
        # Print debug info for first 10 blocks
        print("=== DEBUG: documents & embeddings ===")
        for i, b in enumerate(documents[:10]):
            e = get_embedding(b[1])
            print(f"Block {i} ({len(b[1])} chars): {b[1][:100]}")
            print(f"Embedding sample: {e[:5]}, sum: {np.sum(e)}")

    # Build FAISS index
    embeddings, faiss_index = build_faiss_index(documents)
    save_blocks_to_txt(documents, include_summary=not debug)

    if not debug:
        # Generate summaries for each block using LLM
        combined_text = "\n\n".join([b[1] for b in documents])
        combined_summary_text = ask_openai(
            f"Briefly (1-2 sentences) summarize each block below:\n\n{combined_text}"
        )
        summary_lines = [line.strip() for line in combined_summary_text.split("\n") if line.strip()]

        # Add summary to documents
        documents = [
            (*doc, summary)  # tuple = (path, text, summary)
            for doc, summary in zip(documents, summary_lines)
        ]

        # Generate global project summary from block summaries
        global_summary = ask_openai(
            f"Based on these short summaries, create a global summary of the project:\n" +
            "\n".join(d[2] for d in documents)
        )

    return documents


@app.post("/upload/")
async def upload_project(file: UploadFile):
    """Upload project, generate embeddings, summaries, and global summary."""
    await load_project(file, debug=False)
    return {
        "status": "project loaded",
        "blocks": len(documents),
        "preview_block_summaries": [d[2] for d in documents[:5]],
        "global_summary": global_summary
    }


@app.post("/upload-debug/")
async def upload_project_debug(file: UploadFile):
    """Upload project in debug mode (no summaries, print debug info)."""
    await load_project(file, debug=True)
    return {
        "status": "project loaded",
        "blocks": len(documents),
        "debug_blocks": min(10, len(documents))
    }


@app.get("/test-read/")
async def test_read():
    """Return first few blocks for testing."""
    if not documents:
        return {"error": "No documents loaded"}
    return {"first_blocks": [b[1] for b in documents[:5]]}


@app.post("/debug-query/")
async def debug_query(q: Question):
    """Search and return selected blocks without generating answer."""
    if not documents or faiss_index is None:
        return {"error": "No documents loaded"}

    selected_blocks = search_blocks(q.user_question)

    return {
        "question": q.user_question,
        "selected_blocks": selected_blocks
    }


@app.post("/ask/")
async def ask_question(q: Question):
    """Main endpoint to ask a question and get an LLM answer."""
    if not documents or faiss_index is None:
        return {"error": "No documents loaded"}

    # 1. Find relevant blocks
    selected_blocks = search_blocks(q.user_question)
    if not selected_blocks:
        return {"question": q.user_question, "answer": "No relevant blocks found for the answer."}

    # 2. Build context for the LLM
    context = f"Global summary:\n{global_summary}\n\n"
    context += "\n\n".join(
        [f"Summary: {s}\nText: {t}" for _, t, s in selected_blocks]
    )

    # 3. Generate answer using LLM
    answer = ask_openai(f"Question: {q.user_question}\n\nContext:\n{context}")
    return {"question": q.user_question, "answer": answer}


@app.on_event("startup")
async def startup_event():
    """Initialize globals on app startup."""
    global documents, embeddings, faiss_index, global_summary
    documents = []
    embeddings = None
    faiss_index = None
    global_summary = None
    clear_stored_txt()
