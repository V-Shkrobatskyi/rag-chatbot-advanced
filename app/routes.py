from fastapi import UploadFile, Form, Request, APIRouter
from pydantic import BaseModel
from typing import List, Tuple, Optional

from app.services.indexer import build_faiss_index, search_blocks
from app.services.loader import load_project_from_zip, load_project_from_github
from app.utils import clear_stored_txt, save_blocks_to_txt
from app.llm_utils import ask_llm
from app.state import AppState

router = APIRouter()
Document = Tuple[str, str, Optional[str]]

class Question(BaseModel):
    user_question: str


@router.post("/upload/")
async def upload_project(request: Request, file: UploadFile):
    """Upload project, generate embeddings, summaries, and global summary."""
    state: AppState = request.app.state
    await load_project_from_zip(file, state, debug=False)
    return {
        "status": "project loaded",
        "blocks": len(state.documents),
        "preview_block_summaries": [d[2] for d in state.documents[:5]],
        "global_summary": state.global_summary
    }


@router.post("/upload-debug/")
async def upload_project_debug(request: Request, file: UploadFile):
    """Upload project in debug mode (no summaries, print debug info)."""
    state: AppState = request.app.state
    await load_project_from_zip(file, state, debug=True)
    return {
        "status": "project loaded",
        "blocks": len(state.documents),
        "debug_blocks": min(10, len(state.documents))
    }


@router.post("/upload-github/")
async def upload_github(request: Request, repo_url: str = Form(...)):
    """Download and process a project directly from a GitHub repository URL."""
    state: AppState = request.app.state
    clear_stored_txt()

    try:
        raw_docs = await load_project_from_github(repo_url)
        # Initialize documents: (path, text, summary=None)
        documents: List[Document] = [(path, text, None) for path, text in raw_docs]
    except Exception as e:
        return {"error": str(e)}

    # Build FAISS index
    state.embeddings, state.faiss_index = build_faiss_index(documents)
    state.documents = documents
    save_blocks_to_txt(state.documents)

    # 1. Generate summary for each block
    combined_summary_text = ask_llm(
        "Briefly summarize each block:\n\n" +
        "\n\n".join([b[1] for b in state.documents])
    )
    summary_lines = [line.strip() for line in combined_summary_text.split("\n") if line.strip()]

    # If summary less than documents — extend with empty rows
    if len(summary_lines) < len(state.documents):
        summary_lines += [""] * (len(state.documents) - len(summary_lines))

    # Update documents with summary
    state.documents = [(*doc[:2], summary) for doc, summary in zip(state.documents, summary_lines)]

    # 2. Generate global summary (only unempty summaries)
    summaries = [d[2] for d in state.documents if d[2]]
    state.global_summary = ask_llm(
        "Based on these summaries, create a global summary of the project:\n" +
        "\n".join(summaries)
    )

    return {
        "status": "project loaded from GitHub",
        "blocks": len(state.documents),
        "preview_block_summaries": [d[2] for d in state.documents[:5]],
        "global_summary": state.global_summary
    }


@router.get("/test-read/")
async def test_read(request: Request):
    """Return first few blocks for testing."""
    state: AppState = request.app.state
    if not state.documents:
        return {"error": "No documents loaded"}
    return {"first_blocks": [b[1] for b in state.documents[:5]]}


@router.post("/debug-query/")
async def debug_query(request: Request, q: Question):
    """Search and return selected blocks without generating answer."""
    state: AppState = request.app.state
    if not state.documents or state.faiss_index is None:
        return {"error": "No documents loaded"}

    documents: List[Document] = [
        (path, text, summary if len(doc) > 2 else None)
        for doc in state.documents
        for path, text, *rest in [doc]
        for summary in (rest[0] if rest else None,)
    ]

    selected_blocks = search_blocks(q.user_question, documents, state.faiss_index)

    return {
        "question": q.user_question,
        "selected_blocks": selected_blocks
    }


@router.post("/ask/")
async def ask_question(request: Request, q: Question):
    """Main endpoint to ask a question and get an LLM answer."""
    state: AppState = request.app.state
    if not state.documents or state.faiss_index is None:
        return {"error": "No documents loaded"}

    documents: List[Document] = [
        (path, text, summary if len(doc) > 2 else None)
        for doc in state.documents
        for path, text, *rest in [doc]
        for summary in (rest[0] if rest else None,)
    ]

    # 1. Find relevant blocks
    selected_blocks = search_blocks(q.user_question, documents, state.faiss_index)
    if not selected_blocks:
        return {"question": q.user_question, "answer": "No relevant blocks found for the answer."}

    # 2. Build context for the LLM
    context = f"Global summary:\n{state.global_summary}\n\n"
    context += "\n\n".join(
        [f"Summary: {s}\nText: {t}" for _, t, s in selected_blocks]
    )

    # 3. Generate answer using LLM
    answer = ask_llm(f"Question: {q.user_question}\n\nContext:\n{context}")
    return {"question": q.user_question, "answer": answer}
