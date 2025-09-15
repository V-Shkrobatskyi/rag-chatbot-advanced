import os, requests, shutil, zipfile, numpy as np
from typing import List, Tuple, Optional
from fastapi import UploadFile

from app.config import MAX_BLOCKS_PER_FILE, ALLOWED_EXTENSIONS, TMP_DIR
from app.embeddings import get_embedding
from app.llm_utils import ask_llm
from app.services.indexer import build_faiss_index
from app.utils import extract_text_blocks, save_blocks_to_txt, clear_stored_txt
from app.state import AppState

Document = Tuple[str, str, Optional[str]]


async def load_project_from_zip(file: UploadFile, state: AppState, debug=False):
    """
    Load project from uploaded ZIP file:
    - Extract blocks
    - Build embeddings and FAISS index
    - Optionally generate summaries and global summary
    """
    clear_stored_txt()

    # Extract text blocks
    raw_docs = await extract_text_blocks(file)
    documents: List[Document] = [(path, text, None) for path, text in raw_docs]

    if debug:
        # Print debug info for first 10 blocks
        print("=== DEBUG: documents & embeddings ===")
        for i, b in enumerate(documents[:10]):
            e = get_embedding(b[1])
            print(f"Block {i} ({len(b[1])} chars): {b[1][:100]}")
            print(f"Embedding sample: {e[:5]}, sum: {np.sum(e)}")

    # Build FAISS index
    state.embeddings, state.faiss_index = build_faiss_index(documents)
    state.documents = documents
    save_blocks_to_txt(state.documents, include_summary=not debug)

    if not debug:
        # Generate summaries for each block using LLM
        combined_text = "\n\n".join([b[1] for b in state.documents])
        combined_summary_text = ask_llm(
            f"Briefly (1-2 sentences) summarize each block "
            f"below:\n\n{combined_text}"
        )
        summary_lines = [
            line.strip() for line in combined_summary_text.split("\n") if line.strip()
        ]

        # If LLM answer less — extend summary with empty rows
        if len(summary_lines) < len(state.documents):
            summary_lines += [""] * (len(state.documents) - len(summary_lines))

        # Add summary to documents
        state.documents = [
            (*doc[:2], summary)  # tuple = (path, text, summary)
            for doc, summary in zip(state.documents, summary_lines)
        ]

        # Generate global project summary safely
        non_empty_summaries = [d[2] for d in state.documents if d[2]]
        state.global_summary = ask_llm(
            f"Based on these short summaries, "
            f"create a global summary of the project:\n"
            + "\n".join(non_empty_summaries)
        )

    return state.documents


def process_file(path, max_blocks=MAX_BLOCKS_PER_FILE):
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
    blocks = [
        (path, text[i : i + chunk_size].strip())
        for i in range(0, len(text), chunk_size)
        if text[i : i + chunk_size].strip()
    ]

    return blocks


async def load_project_from_github(
    repo_url: str, extensions=ALLOWED_EXTENSIONS
) -> List[Tuple[str, str]]:
    """
    Download a GitHub repository as a ZIP file and extract text blocks.
    Supports both `main` and `master` branches.
    """
    zip_urls = [
        repo_url.rstrip("/") + "/archive/refs/heads/main.zip",
        repo_url.rstrip("/") + "/archive/refs/heads/master.zip",
    ]

    tmp_path = os.path.join("tmp", "github_repo.zip")
    os.makedirs("tmp", exist_ok=True)

    for zip_url in zip_urls:
        r = requests.get(zip_url, stream=True)
        if r.status_code == 200:
            with open(tmp_path, "wb") as f:
                f.write(r.content)
            break
    else:
        raise ValueError("Cannot download repo: tried main and master branches")

    # Prepare working TMP_DIR
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Extract the ZIP file
    with zipfile.ZipFile(tmp_path, "r") as zip_ref:
        zip_ref.extractall(TMP_DIR)

    # Collect blocks from allowed files
    blocks = []
    for root, _, files in os.walk(TMP_DIR):
        for f_name in files:
            if f_name.endswith(extensions):
                blocks.extend(process_file(os.path.join(root, f_name)))

    return blocks
