import zipfile, os, shutil, requests
from fastapi import UploadFile

from config import (
    STORED_BLOCKS_DIR,
    GLOBAL_SUMMARY_FILE,
    TMP_DIR,
    MAX_BLOCKS_PER_FILE,
    ALLOWED_EXTENSIONS,
)


async def extract_text_blocks(file: UploadFile, extensions=ALLOWED_EXTENSIONS):
    """Extract text blocks from uploaded ZIP file asynchronously."""
    tmp_path = os.path.join("tmp", file.filename)
    # Read file asynchronously
    data = await file.read()
    with open(tmp_path, "wb") as f:
        f.write(data)

    # Extract ZIP contents
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        zip_ref.extractall(TMP_DIR)

    # Process all files with allowed extensions
    blocks = []
    for root, _, files in os.walk(TMP_DIR):
        for f_name in files:
            if f_name.endswith(extensions):
                blocks.extend(process_file(os.path.join(root, f_name)))
    return blocks


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
    blocks = [(path, text[i:i + chunk_size].strip()) for i in range(0, len(text), chunk_size) if
              text[i:i + chunk_size].strip()]

    return blocks


def clear_stored_txt():
    """Clear stored blocks and temporary directories."""
    shutil.rmtree(STORED_BLOCKS_DIR, ignore_errors=True)
    os.makedirs(STORED_BLOCKS_DIR, exist_ok=True)

    if os.path.exists(GLOBAL_SUMMARY_FILE):
        os.remove(GLOBAL_SUMMARY_FILE)

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    os.makedirs(TMP_DIR, exist_ok=True)


async def extract_github_repo(repo_url: str, extensions=ALLOWED_EXTENSIONS):
    """
    Download a GitHub repository as a ZIP file and extract text blocks.
    Supports both `main` and `master` branches.
    """
    zip_urls = [
        repo_url.rstrip("/") + "/archive/refs/heads/main.zip",
        repo_url.rstrip("/") + "/archive/refs/heads/master.zip"
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
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        zip_ref.extractall(TMP_DIR)

    # Collect blocks from allowed files
    blocks = []
    for root, _, files in os.walk(TMP_DIR):
        for f_name in files:
            if f_name.endswith(extensions):
                blocks.extend(process_file(os.path.join(root, f_name)))

    return blocks
