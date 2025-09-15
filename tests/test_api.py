import zipfile
from fastapi.testclient import TestClient
from main import app
import app.routes as routes
from typing import Any, List, Tuple, Optional

client: TestClient = TestClient(app)
Document = Tuple[str, str, Optional[str]]


def mock_save_blocks_to_txt(documents, include_summary=True):
    pass  # just mocked, for prevent save files in test folder


def mock_clear_stored_txt():
    pass  # just mocked, for prevent save files in test folder


# /upload/ endpoint mocks & test
async def mock_load(file: Any, state: Any, debug: bool = False) -> None:
    """
    Mock load_project_from_zip to create 1 block.
    Async as original function.
    """
    state.documents = [("file.txt", "Test content", "Summary")]
    state.global_summary = "Global summary"


def test_upload_zip(monkeypatch: Any, tmp_path: Any) -> None:
    """Test uploading a ZIP file to /upload/ endpoint with mocked loader."""

    # Patch loader to avoid real processing
    monkeypatch.setattr(routes, "load_project_from_zip", mock_load)

    # Create ZIP file
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file.txt", "Test content")

    # Send POST request
    with open(zip_path, "rb") as f:
        response = client.post(
            "/upload/", files={"file": ("test.zip", f, "application/zip")}
        )

    # Assertions
    assert response.status_code == 200
    assert response.json()["blocks"] == 1
    assert "project loaded" in response.json()["status"].lower()


# /upload-github/ endpoint mocks & test
async def mock_load_github(repo_url: str) -> List[Tuple[str, str]]:
    """
    Mock load_project_from_github to return 2 test documents (path, text).
    Async as original function.
    """
    return [
        ("file1.txt", "Test content 1"),
        ("file2.txt", "Test content 2"),
    ]


def mock_ask_llm(prompt: str) -> str:
    return "Mocked summary line 1\nMocked summary line 2"


def test_upload_github(monkeypatch: Any) -> None:
    """Test uploading a GitHub repo URL to /upload-github/ endpoint with mocks."""

    # For prevent save files in test folder
    monkeypatch.setattr(routes, "save_blocks_to_txt", mock_save_blocks_to_txt)
    monkeypatch.setattr(routes, "clear_stored_txt", mock_clear_stored_txt)
    # Patch loader and LLM to avoid real network/API calls
    monkeypatch.setattr(routes, "load_project_from_github", mock_load_github)
    monkeypatch.setattr(routes, "ask_llm", mock_ask_llm)

    # Send POST request with a fake GitHub URL
    response = client.post(
        "/upload-github/", data={"repo_url": "https://github.com/fake/repo"}
    )

    # Assertions
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["blocks"] == 2
    assert "project loaded from github" in json_resp["status"].lower()
    # Check that summaries are from mocked LLM
    assert all("Mocked summary" in s for s in json_resp["preview_block_summaries"])
    assert "Mocked summary" in json_resp["global_summary"]


# /ask/ endpoint mocks & test
def mock_search_blocks(
    query: str, documents: List[Document], faiss_index: Any, k: int = 1
) -> List[Document]:
    """Return first document for testing purposes."""
    if documents:
        return [documents[0]]
    return []


def mock_ask_llm_answer(prompt: str) -> str:
    """Return a mocked LLM answer."""
    return "Mocked answer"


def test_ask_endpoint(monkeypatch: Any) -> None:
    """Test /ask/ endpoint with mocked state, search, and LLM."""

    # Prepare state
    state = app.state
    state.documents = [("file.txt", "This is the content of file", "Short summary")]
    state.faiss_index = object()  # fake FAISS index
    state.global_summary = "Global project summary"

    # Patch search_blocks and ask_llm
    monkeypatch.setattr(routes, "search_blocks", mock_search_blocks)
    monkeypatch.setattr(routes, "ask_llm", mock_ask_llm_answer)

    # Send POST request
    response = client.post("/ask/", json={"user_question": "What is this file about?"})

    # Assertions
    assert response.status_code == 200
    json_resp = response.json()
    assert json_resp["question"] == "What is this file about?"
    assert "mocked answer" in json_resp["answer"].lower()
