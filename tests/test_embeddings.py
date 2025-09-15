import numpy as np
import app.embeddings as embeddings
from typing import Any


# Simple mock function returning np.ndarray
def mock_encode(*args, **kwargs) -> np.ndarray:
    """Mocked encode function returning a fixed vector, ignores all args."""
    return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def test_get_embedding_shape(monkeypatch: Any) -> None:
    """Check that get_embedding returns a vector of the correct shape."""

    # Replace model.encode with the mock function
    monkeypatch.setattr(embeddings.model, "encode", mock_encode)

    text: str = "Test"
    embedding: np.ndarray = embeddings.get_embedding(text)

    # Type check
    assert isinstance(embedding, np.ndarray)
    # Length check
    assert len(embedding) == 3
    # Value check
    assert embedding[0] == 0.1
