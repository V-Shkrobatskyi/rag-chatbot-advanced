import numpy as np
import faiss
from typing import List, Tuple, Optional

from app.config import TOP_K
from app.embeddings import get_embedding

Document = Tuple[str, str, Optional[str]]


def build_faiss_index(documents: List[Document]):
    """
    Build FAISS index from document blocks.
    Returns embeddings array and FAISS index.
    """
    if not documents:
        return np.array([], dtype=np.float32), None
    embeddings = np.array([get_embedding(d[1]) for d in documents], dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)

    index.add(embeddings)  # type: ignore
    return embeddings, index


def search_blocks(query: str, documents: List[Document], faiss_index, k: int = TOP_K) -> List[Document]:
    """
    Search for the most relevant blocks using FAISS index stored in AppState.
    """
    if not documents or faiss_index is None:
        return []

    embedding = get_embedding(query).astype(np.float32).reshape(1, -1)
    D, I = faiss_index.search(embedding, k)
    idxs = [i for i in I[0] if i >= 0 and i < len(documents)]
    return [documents[i] for i in idxs]
