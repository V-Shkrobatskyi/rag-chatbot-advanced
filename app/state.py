from typing import List, Optional
import numpy as np


class AppState:
    documents: List[tuple] = []
    embeddings: Optional[np.ndarray] = None
    faiss_index: Optional[object] = None
    global_summary: Optional[str] = None
