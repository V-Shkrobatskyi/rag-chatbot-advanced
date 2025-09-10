from sentence_transformers import SentenceTransformer
import numpy as np

# local free model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True).astype('float32')
