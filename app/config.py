import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# === General options ===
PROJECT_NAME = "RAG Chatbot Advanced"
TMP_DIR = os.getenv("TMP_DIR", "tmp_uploads")
STORED_BLOCKS_DIR = os.getenv("STORED_BLOCKS_DIR", "stored_blocks")
GLOBAL_SUMMARY_FILE = os.getenv("GLOBAL_SUMMARY_FILE", "global_summary.txt")

# === OpenAI / LLM ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# === Embeddings / FAISS ===
MAX_BLOCKS_PER_FILE = int(os.getenv("MAX_BLOCKS_PER_FILE", 10))
TOP_K = int(os.getenv("TOP_K", 3))

# === Allowed files extensions ===
ALLOWED_EXTENSIONS = tuple(
    os.getenv("ALLOWED_EXTENSIONS", ".py,.txt,.md").split(",")
)
