# RAG Chatbot Advanced

A FastAPI-based Retrieval-Augmented Generation (RAG) Chatbot that allows uploading a ZIP archive of project files, processes them into text blocks, generates embeddings, builds a FAISS index, and enables natural language querying with contextual answers using LLMs.

## Features:
* **Upload project ZIPs** containing `.py`, `.txt`, `.md` files.
* **Automatic text block extraction** and splitting into smaller chunks.
* **Embedding generation** for each block using a custom `get_embedding` function.
* **FAISS-based similarity search** for efficient retrieval of relevant blocks.
* **Block-level and global summaries** generated via an LLM.
* **Interactive question answering** using selected relevant blocks and the global project summary as context.
* **Debug mode** to inspect extracted blocks and embeddings without generating summaries.

---

## Installation:

### 1. Clone the repository:
Open the Git Bash console in the directory where you want to place the project and run command:
```
git clone git@github.com:V-Shkrobatskyi/rag-chatbot-advanced.git
```

### 2. Create and activate virtual environment:
Open the project and run command:
```
python -m .venv venv
```

To activate virtualenv:
a) On windows:
```
source .venv\Scripts\activate
```
b) On mac:
```
source .venv/bin/activate
```

### 3. Install dependencies:
```
pip install -r requirements.txt
```

### 4. Create a .env file
Rename `.env.sample` file to `.env`. Open it and add the all variables to it.
For generate OPENAI_API_KEY:
1. Go to https://platform.openai.com/docs/overview and Sign Up
2. Sign In and go to your profile
3. Go to "API keys" tab and press "Create new secret key"
4. Choose "Project = Default project" and "Create secret key"

---

## Usage:
Start the FastAPI server without "--reload" parameter (generating text blocks for income data will restart server):
```
uvicorn main:app
```
The API will be available at http://127.0.0.1:8000.

---

## API Endpoints:

### 1. Upload Project
**POST** `/upload/`

Upload a ZIP file of your project. The server will:
+ Extract text blocks
+ Generate embeddings
+ Build FAISS index
+ Generate block summaries
+ Generate global project summary

Response:
```json
  "status": "project loaded",
  "blocks": 12,
  "preview_block_summaries": ["Summary 1", "Summary 2", "..."],
  "global_summary": "Overall project description"
```

### 2. Upload Project (Debug Mode)
**POST** `/upload-debug/`

Similar to `/upload/` but:
* Does not generate summaries
* Prints debug information for first 10 blocks

### 3. Test Read
**GET** `/test-read/`

Returns the first few extracted text blocks for inspection.
**Response:**
```json
{
  "first_blocks": ["Text of block 1", "Text of block 2", "..."]
}
```

### 4. Debug Query
**POST** `/debug-query/`

Searches relevant blocks for a user question without generating an answer.
**Request Body:**
```json
{
  "user_question": "Explain the main function"
}
```

**Response:**
```json
{
  "question": "Explain the main function",
  "selected_blocks": [
    ["path/to/file.py", "text content", "summary"]
  ]
}
```

### 5. Ask Question
**POST** `/ask/`

Searches relevant blocks and generates a contextual answer using the global project summary.
**Request Body:**
```json
{
  "user_question": "What does the RAG pipeline do?"
}
```

**Response:**
```json
{
  "question": "What does the RAG pipeline do?",
  "answer": "The pipeline splits files into blocks, generates embeddings..."
}
```

---

## Project Structure

```
main.py              # FastAPI application with endpoints
embeddings.py        # Function for generating embeddings
llm_utils.py         # Function to query LLM (e.g., OpenAI API)
tmp_uploads/         # Temporary directory for uploaded files
stored_blocks/       # Directory for storing processed text blocks
global_summary.txt   # Global project summary file
```

---

## Notes

* Maximum of **10 blocks per file** (configurable via `max_blocks_per_file`).
* Supports **.py, .txt, .md** files.
* Uses **FAISS** for fast vector similarity search.
* Requires a **working OpenAI API key** for block summaries and Q\&A.

---

## License

MIT License

---
