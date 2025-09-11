from openai import OpenAI
from app.config import OPENAI_API_KEY, MODEL_NAME

client = OpenAI(api_key=OPENAI_API_KEY)

def ask_openai(prompt: str) -> str:
    """
    Sends a prompt to the LLM and returns the text response.
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
