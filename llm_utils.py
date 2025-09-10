from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def ask_openai(prompt: str) -> str:
    """
    Надсилає prompt у LLM і повертає текстову відповідь.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # або інша доступна модель
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()
