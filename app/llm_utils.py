from openai import OpenAI
from mistralai import Mistral
from app.config import LLM_PROVIDER, MODEL_NAME, OPENAI_API_KEY, MISTRAL_API_KEY

# Initialize clients
if LLM_PROVIDER == "openai":
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
elif LLM_PROVIDER == "mistral":
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def ask_llm(prompt: str) -> str:
    """
    Sends a prompt to the selected LLM (OpenAI or Mistral) and returns the text response.
    """
    if LLM_PROVIDER == "openai":
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    elif LLM_PROVIDER == "mistral":
        try:
            # using method chat.complete for getting answer
            chat_response = mistral_client.chat.complete(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            return chat_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Mistral API error: {e}")
            return ""

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
