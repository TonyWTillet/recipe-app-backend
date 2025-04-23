from dotenv import load_dotenv
import openai
import os

load_dotenv()

# Remplace ta clé API ici
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(user_input: str) -> str:
    response = openai.Completion.create(
        engine="text-davinci-003",  # Choisis un moteur (ex : GPT-3.5, GPT-4...)
        prompt=user_input,
        max_tokens=150  # Limite la taille de la réponse
    )
    return response.choices[0].text.strip()
