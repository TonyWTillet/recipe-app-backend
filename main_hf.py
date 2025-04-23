from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
from typing import Optional
import time
import json

load_dotenv()
app = FastAPI()

# Get Hugging Face token
HUGGINGFACE_API_TOKEN = os.getenv("HF_API_TOKEN")

if not HUGGINGFACE_API_TOKEN:
    raise ValueError("HF_API_TOKEN not found in environment variables")

# Request model
class UserMessage(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str = ""  # Default empty string
    error: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
def chat(user_message: UserMessage):
    if not user_message.user_input.strip():
        raise HTTPException(status_code=400, detail="Input message cannot be empty")

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    # Using a better model for French text generation
    API_URL = "https://api-inference.huggingface.co/models/gpt2-large"

    try:
        # Format the prompt for the model
        formatted_prompt = f"""Tu es un chef cuisinier expert. Voici la recette demandée :

{user_message.user_input}

Voici la recette détaillée :"""

        # Send the request to the model
        response = requests.post(
            API_URL, 
            headers=headers, 
            json={
                "inputs": formatted_prompt,
                "parameters": {
                    "max_length": 500,
                    "temperature": 0.9,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            },
            timeout=30
        )

        # Check for HTTP errors
        if response.status_code != 200:
            error_msg = f"API request failed with status code {response.status_code}"
            try:
                error_details = response.json()
                if "error" in error_details:
                    error_msg = f"API Error: {error_details['error']}"
            except:
                pass
            return ChatResponse(response="", error=error_msg)

        # Parse the response
        result = response.json()
        if isinstance(result, list):
            result = result[0]
        if "generated_text" in result:
            response_text = result["generated_text"].strip()
            # Clean up the response
            response_text = response_text.replace(formatted_prompt, "").strip()
            return ChatResponse(response=response_text)
        else:
            return ChatResponse(response="", error="Format de réponse inattendu")

    except requests.exceptions.Timeout:
        return ChatResponse(response="", error="La requête a expiré. Veuillez réessayer.")
    except requests.exceptions.RequestException as e:
        return ChatResponse(response="", error=f"Erreur de connexion à l'API: {str(e)}")
    except Exception as e:
        return ChatResponse(response="", error=f"Une erreur inattendue s'est produite: {str(e)}") 