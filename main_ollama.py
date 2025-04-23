from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

# Activer CORS pour ton frontend Vue
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Contexte de cuisine à utiliser pour la requête
cooking_context = """
Tu es un assistant de cuisine expérimenté. Ta mission est d'aider les utilisateurs à trouver des recettes à cuisiner en fonction de leurs critères (ingrédients disponibles, régime alimentaire, type de plat...).

Pour chaque demande, propose exactement 3 recettes différentes que l'utilisateur peut cuisiner. Chaque recette doit respecter les critères fournis.

Format attendu :
1. [Nom de la recette] - [Brève description de la recette]
2. [Nom de la recette] - [Brève description de la recette]
3. [Nom de la recette] - [Brève description de la recette]

"""

def format_response(text: str) -> str:
    emoji_map = {
        "salade": "🥗", "quiche": "🥧", "soupe": "🍲", "riz": "🍚", "pâtes": "🍝",
        "pizza": "🍕", "tarte": "🥧", "viande": "🥩", "poisson": "🐟",
        "légumes": "🥦", "dessert": "🍰", "chocolat": "🍫", "poulet": "🍗",
        "crêpe": "🥞", "burger": "🍔", "oeuf": "🥚", "fruit": "🍓", "gratin": "🧀",
        "curry": "🍛", "tofu": "🧈", "nouilles": "🍜", "miso": "🍥", "pâtes": "🍝",
    }

    lines = text.strip().split("\n")
    formatted_lines = []

    for line in lines:
        if line.strip().startswith(("1.", "2.", "3.")):
            # Séparer numéro, nom et description
            parts = line.split(" - ", 1)
            if len(parts) == 2:
                title, description = parts[0], parts[1]
                # Extraire le numéro et le titre
                num_parts = title.split(".", 1)
                if len(num_parts) == 2:
                    number, title_text = num_parts[0] + ".", num_parts[1].strip()
                    title_lower = title_text.lower()
                    emoji = next((e for k, e in emoji_map.items() if k in title_lower), "🍽️")
                    formatted_lines.append(f"{number} {emoji} {title_text}\n{description}")
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)

    return "\n\n".join(formatted_lines)


# === 📩 Schéma de la requête POST ===
class Prompt(BaseModel):
    prompt: str
    model: str = "mistral"  # Par défaut

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API locale Ollama pour les idées de recettes !"}

@app.post("/generate_recipe")
def generate_recipe(prompt: Prompt):
    try:
        full_prompt = cooking_context + prompt.prompt + "\nRéponse :"

        # Appel à l'API Ollama locale
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": prompt.model,
                "prompt": full_prompt,
                "stream": False
            }
        )

        response.raise_for_status()
        result = response.json()
        final_response = format_response(result.get("response", "").strip())
        return {"response": final_response}

    except Exception as e:
        return {"response": f"Désolé, une erreur est survenue : {str(e)}"}
    

    
