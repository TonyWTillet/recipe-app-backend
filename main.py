from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM, BloomTokenizerFast, BloomForCausalLM
import torch

# === 🔁 Charger les modèles fine-tunés ===
# MT5 model
mt5_model_path = "./mt5-recipes/checkpoint-1000"
mt5_tokenizer = MT5Tokenizer.from_pretrained(mt5_model_path)
mt5_model = MT5ForConditionalGeneration.from_pretrained(mt5_model_path)

# GPT2 model (essayobsmrm8488/gpt2-finetuned-recipes-cooking_v2)
gpt2_tokenizer = AutoTokenizer.from_pretrained("mrm8488/gpt2-finetuned-recipes-cooking_v2")
gpt2_model = AutoModelForCausalLM.from_pretrained("mrm8488/gpt2-finetuned-recipes-cooking_v2")

# BLOOM model
bloom_tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
bloom_model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")

# === 🌐 Contexte pour les idées de recettes ===
cooking_context = """Tu es un chef créatif qui propose des idées de recettes. Pour chaque demande, donne exactement 3 suggestions de recettes différentes.
Format de réponse attendu :
1. [Nom de la recette 1] - [Description courte en une phrase]
2. [Nom de la recette 2] - [Description courte en une phrase]
3. [Nom de la recette 3] - [Description courte en une phrase]"""

# === 🍳 Générateur d'idées de recettes pour MT5 ===
def generate_text_mt5(prompt: str) -> str:
    try:
        prompt_with_context = f"{cooking_context}\n\nDemande : {prompt}\n\nSuggestions :"

        input_ids = mt5_tokenizer.encode(prompt_with_context, return_tensors="pt", truncation=True, max_length=512)

        output_ids = mt5_model.generate(
            input_ids,
            max_length=200,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,  # Activé pour permettre l'utilisation de temperature et top_p
            temperature=0.9,
            top_p=0.95,
            repetition_penalty=1.2,
            min_length=50
        )

        response = mt5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Nettoyer la réponse
        response = response.replace(prompt_with_context, "").strip()
        
        # Vérifier si la réponse contient au moins 2 suggestions
        if not response or len(response.split('\n')) < 2:
            raise ValueError("Pas assez de suggestions générées.")

        return response

    except Exception as e:
        return f"Désolé, une erreur est survenue : {str(e)}. Veuillez réessayer plus tard."

# === 🍳 Générateur d'idées de recettes pour GPT2 ===
def generate_text_gpt2(prompt: str) -> str:
    try:
        formatted_prompt = f"{cooking_context}\n\nDemande : {prompt}\n\nSuggestions :"
        
        inputs = gpt2_tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = gpt2_model.generate(
            **inputs,
            max_length=200,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,  # Activé pour permettre l'utilisation de temperature et top_p
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            min_length=50
        )
        
        generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Nettoyer la réponse
        generated_text = generated_text.replace(formatted_prompt, "").strip()

        # Vérifier si la réponse contient au moins 2 suggestions
        if not generated_text or len(generated_text.split('\n')) < 2:
            raise ValueError("Pas assez de suggestions générées.")

        return generated_text

    except Exception as e:
        return f"Désolé, une erreur est survenue : {str(e)}. Veuillez réessayer plus tard."
    
# === 🍳 Générateur d'idées de recettes pour BLOOM ===
def generate_text_bloom(prompt: str) -> str:
    try:
        formatted_prompt = f"{cooking_context}\n\nDemande : {prompt}\n\nSuggestions :"
        
        inputs = bloom_tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = bloom_model.generate(
            **inputs,
            max_length=500,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=True,
            do_sample=True,  # Activé pour permettre l'utilisation de temperature et top_p
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            min_length=50
        )
        
        generated_text = bloom_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Nettoyer la réponse
        generated_text = generated_text.replace(formatted_prompt, "").strip()

        # Vérifier si la réponse contient au moins 2 suggestions
        if not generated_text or len(generated_text.split('\n')) < 0:
            raise ValueError("Pas assez de suggestions générées.")

        return generated_text

    except Exception as e:
        return f"Désolé, une erreur est survenue : {str(e)}. Veuillez réessayer plus tard."

# === 🚀 FastAPI ===
app = FastAPI()

class Prompt(BaseModel):
    prompt: str
    model: str = "mt5"  # Paramètre pour choisir quel modèle utiliser

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur le backend IA de suggestions de recettes !"}

@app.post("/generate_recipe")
def generate_recipe(prompt: Prompt):
    if prompt.model == "mt5":
        response = generate_text_mt5(prompt.prompt)
    elif prompt.model == "gpt2":
        response = generate_text_gpt2(prompt.prompt)
    elif prompt.model == "bloom":
        response = generate_text_bloom(prompt.prompt)
    else:
        response = "Modèle non reconnu. Veuillez choisir 'mt5', 'gpt2' ou 'bloom'."
    return {"response": response}
