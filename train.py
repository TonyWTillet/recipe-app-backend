import pandas as pd
from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import os

# === 1. Charger et préparer les données ===
def load_and_split_data(file_path, chunk_size=10000):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["title", "NER"])  # "NER" = ingrédients, "title" = titre

    def format_row(row):
        input_text = f"Titre: {row['title']}\nIngrédients: {row['NER']}"
        output_text = "Prépare une recette à partir de ces ingrédients."
        return {"input": input_text, "output": output_text}

    formatted_data = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        formatted_data_chunk = [format_row(row) for _, row in chunk.iterrows()]
        formatted_data.append(formatted_data_chunk)
    
    return formatted_data

# === 2. Initialisation du modèle et tokenizer ===
def load_model_and_tokenizer(model_name="google/mt5-small"):
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

# === 3. Prétraiter les données ===
def preprocess_data(chunk, tokenizer, max_input_length=512, max_target_length=256):
    def preprocess(examples):
        inputs = [str(x) for x in examples["input"]]
        targets = [str(x) for x in examples["output"]]
        
        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding="max_length"
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                truncation=True,
                padding="max_length"
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset = Dataset.from_list(chunk)
    return dataset.map(preprocess, batched=True)


# === 4. Configuration de l'entraînement ===
def setup_training_args(output_dir, logging_dir):
    return TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=500,  # Sauvegarde fréquente
        logging_steps=100,
        fp16=False  # Mets True si tu as un GPU compatible
    )

# === 5. Lancer l'entraînement sur un chunk ===
def train_model_on_chunk(chunk, model, tokenizer, training_args, output_dir, resume=False):
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=chunk,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    checkpoint = output_dir if resume else None
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

# === 6. Fonction principale de fractionnement et d'entraînement ===
def train_in_chunks(file_path, chunk_size=10000, model_name="google/mt5-small"):
    tokenizer, model = load_model_and_tokenizer(model_name)
    chunks = load_and_split_data(file_path, chunk_size)
    output_dir = "./mt5-recipes"
    os.makedirs(output_dir, exist_ok=True)

    for i, chunk in enumerate(chunks):
        print(f"\n--- Training on chunk {i+1}/{len(chunks)} ---")

        tokenized_dataset = preprocess_data(chunk, tokenizer)
        logging_dir = os.path.join(output_dir, f"logs_chunk_{i+1}")
        training_args = setup_training_args(output_dir, logging_dir)
        resume = os.path.isdir(output_dir) and any("checkpoint" in d for d in os.listdir(output_dir))

        train_model_on_chunk(tokenized_dataset, model, tokenizer, training_args, output_dir, resume)

# === 7. Lancer ===
if __name__ == "__main__":
    train_in_chunks("dataset/recipe_nlg.csv", chunk_size=10000)
