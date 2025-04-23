from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define FastAPI application
app = FastAPI()

# Input request model
class Message(BaseModel):
    user_input: str
    location: str

# Function to generate response with OpenAI
def generate_response(user_input: str, location: str) -> str:
    try:
        # Modify prompt to include location
        prompt = f"{user_input} (Location: {location})"

        # Use the chat completion endpoint
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI Error: {str(e)}"

# FastAPI route to receive message and generate response
@app.post("/chat")
async def chat(message: Message):
    user_input = message.user_input
    location = message.location
    response = generate_response(user_input, location)
    return {"response": response}
