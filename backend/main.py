import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from google import genai

# Setup logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="FastAPI + Gemini Chatbot", version="1.0.0")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

chat_history: List[str] = []

@app.get("/")
def root():
    logging.info("Root endpoint called")
    return {"message": "FastAPI Gemini chatbot is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_msg = req.message
    logging.info(f"User message: {user_msg}")
    chat_history.append(f"user: {user_msg}")

    try:
        chat_session = client.chats.create(model="gemini-2.0-flash")
        resp = chat_session.send_message(user_msg)
        reply_text = resp.text

        logging.info(f"Gemini reply: {reply_text}")
        chat_history.append(f"bot: {reply_text}")

        return ChatResponse(reply=reply_text)

    except Exception as e:
        logging.error(f"Gemini API error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

@app.get("/history")
def get_history():
    logging.info("History endpoint called")
    return {"history": chat_history}
