from fastapi import APIRouter, Body
from typing import Dict, Any
from app.agents.chat_assistant import chat_assistant

router = APIRouter()

@router.post("/")
async def chat_message(payload: Dict[str, Any] = Body(...)):
    """Conversational Assistant agentic RAG chat endpoint."""
    user_msg = payload.get("message", "")
    response = await chat_assistant.process_chat_message(user_msg)
    return response
