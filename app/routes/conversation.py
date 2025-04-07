from fastapi import APIRouter, HTTPException, Depends
import uuid
from typing import Any, Dict, List

from app.schemas.conversation import Conversation, Message
from app.services.conversation_manager import ConversationManager

router = APIRouter()

@router.post("/conversations", response_model=Conversation)
async def create_conversation():
    """
    Create a new conversation.
    """
    try:
        conversation_manager = ConversationManager()
        conversation = await conversation_manager.create_conversation()
        return conversation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@router.get("/conversations", response_model=List[Conversation])
async def list_conversations():
    """
    List all conversations.
    """
    try:
        conversation_manager = ConversationManager()
        conversations = await conversation_manager.get_conversations()
        return conversations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str):
    """
    Get a specific conversation by ID.
    """
    try:
        conversation_manager = ConversationManager()
        conversation = await conversation_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")

@router.post("/conversations/{conversation_id}/messages", response_model=Message)
async def add_message(conversation_id: str, message: Message):
    """
    Add a message to a conversation.
    """
    try:
        conversation_manager = ConversationManager()
        added_message = await conversation_manager.add_message(conversation_id, message)
        return added_message
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

# app/routes/conversation.py - Update this file

@router.get("/user/{user_id}/conversations", response_model=List[Conversation])
async def list_user_conversations(user_id: str):
    """
    List all conversations for a specific user.
    """
    try:
        conversation_manager = ConversationManager()
        conversations = await conversation_manager.get_user_conversations(user_id)
        return conversations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

@router.put("/conversations/{conversation_id}", response_model=Conversation)
async def update_conversation(conversation_id: str, updates: Dict[str, Any]):
    """
    Update a conversation's metadata (title, archived status).
    """
    try:
        conversation_manager = ConversationManager()
        conversation = await conversation_manager.update_conversation(conversation_id, updates)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return conversation
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update conversation: {str(e)}")