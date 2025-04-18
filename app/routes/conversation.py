import json
import re
from fastapi import APIRouter, HTTPException, Depends, Request
import uuid
from typing import Any, Dict, List



from app.schemas.conversation import Conversation, Message
from app.services.conversation_manager import ConversationManager
from app.core.logging import logger

router = APIRouter()

# In your conversations.py route

@router.post("/conversations")
async def create_conversation(request: Request):
    """
    Create a new conversation.
    """
    try:
        # Get request body
        body = await request.json()
        logger.info(f"Received conversation creation request: {body}")

        user_id = body.get("user_id")
        title = body.get("title")
        metadata = body.get("metadata", {})

        logger.info(f"Creating conversation for user_id: {user_id}, title: {title}")

        conversation_manager = ConversationManager()
        conversation = await conversation_manager.create_conversation(
            user_id=user_id,
            title=title,
            metadata=metadata
        )

        logger.info(f"Successfully created conversation: {conversation}")
        return conversation

    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
        # Log the full stack trace
        import traceback
        logger.error(traceback.format_exc())
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

@router.get("/conversations/verified-info")
async def get_conversations_with_verified_queries():
    try:
        conversation_manager = ConversationManager()
        conversation_ids = await conversation_manager.get_conversations_with_verified_queries()
        
        return {
            "conversation_ids": conversation_ids
        }
        
    except Exception as e:
        logger.error(f"Error in verified queries endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
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
        
        # Ensure metadata is a dictionary (and not a string)
        if isinstance(conversation.get("metadata"), str):
            conversation["metadata"] = {}  # Default to empty dict
        
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
        
        # Ensure metadata is a dictionary (and not a string)
        if isinstance(added_message.get("metadata"), str):
            added_message["metadata"] = {}  # Default to empty dict
        
        # Convert back to a Message model if needed
        if not isinstance(added_message, Message):
            return Message(**added_message)
        
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
        
        # Ensure metadata is a dictionary for each conversation
        for conversation in conversations:
            if isinstance(conversation.get("metadata"), str):
                conversation["metadata"] = {}  # Default to empty dict
        
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
        
        # Ensure metadata is a dictionary (and not a string)
        if isinstance(conversation.get("metadata"), str):
            conversation["metadata"] = {}  # Default to empty dict
        
        return conversation
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update conversation: {str(e)}")

@router.get("/conversations/{conversation_id}/summary")
async def get_conversation_summary(conversation_id: str):
    """
    Get a summary of the conversation.
    """
    try:
        conversation_manager = ConversationManager()
        conversation = await conversation_manager.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Extract messages for context
        messages = conversation.get("messages", [])
        
        # Use the summarize_conversation function
        from app.services.conversation_summarizer import summarize_conversation
        summary = await summarize_conversation(messages)
        
        return {"summary": summary}
        
        
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

# Add to app/routes/conversation.py

@router.get("/conversations/{conversation_id}/title", response_model=dict)
async def get_conversation_title(conversation_id: str):
    """
    Generate or retrieve a title for a conversation.
    """
    try:
        conversation_manager = ConversationManager()
        conversation = await conversation_manager.get_conversation(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # If conversation already has a title, return it
        if conversation.get("title"):
            return {"title": conversation["title"]}
        
        # Extract messages for title generation
        messages = conversation.get("messages", [])
        
        # If no messages, return default title
        if not messages:
            return {"title": "Empty Conversation"}
        
        # Get first user message as default title
        default_title = "Untitled Conversation"
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    default_title = content
                    # Limit title length
                    if len(default_title) > 50:
                        default_title = default_title[:47] + "..."
                    break
        
        # More advanced title generation could go here
        
        # Update conversation with the title
        updated = await conversation_manager.update_conversation(
            conversation_id=conversation_id,
            updates={"title": default_title}
        )
        
        return {"title": default_title}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate title: {str(e)}")

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    try:
        conversation_manager = ConversationManager()
        success = await conversation_manager.delete_conversation(conversation_id)
        if success:
            return {"status": "success"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

