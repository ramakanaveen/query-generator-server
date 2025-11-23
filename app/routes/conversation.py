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
async def get_conversation(conversation_id: str, request: Request):
    """
    Get a specific conversation by ID.

    Optional query params for access control:
    - user_id: User requesting access
    - share_token: Share token for shared conversations

    If neither is provided, access is granted (backward compatibility).
    """
    try:
        conversation_manager = ConversationManager()

        # Get optional access control params
        user_id = request.query_params.get("user_id")
        share_token = request.query_params.get("share_token")

        # If access control params provided, check permissions
        if user_id or share_token:
            access_info = await conversation_manager.check_conversation_access(
                conversation_id=conversation_id,
                user_id=user_id or "anonymous",
                share_token=share_token
            )

            if not access_info["can_view"]:
                raise HTTPException(
                    status_code=403,
                    detail=access_info.get("error", "Access denied")
                )

        # Get conversation
        conversation = await conversation_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Ensure metadata is a dictionary (and not a string)
        if isinstance(conversation.get("metadata"), str):
            conversation["metadata"] = {}  # Default to empty dict

        # Add access info if checked
        if user_id or share_token:
            conversation["access_info"] = access_info

        return conversation

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")

@router.post("/conversations/{conversation_id}/messages", response_model=Message)
async def add_message(conversation_id: str, message: Message, request: Request):
    """
    Add a message to a conversation.

    Optional query params for access control:
    - user_id: User adding the message

    If user_id is provided, checks if user has permission to add messages.
    Only owners or users with 'edit' access can add messages.
    """
    try:
        conversation_manager = ConversationManager()

        # Get optional user_id for access control
        user_id = request.query_params.get("user_id")

        # If user_id provided, check permissions
        if user_id:
            access_info = await conversation_manager.check_conversation_access(
                conversation_id=conversation_id,
                user_id=user_id
            )

            if not access_info["can_add_messages"]:
                raise HTTPException(
                    status_code=403,
                    detail=f"You don't have permission to add messages. Access level: {access_info.get('access_level', 'none')}"
                )

        # Add message
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
    except HTTPException:
        raise
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


# ==========================================
# SHARE MANAGEMENT ENDPOINTS
# ==========================================

@router.post("/conversations/{conversation_id}/share")
async def create_conversation_share(
    conversation_id: str,
    request: Request
):
    """
    Create a shareable link for a conversation.
    Only owner can create shares.

    Request Body:
    {
        "user_id": "user-A-id",  # Required: who is creating the share
        "access_level": "view",  # optional, defaults to "view"
        "expires_at": "2025-12-31T23:59:59Z",  # optional
        "shared_with": "user-123"  # optional, specific user
    }

    Response:
    {
        "success": true,
        "share_token": "abc123xyz",
        "share_url": "http://localhost:8000/api/shared/abc123xyz",
        "conversation_id": "conv-123",
        "access_level": "view",
        "expires_at": "2025-12-31T23:59:59Z"
    }
    """
    try:
        body = await request.json()
        user_id = body.get("user_id")

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")

        access_level = body.get("access_level", "view")
        expires_at_str = body.get("expires_at")
        shared_with = body.get("shared_with")

        # Parse expires_at if provided
        expires_at = None
        if expires_at_str:
            from datetime import datetime
            expires_at = datetime.fromisoformat(expires_at_str.replace('Z', '+00:00'))

        conversation_manager = ConversationManager()

        # Create the share
        share = await conversation_manager.create_share(
            conversation_id=conversation_id,
            shared_by=user_id,
            access_level=access_level,
            expires_at=expires_at,
            shared_with=shared_with
        )

        # Build share URL
        share_url = f"http://localhost:8000/api/shared/{share['share_token']}"

        return {
            "success": True,
            "share_token": share['share_token'],
            "share_url": share_url,
            "conversation_id": share['conversation_id'],
            "access_level": share['access_level'],
            "expires_at": share['expires_at'].isoformat() if share.get('expires_at') else None,
            "created_at": share['created_at'].isoformat() if share.get('created_at') else None
        }

    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating share: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create share: {str(e)}")


@router.get("/shared/{share_token}")
async def get_shared_conversation(share_token: str):
    """
    Access a shared conversation via token.
    Validates token and returns conversation with access info.

    Response:
    {
        "id": "conv-123",
        "title": "Trading Queries",
        "messages": [...],
        "shared_by": "user-A-id",
        "access_info": {
            "has_access": true,
            "access_level": "view",
            "can_view": true,
            "can_edit": false,
            "can_add_messages": false,
            "is_owner": false
        },
        "created_at": "...",
        "updated_at": "..."
    }
    """
    try:
        conversation_manager = ConversationManager()

        # Get conversation via share token
        conversation = await conversation_manager.get_shared_conversation(
            share_token=share_token,
            increment_access_count=True
        )

        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Shared conversation not found or link is invalid/expired"
            )

        # Add access info for frontend
        conversation['access_info'] = {
            "has_access": True,
            "access_level": conversation.get('access_level', 'view'),
            "can_view": True,
            "can_edit": conversation.get('access_level') == 'edit',
            "can_add_messages": conversation.get('access_level') == 'edit',
            "is_owner": False
        }

        return conversation

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error accessing shared conversation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to access shared conversation: {str(e)}")


@router.get("/conversations/{conversation_id}/shares")
async def list_conversation_shares(
    conversation_id: str,
    request: Request
):
    """
    List all active shares for a conversation.
    Only owner can see this.

    Query params:
    ?user_id=user-A-id  # Required

    Response:
    {
        "conversation_id": "conv-123",
        "shares": [
            {
                "id": 1,
                "share_token": "abc123",
                "shared_with": "user-B-id",
                "access_level": "view",
                "created_at": "...",
                "expires_at": "...",
                "access_count": 5,
                "last_accessed_at": "...",
                "is_active": true
            }
        ]
    }
    """
    try:
        # Get user_id from query params
        user_id = request.query_params.get("user_id")

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id query parameter is required")

        conversation_manager = ConversationManager()

        # Verify user owns the conversation
        conversation = await conversation_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation.get('user_id') != user_id:
            raise HTTPException(status_code=403, detail="Only conversation owner can list shares")

        # Get all shares
        shares = await conversation_manager.get_conversation_shares(conversation_id)

        # Format response
        formatted_shares = []
        for share in shares:
            formatted_shares.append({
                "id": share['id'],
                "share_token": share['share_token'],
                "shared_with": share.get('shared_with'),
                "access_level": share['access_level'],
                "created_at": share['created_at'].isoformat() if share.get('created_at') else None,
                "expires_at": share['expires_at'].isoformat() if share.get('expires_at') else None,
                "access_count": share.get('access_count', 0),
                "last_accessed_at": share['last_accessed_at'].isoformat() if share.get('last_accessed_at') else None,
                "is_active": share.get('is_active', True)
            })

        return {
            "conversation_id": conversation_id,
            "shares": formatted_shares
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing shares: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list shares: {str(e)}")


@router.delete("/conversations/{conversation_id}/shares/{share_id}")
async def revoke_conversation_share(
    conversation_id: str,
    share_id: int,
    request: Request
):
    """
    Revoke a share (set is_active = false).
    Only owner can revoke.

    Query params:
    ?user_id=user-A-id  # Required

    Response:
    {
        "success": true,
        "message": "Share revoked successfully"
    }
    """
    try:
        # Get user_id from query params
        user_id = request.query_params.get("user_id")

        if not user_id:
            raise HTTPException(status_code=400, detail="user_id query parameter is required")

        conversation_manager = ConversationManager()

        # Revoke the share
        success = await conversation_manager.revoke_share(share_id, user_id)

        if not success:
            raise HTTPException(
                status_code=403,
                detail="Share not found or you don't have permission to revoke it"
            )

        return {
            "success": True,
            "message": "Share revoked successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking share: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to revoke share: {str(e)}")


@router.get("/user/{user_id}/conversations/all")
async def list_all_user_conversations(user_id: str, include_shared: bool = True):
    """
    Get both owned and shared conversations for a user.

    Response:
    {
        "owned": [
            {
                "id": "conv-123",
                "title": "My Query",
                "user_id": "user-A-id",
                "ownership": "owner",
                "shared_with_count": 2,
                "message_count": 10,
                "created_at": "...",
                "updated_at": "..."
            }
        ],
        "shared_with_me": [
            {
                "id": "conv-456",
                "title": "Team Query",
                "user_id": "user-B-id",
                "ownership": "shared",
                "shared_by": "user-B-id",
                "access_level": "view",
                "share_token": "xyz789",
                "shared_at": "...",
                "message_count": 5
            }
        ]
    }
    """
    try:
        conversation_manager = ConversationManager()

        # Get owned conversations
        owned_conversations = await conversation_manager.get_user_conversations(user_id)

        # For each owned conversation, get share count
        for conv in owned_conversations:
            shares = await conversation_manager.get_conversation_shares(conv['id'])
            active_shares = [s for s in shares if s.get('is_active', True)]
            conv['shared_with_count'] = len(active_shares)
            conv['ownership'] = 'owner'

        # Get shared conversations if requested
        shared_conversations = []
        if include_shared:
            shared_conversations = await conversation_manager.get_user_shared_conversations(user_id)

        return {
            "owned": owned_conversations,
            "shared_with_me": shared_conversations
        }

    except Exception as e:
        logger.error(f"Error listing all user conversations: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list conversations: {str(e)}")

