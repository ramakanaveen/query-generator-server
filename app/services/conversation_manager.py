# app/services/conversation_manager.py
import uuid
import json
from datetime import datetime , date
from typing import List, Dict, Any, Optional
import asyncpg

from app.core.config import settings
from app.core.logging import logger

# Add a custom JSON encoder class
class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that can serialize datetime objects."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)
    
class ConversationManager:
    """
    Service for managing conversations using a PostgreSQL database.
    """
    
    def __init__(self):
        self.db_url = settings.DATABASE_URL
    
    async def _get_db_connection(self):
        """Get a database connection."""
        return await asyncpg.connect(self.db_url)
    
    async def create_conversation(self, user_id: str, title: str = None, metadata: Dict = None) -> Dict[str, Any]:
        try:
            conversation_id = str(uuid.uuid4())
            
            conn = await self._get_db_connection()
            try:
                # Insert new conversation with required user_id
                await conn.execute(
                    """
                    INSERT INTO conversations 
                    (id, user_id, title, messages, created_at, updated_at, last_accessed_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    conversation_id,
                    user_id,  # Make sure this is passed through
                    title,
                    '[]',  # Empty JSON array for messages
                    datetime.now(),
                    datetime.now(),
                    datetime.now(),
                    json.dumps(metadata or {})
                )
                
                # Get the created conversation
                row = await conn.fetchrow(
                    "SELECT * FROM conversations WHERE id = $1",
                    conversation_id
                )
                
                # Convert to dict
                conversation = dict(row)
                conversation["messages"] = json.loads(conversation.get("messages", "[]"))
                
                return conversation
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
            raise
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        try:
            conn = await self._get_db_connection()
            try:
                # Get conversation
                conversation_row = await conn.fetchrow(
                    "SELECT * FROM conversations WHERE id = $1",
                    conversation_id
                )
                
                if not conversation_row:
                    return None
                
                # Convert to dict
                conversation = dict(conversation_row)
                
                # Parse JSONB fields to Python objects
                conversation = self._parse_json_fields(conversation, ['messages', 'metadata'])
                # Update last_accessed_at
                await conn.execute(
                    """
                    UPDATE conversations 
                    SET last_accessed_at = $1 
                    WHERE id = $2
                    """,
                    datetime.now(),
                    conversation_id
                )
                
                return conversation
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting conversation: {str(e)}", exc_info=True)
            return None
    
    async def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations."""
        try:
            conn = await self._get_db_connection()
            try:
                rows = await conn.fetch(
                    """
                    SELECT * FROM conversations 
                    ORDER BY last_accessed_at DESC
                    """
                )
                
                conversations = []
                for row in rows:
                    conversation = self._parse_json_fields(dict(row), ['messages', 'metadata'])
                    conversations.append(conversation)
                
                return conversations
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting conversations: {str(e)}", exc_info=True)
            return []
    
    async def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a specific user."""
        try:
            conn = await self._get_db_connection()
            try:
                rows = await conn.fetch(
                    """
                    SELECT * FROM conversations 
                    WHERE user_id = $1 
                    ORDER BY last_accessed_at DESC
                    """,
                    user_id
                )
                
                conversations = []
                for row in rows:
                    conversation = self._parse_json_fields(dict(row), ['messages', 'metadata'])
                    # For listing, we don't need full messages
                    message_count = len(conversation.get("messages", []))
                    conversation["message_count"] = message_count
                    conversation["messages"] = []  # Clear messages for listing
                    
                    conversations.append(conversation)
                
                return conversations
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error getting user conversations: {str(e)}", exc_info=True)
            return []
    # Updating the _parse_json_fields method in the ConversationManager class

    def _parse_json_fields(self, data, fields):
        """Parse JSON fields to Python objects."""
        result = data.copy()
        for field in fields:
            if field in result:
                # Default values based on field type
                default_value = [] if field == 'messages' else {}
                
                # Handle string JSON
                if isinstance(result[field], str):
                    try:
                        result[field] = json.loads(result[field])
                    except json.JSONDecodeError:
                        result[field] = default_value
                # Handle None values
                elif result[field] is None:
                    result[field] = default_value
                # Ensure metadata is always a dict, not a string representation of empty dict
                elif field == 'metadata' and result[field] == '{}':
                    result[field] = {}
        
        return result
    async def add_message(self, conversation_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Add a message to a conversation."""
        try:
            conn = await self._get_db_connection()
            try:
                # Check if conversation exists
                conversation_row = await conn.fetchrow(
                    "SELECT * FROM conversations WHERE id = $1",
                    conversation_id
                )
                
                if not conversation_row:
                    # Auto-create the conversation if it doesn't exist
                    await conn.execute(
                        """
                        INSERT INTO conversations 
                        (id, created_at, updated_at, last_accessed_at)
                        VALUES ($1, $2, $3, $4)
                        """,
                        conversation_id,
                        datetime.now(),
                        datetime.now(),
                        datetime.now()
                    )
                    
                    # Get the newly created conversation
                    conversation_row = await conn.fetchrow(
                        "SELECT * FROM conversations WHERE id = $1",
                        conversation_id
                    )
                    
                    logger.info(f"Auto-created conversation {conversation_id} for new message")
                
                # Convert to dict
                conversation = dict(conversation_row)
                
                # Get existing messages
                try:
                    messages = json.loads(conversation.get("messages", "[]"))
                except:
                    messages = []
                
                # Convert message to dict if it's a Pydantic model
                if hasattr(message, "model_dump"):
                    message_dict = message.model_dump()
                elif hasattr(message, "dict"):
                    message_dict = message.dict()
                else:
                    message_dict = dict(message)  # Assume it's dict-like
                
                # Ensure message has an ID
                if "id" not in message_dict:
                    message_dict["id"] = str(uuid.uuid4())
                
                # Ensure message has a timestamp
                if "timestamp" not in message_dict:
                    message_dict["timestamp"] = datetime.now().isoformat()
                
                # Add new message
                messages.append(message_dict)
                
                # Update conversation with new messages
                await conn.execute(
                    """
                    UPDATE conversations 
                    SET messages = $1, updated_at = $2, last_accessed_at = $2 
                    WHERE id = $3
                    """,
                    json.dumps(messages,cls=DateTimeEncoder), 
                    datetime.now(),
                    conversation_id
                )
                
                return message_dict
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error adding message: {str(e)}", exc_info=True)
            raise ValueError(f"Error adding message: {str(e)}")
    
    async def update_conversation(self, conversation_id: str, 
                               updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update conversation metadata."""
        try:
            conn = await self._get_db_connection()
            try:
                # Check if conversation exists
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM conversations WHERE id = $1)",
                    conversation_id
                )
                
                if not exists:
                    return None
                
                # Build update query dynamically
                set_clauses = []
                params = [conversation_id]  # First param is always conversation_id
                param_idx = 2  # PostgreSQL uses $1, $2, etc.
                
                # Only update allowed fields
                allowed_fields = ["title", "is_archived", "summary", "user_id"]
                for field in allowed_fields:
                    if field in updates:
                        set_clauses.append(f"{field} = ${param_idx}")
                        params.append(updates[field])
                        param_idx += 1
                
                # Always update these fields
                set_clauses.append(f"updated_at = ${param_idx}")
                params.append(datetime.now())
                param_idx += 1
                
                set_clauses.append(f"last_accessed_at = ${param_idx}")
                params.append(datetime.now())
                
                # Build and execute query
                if set_clauses:
                    query = f"""
                    UPDATE conversations 
                    SET {", ".join(set_clauses)} 
                    WHERE id = $1
                    RETURNING *
                    """
                    
                    row = await conn.fetchrow(query, *params)
                    
                    if row:
                        conversation = self._parse_json_fields(dict(row), ['messages', 'metadata'])
                        return conversation
                
                return None
            finally:
                await conn.close()
        except Exception as e:
            logger.error(f"Error updating conversation: {str(e)}", exc_info=True)
            return None
    
    async def get_conversation_context(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from a conversation for context."""
        try:
            conversation = await self.get_conversation(conversation_id)
            if not conversation:
                return []
            
            messages = conversation.get("messages", [])
            return messages[-limit:] if messages else []
        except Exception as e:
            logger.error(f"Error getting conversation context: {str(e)}", exc_info=True)
            return []
    
    async def update_conversation_summary(self, conversation_id: str, summary: str) -> bool:
        """Generate and update the summary for a conversation."""
        return await self.update_conversation(
            conversation_id=conversation_id,
            updates={"summary": summary}
        ) is not None
    async def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: ID of the conversation to delete
            
        Returns:
            bool: True if deletion was successful, False if conversation not found
        """
        try:
            conn = await self._get_db_connection()
            try:
                # Check if conversation exists
                exists = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM conversations WHERE id = $1)",
                    conversation_id
                )
                
                if not exists:
                    logger.warning(f"Attempted to delete non-existent conversation: {conversation_id}")
                    return False
                
                # Delete the conversation
                await conn.execute(
                    "DELETE FROM conversations WHERE id = $1",
                    conversation_id
                )
                
                logger.info(f"Successfully deleted conversation: {conversation_id}")
                return True
                
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to delete conversation: {str(e)}")