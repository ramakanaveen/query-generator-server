# app/services/conversation_manager.py
import uuid
import json
import re
from datetime import datetime , date
from typing import List, Dict, Any, Optional

# UPDATED: Import db_pool instead of asyncpg
from app.core.db import db_pool
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
    Enhanced service for managing conversations with conversation essence support.
    Updated to use DatabasePool for connection management.
    """

    def __init__(self):
        # REMOVED: self.db_url = settings.DATABASE_URL (no longer needed)
        pass

    # UPDATED: Use db_pool instead of direct connection
    async def _get_db_connection(self):
        """Get a database connection from the pool."""
        return await db_pool.get_connection()

    async def create_conversation(self, user_id: str, title: str = None, metadata: Dict = None) -> Dict[str, Any]:
        try:
            conversation_id = str(uuid.uuid4())

            # UPDATED: Use db_pool pattern
            conn = await self._get_db_connection()
            try:
                await conn.execute(
                    """
                    INSERT INTO conversations
                    (id, user_id, title, messages, created_at, updated_at, last_accessed_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    conversation_id,
                    user_id,
                    title,
                    '[]',
                    datetime.now(),
                    datetime.now(),
                    datetime.now(),
                    json.dumps(metadata or {})
                )

                row = await conn.fetchrow(
                    "SELECT * FROM conversations WHERE id = $1",
                    conversation_id
                )

                conversation = dict(row)
                conversation["messages"] = json.loads(conversation.get("messages", "[]"))

                return conversation
            finally:
                # UPDATED: Use db_pool.release_connection instead of conn.close()
                await db_pool.release_connection(conn)
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
            raise

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        try:
            conn = await self._get_db_connection()
            try:
                conversation_row = await conn.fetchrow(
                    "SELECT * FROM conversations WHERE id = $1",
                    conversation_id
                )

                if not conversation_row:
                    return None

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
                await db_pool.release_connection(conn)
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
                await db_pool.release_connection(conn)
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
                await db_pool.release_connection(conn)
        except Exception as e:
            logger.error(f"Error getting user conversations: {str(e)}", exc_info=True)
            return []

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
                    json.dumps(messages, cls=DateTimeEncoder),
                    datetime.now(),
                    conversation_id
                )

                return message_dict
            finally:
                await db_pool.release_connection(conn)
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
                allowed_fields = ["title", "is_archived", "summary", "user_id", "metadata"]
                for field in allowed_fields:
                    if field in updates:
                        if field == "metadata":
                            # Handle metadata specially - ensure it's JSON
                            metadata_value = updates[field]
                            if isinstance(metadata_value, dict):
                                metadata_value = json.dumps(metadata_value, cls=DateTimeEncoder)
                            set_clauses.append(f"{field} = ${param_idx}")
                            params.append(metadata_value)
                        else:
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
                await db_pool.release_connection(conn)
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

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages."""
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
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to delete conversation: {str(e)}")

    async def get_conversations_with_verified_queries(self) -> List[str]:
        """Get list of conversation IDs that have verified queries."""
        try:
            conn = await self._get_db_connection()
            try:
                # Get conversation IDs that have verified queries
                query = """
                        SELECT DISTINCT conversation_id
                        FROM verified_queries
                        WHERE conversation_id IS NOT NULL \
                        """
                results = await conn.fetch(query)

                return [str(row['conversation_id']) for row in results]

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error getting conversations with verified queries: {str(e)}", exc_info=True)
            raise Exception(f"Failed to get conversations with verified queries: {str(e)}")

    # NEW: Conversation Essence Methods (using db_pool pattern)
    async def get_conversation_essence(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation essence from metadata."""
        try:
            conn = await self._get_db_connection()
            try:
                metadata = await conn.fetchval(
                    "SELECT metadata FROM conversations WHERE id = $1",
                    conversation_id
                )

                if not metadata:
                    return {}

                # Parse metadata if it's a string
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                return metadata.get("conversation_essence", {})

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error getting conversation essence: {str(e)}", exc_info=True)
            return {}

    async def update_conversation_essence(self, conversation_id: str, essence_updates: Dict[str, Any]) -> bool:
        """Update conversation essence and generate enhanced summary."""
        try:
            conn = await self._get_db_connection()
            try:
                # Get current metadata
                current_metadata_raw = await conn.fetchval(
                    "SELECT metadata FROM conversations WHERE id = $1",
                    conversation_id
                )

                # Parse current metadata
                current_metadata = {}
                if current_metadata_raw:
                    if isinstance(current_metadata_raw, str):
                        current_metadata = json.loads(current_metadata_raw)
                    else:
                        current_metadata = current_metadata_raw

                # Update conversation essence
                current_metadata["conversation_essence"] = essence_updates

                # Generate enhanced summary from essence
                enhanced_summary = await self._generate_essence_aware_summary(conversation_id, essence_updates)

                # Update both metadata and summary
                await conn.execute(
                    """
                    UPDATE conversations
                    SET metadata = $1, summary = $2, updated_at = $3, last_accessed_at = $3
                    WHERE id = $4
                    """,
                    json.dumps(current_metadata, cls=DateTimeEncoder),
                    enhanced_summary,
                    datetime.now(),
                    conversation_id
                )

                logger.info(f"Updated conversation essence for {conversation_id}")
                return True

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error updating conversation essence: {str(e)}", exc_info=True)
            return False

    async def append_feedback_to_trail(self, conversation_id: str, feedback_entry: Dict[str, Any]) -> bool:
        """Append new feedback to the feedback trail in conversation essence."""
        try:
            # Get current essence
            current_essence = await self.get_conversation_essence(conversation_id)

            # Initialize feedback trail if it doesn't exist
            if "feedback_trail" not in current_essence:
                current_essence["feedback_trail"] = []

            # Add timestamp to feedback entry
            feedback_entry["timestamp"] = datetime.now().isoformat()

            # Append new feedback
            current_essence["feedback_trail"].append(feedback_entry)

            # Keep only last 10 feedback entries to prevent unlimited growth
            current_essence["feedback_trail"] = current_essence["feedback_trail"][-10:]

            # Update essence
            return await self.update_conversation_essence(conversation_id, current_essence)

        except Exception as e:
            logger.error(f"Error appending feedback to trail: {str(e)}", exc_info=True)
            return False

    async def update_original_intent(self, conversation_id: str, intent: str) -> bool:
        """Update the original intent in conversation essence."""
        try:
            current_essence = await self.get_conversation_essence(conversation_id)

            # Only update if not already set (preserve original intent)
            if not current_essence.get("original_intent"):
                current_essence["original_intent"] = intent
                current_essence["created_at"] = datetime.now().isoformat()

                return await self.update_conversation_essence(conversation_id, current_essence)

            return True  # Already set, no update needed

        except Exception as e:
            logger.error(f"Error updating original intent: {str(e)}", exc_info=True)
            return False

    async def update_current_understanding(self, conversation_id: str, understanding: str) -> bool:
        """Update the current system understanding of user needs."""
        try:
            current_essence = await self.get_conversation_essence(conversation_id)
            current_essence["current_understanding"] = understanding
            current_essence["last_updated"] = datetime.now().isoformat()

            return await self.update_conversation_essence(conversation_id, current_essence)

        except Exception as e:
            logger.error(f"Error updating current understanding: {str(e)}", exc_info=True)
            return False

    async def add_key_context(self, conversation_id: str, context_items: List[str]) -> bool:
        """Add key context items (directives, entities, constraints) to conversation essence."""
        try:
            current_essence = await self.get_conversation_essence(conversation_id)

            # Initialize key_context if it doesn't exist
            if "key_context" not in current_essence:
                current_essence["key_context"] = []

            # Add new context items (avoid duplicates)
            for item in context_items:
                if item not in current_essence["key_context"]:
                    current_essence["key_context"].append(item)

            return await self.update_conversation_essence(conversation_id, current_essence)

        except Exception as e:
            logger.error(f"Error adding key context: {str(e)}", exc_info=True)
            return False

    async def _generate_essence_aware_summary(self, conversation_id: str, essence: Dict[str, Any]) -> str:
        """Generate an enhanced summary that incorporates conversation essence."""
        try:
            summary_parts = []

            # Add original intent
            if essence.get("original_intent"):
                summary_parts.append(f"Intent: {essence['original_intent']}")

            # Add current understanding
            if essence.get("current_understanding"):
                summary_parts.append(f"Focus: {essence['current_understanding']}")

            # Add key context
            if essence.get("key_context"):
                context_str = ", ".join(essence["key_context"][:5])  # Limit to 5 items
                summary_parts.append(f"Context: {context_str}")

            # Add feedback trail summary
            feedback_trail = essence.get("feedback_trail", [])
            if feedback_trail:
                correction_count = len(feedback_trail)
                latest_correction = feedback_trail[-1].get("correction", "refinement")
                summary_parts.append(f"Corrections: {correction_count} applied, latest: {latest_correction}")

            # Add status
            if feedback_trail:
                summary_parts.append("Status: Iteratively refined")
            else:
                summary_parts.append("Status: Initial query")

            return " | ".join(summary_parts) if summary_parts else "Active conversation"

        except Exception as e:
            logger.error(f"Error generating essence-aware summary: {str(e)}", exc_info=True)
            # Fallback to simple summary
            return f"Conversation with {len(essence.get('feedback_trail', []))} interactions"

    async def extract_directives_from_history(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract directives from conversation history."""
        directives = []

        for msg in conversation_history:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                directive_matches = re.findall(r'@([A-Z]+)', content)
                for directive in directive_matches:
                    if directive not in directives:
                        directives.append(directive)

        return directives

    async def build_conversation_context_for_retry(self, conversation_id: str) -> Dict[str, Any]:
        """Build comprehensive context for retry requests."""
        try:
            # Get conversation essence
            essence = await self.get_conversation_essence(conversation_id)

            # Get recent conversation history
            conversation = await self.get_conversation(conversation_id)
            messages = conversation.get("messages", []) if conversation else []

            # Extract directives from history
            directives = await self.extract_directives_from_history(messages)

            return {
                "essence": essence,
                "original_intent": essence.get("original_intent"),
                "current_understanding": essence.get("current_understanding"),
                "feedback_trail": essence.get("feedback_trail", []),
                "key_context": essence.get("key_context", []),
                "directives_from_history": directives,
                "recent_messages": messages[-5:],  # Last 5 messages
                "conversation_summary": conversation.get("summary") if conversation else None
            }

        except Exception as e:
            logger.error(f"Error building retry context: {str(e)}", exc_info=True)
            return {
                "essence": {},
                "original_intent": None,
                "current_understanding": None,
                "feedback_trail": [],
                "key_context": [],
                "directives_from_history": [],
                "recent_messages": [],
                "conversation_summary": None
            }