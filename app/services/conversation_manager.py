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

    # ==========================================
    # SHARE MANAGEMENT METHODS
    # ==========================================

    async def create_share(
        self,
        conversation_id: str,
        shared_by: str,
        access_level: str = "view",
        expires_at: Optional[datetime] = None,
        shared_with: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a shareable link for a conversation.

        Args:
            conversation_id: ID of conversation to share
            shared_by: User ID creating the share
            access_level: 'view' or 'edit' (only 'view' supported now)
            expires_at: Optional expiry datetime
            shared_with: Optional specific user ID, None for public link

        Returns:
            {
                "id": int,
                "share_token": str,
                "conversation_id": str,
                "access_level": str,
                "expires_at": datetime,
                "created_at": datetime
            }
        """
        import secrets

        try:
            # Verify user owns the conversation
            conn = await self._get_db_connection()
            try:
                owner_id = await conn.fetchval(
                    "SELECT user_id FROM conversations WHERE id = $1",
                    conversation_id
                )

                if not owner_id:
                    raise ValueError(f"Conversation {conversation_id} not found")

                if owner_id != shared_by:
                    raise ValueError(f"User {shared_by} does not own conversation {conversation_id}")

                # Generate secure token
                share_token = secrets.token_urlsafe(32)

                # Insert share record
                row = await conn.fetchrow(
                    """
                    INSERT INTO shared_conversations
                    (conversation_id, share_token, shared_by, access_level, expires_at, shared_with)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id, conversation_id, share_token, shared_by, access_level,
                              expires_at, created_at, shared_with, is_active
                    """,
                    conversation_id,
                    share_token,
                    shared_by,
                    access_level,
                    expires_at,
                    shared_with
                )

                result = dict(row)
                logger.info(f"Created share for conversation {conversation_id} by {shared_by}")
                return result

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error creating share: {str(e)}", exc_info=True)
            raise

    async def get_share_by_token(
        self,
        share_token: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get share details by token.
        Validates token is active and not expired.

        Returns None if token is invalid, expired, or inactive.
        """
        try:
            conn = await self._get_db_connection()
            try:
                row = await conn.fetchrow(
                    """
                    SELECT
                        sc.*,
                        c.user_id as owner_id,
                        c.title as conversation_title
                    FROM shared_conversations sc
                    INNER JOIN conversations c ON sc.conversation_id = c.id
                    WHERE sc.share_token = $1
                      AND sc.is_active = true
                    """,
                    share_token
                )

                if not row:
                    return None

                share = dict(row)

                # Check if expired
                if share['expires_at'] and share['expires_at'] < datetime.now(share['expires_at'].tzinfo):
                    logger.warning(f"Share token {share_token} has expired")
                    return None

                return share

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error getting share by token: {str(e)}", exc_info=True)
            return None

    async def get_conversation_shares(
        self,
        conversation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all shares for a conversation.
        Used by owner to see who has access.
        """
        try:
            conn = await self._get_db_connection()
            try:
                rows = await conn.fetch(
                    """
                    SELECT
                        id,
                        conversation_id,
                        share_token,
                        shared_by,
                        shared_with,
                        access_level,
                        is_active,
                        created_at,
                        expires_at,
                        access_count,
                        last_accessed_at,
                        metadata
                    FROM shared_conversations
                    WHERE conversation_id = $1
                    ORDER BY created_at DESC
                    """,
                    conversation_id
                )

                return [dict(row) for row in rows]

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error getting conversation shares: {str(e)}", exc_info=True)
            return []

    async def revoke_share(
        self,
        share_id: int,
        user_id: str
    ) -> bool:
        """
        Revoke a share (set is_active = false).
        Only owner can revoke.

        Returns True if revoked, False if not found or unauthorized.
        """
        try:
            conn = await self._get_db_connection()
            try:
                # Verify user owns the conversation
                owner_id = await conn.fetchval(
                    """
                    SELECT c.user_id
                    FROM shared_conversations sc
                    INNER JOIN conversations c ON sc.conversation_id = c.id
                    WHERE sc.id = $1
                    """,
                    share_id
                )

                if not owner_id:
                    logger.warning(f"Share {share_id} not found")
                    return False

                if owner_id != user_id:
                    logger.warning(f"User {user_id} attempted to revoke share {share_id} owned by {owner_id}")
                    return False

                # Revoke the share
                result = await conn.execute(
                    """
                    UPDATE shared_conversations
                    SET is_active = false
                    WHERE id = $1
                    """,
                    share_id
                )

                logger.info(f"Revoked share {share_id}")
                return True

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error revoking share: {str(e)}", exc_info=True)
            return False

    async def check_conversation_access(
        self,
        conversation_id: str,
        user_id: str,
        share_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check user's access level to a conversation.

        Returns:
            {
                "has_access": bool,
                "access_level": "owner" | "view" | "edit" | None,
                "can_view": bool,
                "can_edit": bool,
                "can_add_messages": bool,
                "is_owner": bool,
                "error": Optional[str]
            }
        """
        try:
            conn = await self._get_db_connection()
            try:
                # Check if user is the owner
                owner_id = await conn.fetchval(
                    "SELECT user_id FROM conversations WHERE id = $1",
                    conversation_id
                )

                if not owner_id:
                    return {
                        "has_access": False,
                        "access_level": None,
                        "can_view": False,
                        "can_edit": False,
                        "can_add_messages": False,
                        "is_owner": False,
                        "error": "Conversation not found"
                    }

                if owner_id == user_id:
                    return {
                        "has_access": True,
                        "access_level": "owner",
                        "can_view": True,
                        "can_edit": True,
                        "can_add_messages": True,
                        "is_owner": True
                    }

                # Check if conversation is shared with this user
                # First try by share_token if provided
                if share_token:
                    share = await conn.fetchrow(
                        """
                        SELECT access_level, expires_at, is_active
                        FROM shared_conversations
                        WHERE conversation_id = $1
                          AND share_token = $2
                          AND is_active = true
                        """,
                        conversation_id,
                        share_token
                    )
                else:
                    # Try by user_id (specific share)
                    share = await conn.fetchrow(
                        """
                        SELECT access_level, expires_at, is_active
                        FROM shared_conversations
                        WHERE conversation_id = $1
                          AND (shared_with = $2 OR shared_with IS NULL)
                          AND is_active = true
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        conversation_id,
                        user_id
                    )

                if share:
                    # Check if expired
                    if share['expires_at'] and share['expires_at'] < datetime.now(share['expires_at'].tzinfo):
                        return {
                            "has_access": False,
                            "access_level": None,
                            "can_view": False,
                            "can_edit": False,
                            "can_add_messages": False,
                            "is_owner": False,
                            "error": "Share link has expired"
                        }

                    access_level = share['access_level']
                    return {
                        "has_access": True,
                        "access_level": access_level,
                        "can_view": True,
                        "can_edit": access_level == "edit",
                        "can_add_messages": access_level == "edit",
                        "is_owner": False
                    }

                # No access
                return {
                    "has_access": False,
                    "access_level": None,
                    "can_view": False,
                    "can_edit": False,
                    "can_add_messages": False,
                    "is_owner": False,
                    "error": "No access to this conversation"
                }

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error checking conversation access: {str(e)}", exc_info=True)
            return {
                "has_access": False,
                "access_level": None,
                "can_view": False,
                "can_edit": False,
                "can_add_messages": False,
                "is_owner": False,
                "error": f"Error checking access: {str(e)}"
            }

    async def get_shared_conversation(
        self,
        share_token: str,
        increment_access_count: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get conversation via share token.
        Validates access and increments tracking.
        """
        try:
            # Get and validate share
            share = await self.get_share_by_token(share_token)

            if not share:
                return None

            # Get the conversation
            conversation = await self.get_conversation(share['conversation_id'])

            if not conversation:
                return None

            # Update access tracking
            if increment_access_count:
                await self.update_share_access_tracking(share_token)

            # Add share info to conversation
            conversation['shared_by'] = share['shared_by']
            conversation['access_level'] = share['access_level']
            conversation['share_info'] = {
                'share_token': share_token,
                'access_level': share['access_level'],
                'shared_by': share['shared_by'],
                'expires_at': share['expires_at'].isoformat() if share['expires_at'] else None,
                'access_count': share['access_count']
            }

            return conversation

        except Exception as e:
            logger.error(f"Error getting shared conversation: {str(e)}", exc_info=True)
            return None

    async def update_share_access_tracking(
        self,
        share_token: str
    ) -> None:
        """
        Update access_count and last_accessed_at for analytics.
        """
        try:
            conn = await self._get_db_connection()
            try:
                await conn.execute(
                    """
                    UPDATE shared_conversations
                    SET access_count = access_count + 1,
                        last_accessed_at = $1
                    WHERE share_token = $2
                    """,
                    datetime.now(),
                    share_token
                )
            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error updating share access tracking: {str(e)}", exc_info=True)

    async def get_user_shared_conversations(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all conversations shared WITH this user.
        Used in conversation list to show "Shared With Me" section.
        """
        try:
            conn = await self._get_db_connection()
            try:
                rows = await conn.fetch(
                    """
                    SELECT
                        c.*,
                        sc.shared_by,
                        sc.access_level,
                        sc.share_token,
                        sc.created_at as shared_at,
                        'shared' as ownership
                    FROM conversations c
                    INNER JOIN shared_conversations sc ON c.id = sc.conversation_id
                    WHERE (sc.shared_with = $1 OR sc.shared_with IS NULL)
                      AND sc.is_active = true
                      AND (sc.expires_at IS NULL OR sc.expires_at > $2)
                    ORDER BY sc.created_at DESC
                    """,
                    user_id,
                    datetime.now()
                )

                conversations = []
                for row in rows:
                    conv = self._parse_json_fields(dict(row), ['messages', 'metadata'])
                    # For listing, we don't need full messages
                    message_count = len(conv.get("messages", []))
                    conv["message_count"] = message_count
                    conv["messages"] = []  # Clear messages for listing
                    conversations.append(conv)

                return conversations

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error getting user shared conversations: {str(e)}", exc_info=True)
            return []