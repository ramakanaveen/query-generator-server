import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

class ConversationManager:
    """
    Service for managing conversations.
    In a real implementation, this would use a database.
    """
    
    def __init__(self):
        # In-memory store for development
        self.conversations = {}
    
    async def create_conversation(self) -> Dict[str, Any]:
        """Create a new conversation."""
        conversation_id = str(uuid.uuid4())
        conversation = {
            "id": conversation_id,
            "messages": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metadata": {}
        }
        self.conversations[conversation_id] = conversation
        return conversation
    
    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    async def get_conversations(self) -> List[Dict[str, Any]]:
        """Get all conversations."""
        return list(self.conversations.values())
    
    async def add_message(self, conversation_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """Add a message to a conversation."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        # Ensure message has an ID
        if "id" not in message:
            message["id"] = str(uuid.uuid4())
        
        # Ensure message has a timestamp
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()
        
        # Add message to conversation
        self.conversations[conversation_id]["messages"].append(message)
        self.conversations[conversation_id]["updated_at"] = datetime.now()
        
        return message
    
    async def get_conversation_context(self, conversation_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent messages from a conversation for context."""
        if conversation_id not in self.conversations:
            return []
        
        messages = self.conversations[conversation_id]["messages"]
        return messages[-limit:] if messages else []