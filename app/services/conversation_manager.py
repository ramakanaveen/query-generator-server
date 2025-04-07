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
    
    async def create_conversation(self, user_id=None, title=None) -> Dict[str, Any]:
        """Create a new conversation with optional user ID and title."""
        conversation_id = str(uuid.uuid4())
        conversation = {
            "id": conversation_id,
            "user_id": user_id,
            "title": title,
            "messages": [],
            "summary": "",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "last_accessed_at": datetime.now(),
            "is_archived": False,
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
    
    async def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a specific user."""
        return [
            conv for conv in self.conversations.values()
            if conv.get("user_id") == user_id
        ]
    async def update_conversation(self, conversation_id: str, 
                                 updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update conversation metadata (title, archived status, etc.)."""
        if conversation_id not in self.conversations:
            return None
            
        # Update allowed fields
        for field in ["title", "is_archived"]:
            if field in updates:
                self.conversations[conversation_id][field] = updates[field]
                
        # Always update last_accessed_at when touching a conversation
        self.conversations[conversation_id]["last_accessed_at"] = datetime.now()
        self.conversations[conversation_id]["updated_at"] = datetime.now()
        
        return self.conversations[conversation_id]
    
    async def update_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """Generate and update the summary for a conversation."""
        if conversation_id not in self.conversations:
            return None
            
        conversation = self.conversations[conversation_id]
        messages = conversation["messages"]
        
        # If fewer than 3 messages, don't summarize yet
        if len(messages) < 3:
            return ""
            
        # Use conversation_summarizer service to generate summary
        from app.services.conversation_summarizer import summarize_conversation
        summary = await summarize_conversation(messages)
        
        # Update the conversation with the new summary
        conversation["summary"] = summary
        return summary
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