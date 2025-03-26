import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.core.logging import logger

class FeedbackManager:
    """
    Service for managing user feedback on generated queries.
    In a real implementation, this would use a database.
    """
    
    def __init__(self):
        # In-memory store for development
        self.feedback_store = {}
    
    async def save_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save feedback for a query."""
        feedback_id = str(uuid.uuid4())
        
        # Add timestamps and IDs
        feedback_entry = {
            "id": feedback_id,
            "timestamp": datetime.now().isoformat(),
            **feedback_data
        }
        
        # Store feedback
        self.feedback_store[feedback_id] = feedback_entry
        logger.info(f"Saved feedback with ID {feedback_id} for query {feedback_data.get('query_id')}")
        
        return feedback_entry
    
    async def save_batch_feedback(self, feedback_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Save multiple feedback items in batch."""
        saved_items = []
        
        for item in feedback_items:
            saved_item = await self.save_feedback(item)
            saved_items.append(saved_item)
        
        logger.info(f"Saved {len(saved_items)} feedback items in batch")
        return saved_items
    
    async def get_feedback(self, query_id: str) -> List[Dict[str, Any]]:
        """Get feedback for a specific query."""
        result = []
        for feedback_id, feedback in self.feedback_store.items():
            if feedback.get("query_id") == query_id:
                result.append(feedback)
        
        return result
    
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get aggregate feedback statistics."""
        total_count = len(self.feedback_store)
        positive_count = 0
        negative_count = 0
        
        for feedback in self.feedback_store.values():
            if feedback.get("feedback_type") == "positive":
                positive_count += 1
            elif feedback.get("feedback_type") == "negative":
                negative_count += 1
        
        return {
            "total_count": total_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_percentage": round(positive_count / total_count * 100, 2) if total_count > 0 else 0,
            "negative_percentage": round(negative_count / total_count * 100, 2) if total_count > 0 else 0
        }