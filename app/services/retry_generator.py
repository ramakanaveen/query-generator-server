from typing import Tuple, List, Dict, Any, Optional
import uuid

from app.core.logging import logger
from app.services.llm_provider import LLMProvider

class RetryGenerator:
    """
    Service for generating improved queries based on feedback.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def _format_conversation_history(self, history):
        """Format conversation history for the prompt."""
        if not history:
            return ""
            
        formatted = "Previous conversation:\n"
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Generated query: {content}\n"
        return formatted
    
    async def generate_improved_query(
        self, 
        original_query: str,
        original_generated_query: str,
        feedback: str,
        database_type: str = "kdb",
        conversation_id: Optional[str] = None,
        conversation_history: List[Dict[str, Any]] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate an improved query based on user feedback.
        
        Args:
            original_query: The original natural language query
            original_generated_query: The previously generated query
            feedback: User feedback about what was wrong
            database_type: Type of database (kdb, sql, etc.)
            conversation_id: Optional conversation ID for context
            conversation_history: Optional conversation history for context
            
        Returns:
            Tuple of (improved_query, thinking_steps)
        """
        try:
            thinking = []
            thinking.append(f"Received request to improve query based on feedback")
            thinking.append(f"Original query: {original_query}")
            thinking.append(f"Original generated query: {original_generated_query}")
            thinking.append(f"User feedback: {feedback}")
            
            # Add context from conversation history
            conversation_context = ""
            if conversation_history:
                thinking.append(f"Using conversation history with {len(conversation_history)} messages")
                conversation_context = self._format_conversation_history(conversation_history)
                thinking.append(f"Conversation context: {conversation_context}")
            
            # Create a prompt for the language model
            prompt_template = """
            You are an expert in {db_type} queries. The user asked:
            
            "{original_query}"
            
            You generated this query:
            ```
            {generated_query}
            ```
            
            But the user provided this feedback:
            "{feedback}"
            
            {conversation_context}
            
            Please create an improved version of the query that addresses the user's feedback.
            
            Important for KDB/Q queries:
            - Use backticks for symbols: `AAPL
            - For dates, use .z.d for today
            - Use xasc/xdesc for sorting: `select ... | xdesc `column`
            - For top N: `select top N ...`
            
            Provide ONLY the improved query, no explanations.
            """
            
            prompt = prompt_template.format(
                db_type=database_type.upper(),
                original_query=original_query,
                generated_query=original_generated_query,
                feedback=feedback,
                conversation_context=conversation_context
            )
            
            thinking.append("Generating improved query based on feedback...")
            
            # Send to language model
            from langchain.prompts import ChatPromptTemplate
            chat_prompt = ChatPromptTemplate.from_template(prompt)
            chain = chat_prompt | self.llm
            
            response = await chain.ainvoke({})
            improved_query = response.content.strip()
            
            thinking.append(f"Generated improved query: {improved_query}")
            
            return improved_query, thinking
        
        except Exception as e:
            logger.error(f"Error generating improved query: {str(e)}", exc_info=True)
            thinking.append(f"Error: {str(e)}")
            return f"// Error generating improved query: {str(e)}", thinking