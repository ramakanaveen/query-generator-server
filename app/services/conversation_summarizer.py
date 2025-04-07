# app/services/conversation_summarizer.py

from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from app.services.llm_provider import LLMProvider
from app.core.logging import logger
from app.services.query_generation.prompts.summarizer_prompts import CONVERSATION_SUMMARY_PROMPT_TEMPLATE

async def summarize_conversation(messages: List[Dict[str, Any]]) -> str:
    """
    Generate a concise summary of a conversation.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        A string summary of the conversation
    """
    try:
        # If we have too few messages, don't summarize
        if len(messages) < 3:
            return ""
        
        # Format messages for the LLM
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_messages.append(f"{role.capitalize()}: {content}")
        
        conversation_text = "\n".join(formatted_messages)
        
        # Initialize LLM
        llm_provider = LLMProvider()
        llm = llm_provider.get_model("gemini")  # Use fastest model for summarization
        
        # Generate summary using the prompt from the prompts folder
        chat_prompt = ChatPromptTemplate.from_template(CONVERSATION_SUMMARY_PROMPT_TEMPLATE)
        chain = chat_prompt | llm
        
        response = await chain.ainvoke({"conversation_text": conversation_text})
        summary = response.content.strip()
        
        logger.info(f"Generated conversation summary: {summary[:50]}...")
        return summary
        
    except Exception as e:
        logger.error(f"Error summarizing conversation: {str(e)}", exc_info=True)
        # Return empty summary on error to avoid breaking the application
        return ""