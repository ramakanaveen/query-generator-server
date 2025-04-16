# app/services/conversation_summarizer.py

import re
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate
from app.services.llm_provider import LLMProvider
from app.core.logging import logger
from app.services.query_generation.prompts.summarizer_prompts import CONVERSATION_SUMMARY_PROMPT_TEMPLATE

async def summarize_conversation(conversation_history):
    """Generate a specialized summary focusing on schema context and query patterns."""
    try:
        # Format messages for the LLM
        formatted_messages = []
        for msg in conversation_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_messages.append(f"{role.capitalize()}: {content}")
        
        conversation_text = "\n".join(formatted_messages)
        
        # Initialize LLM
        llm_provider = LLMProvider()
        llm = llm_provider.get_model("gemini")
        
        # Generate summary with a schema and query-focused prompt
        prompt = ChatPromptTemplate.from_template(CONVERSATION_SUMMARY_PROMPT_TEMPLATE)
        # Create a chain
        chain = prompt | llm
        # Get LLM response
        response = await chain.ainvoke({"conversation_text": conversation_text})
        summary = response.content.strip()
        
        return summary
        
    except Exception as e:
        logger.error(f"Error summarizing conversation: {str(e)}", exc_info=True)
        # Extract basic info if LLM fails
        directives = []
        tables = []
        
        for msg in conversation_history:
            content = msg.get("content", "")
            # Extract directives
            directive_matches = re.findall(r'@([A-Z]+)', content)
            for directive in directive_matches:
                if directive not in directives:
                    directives.append(directive)
            
            # Extract tables
            if msg.get("role") == "assistant":
                table_matches = re.findall(r'from\s+(\w+)', content.lower())
                for table in table_matches:
                    if table not in tables:
                        tables.append(table)
        
        fallback_summary = f"Directives used: {', '.join(directives) if directives else 'none'}. "
        fallback_summary += f"Tables queried: {', '.join(tables) if tables else 'none'}."
        return fallback_summary