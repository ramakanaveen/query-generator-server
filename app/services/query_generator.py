# app/services/query_generator.py
from typing import Tuple, List, Dict, Any, Optional
import uuid
from langgraph.graph import StateGraph , END
from pydantic import BaseModel, Field
from app.core.logging import logger
from app.services.query_generation.nodes import (
    query_analyzer,
    schema_retriever,
    query_generator_node,
    query_validator,
    query_refiner
)

class QueryGenerationState(BaseModel):
    """State for the query generation process."""
    query: str = Field(..., description="The original natural language query")
    llm: Any = Field(..., description="The language model to use")
    directives: List[str] = Field(default_factory=list, description="Extracted directives from the query")
    database_type: str = Field(default="kdb", description="Type of database to query")
    entities: List[str] = Field(default_factory=list, description="Extracted entities from the query")
    intent: Optional[str] = Field(default=None, description="Extracted intent from the query")
    schema: Dict[str, Any] = Field(default_factory=dict, description="Retrieved schema information")
    generated_query: Optional[str] = Field(default=None, description="Generated database query")
    validation_result: Optional[bool] = Field(default=None, description="Whether the query is valid")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors if any")
    thinking: List[str] = Field(default_factory=list, description="Thinking process during generation")
    conversation_id: Optional[str] = Field(default=None, description="ID of the conversation for context")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")

class QueryGenerator:
    """
    Service for generating database queries from natural language using LangGraph.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for query generation."""
        # Initialize the workflow with our state
        workflow = StateGraph(QueryGenerationState)
        
        # Add nodes
        workflow.add_node("query_analyzer", query_analyzer.analyze_query)
        workflow.add_node("schema_retriever", schema_retriever.retrieve_schema)
        workflow.add_node("query_generator", query_generator_node.generate_query)
        workflow.add_node("query_validator", query_validator.validate_query)
        workflow.add_node("query_refiner", query_refiner.refine_query)
        
        # Set the entrypoint
        workflow.set_entry_point("query_analyzer")
        
        # Add standard edges
        workflow.add_edge("query_analyzer", "schema_retriever")
        workflow.add_edge("schema_retriever", "query_generator")
        workflow.add_edge("query_generator", "query_validator")
        workflow.add_edge("query_refiner", "query_generator")
        
        # Add conditional edge with a simpler approach
        workflow.add_conditional_edges(
            "query_validator",
            lambda state: "query_refiner" if not state.validation_result else END
        )
        
        # Compile the workflow
        return workflow.compile()
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for the LLM prompt."""
        if not history:
            return ""
            
        formatted = "Previous conversation:\n"
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant generated this query: {content}\n"
        return formatted
    
    async def generate(
        self, 
        query: str, 
        database_type: str = "kdb", 
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, List[str]]:
        """
        Generate a database query from natural language.
        
        Args:
            query: The natural language query
            database_type: Type of database (kdb, sql, etc.)
            conversation_id: Optional conversation ID for context
            conversation_history: Optional conversation history for context
            
        Returns:
            Tuple of (generated_query, thinking_steps)
        """
        try:
            # Initialize the state
            initial_state = QueryGenerationState(
                query=query,
                llm=self.llm,
                database_type=database_type,
                conversation_id=conversation_id,
                conversation_history=conversation_history or []
            )
            
            # Add a thinking step for initialization
            initial_state.thinking.append(f"Received query: {query}")
            if database_type != "kdb":
                initial_state.thinking.append(f"Database type: {database_type}")
                
            # Add thinking step for conversation context
            if conversation_history:
                try:
                    # Ensure conversation history is properly formatted
                    sanitized_history = []
                    for msg in conversation_history:
                        # Convert to dictionary if it's not already
                        msg_dict = msg
                        if not isinstance(msg, dict):
                            try:
                                # Try to convert to dict if it's a model
                                if hasattr(msg, 'dict'):
                                    msg_dict = msg.dict()
                                elif hasattr(msg, 'model_dump'):
                                    msg_dict = msg.model_dump()
                                else:
                                    # Skip invalid messages
                                    logger.warning(f"Skipping invalid message in conversation history: {msg}")
                                    continue
                            except Exception as e:
                                logger.warning(f"Error processing message in conversation history: {e}")
                                continue
                        
                        # Ensure required fields exist
                        if 'role' not in msg_dict or 'content' not in msg_dict:
                            logger.warning(f"Skipping message missing required fields: {msg_dict}")
                            continue
                            
                        sanitized_history.append(msg_dict)
                    
                    initial_state.thinking.append(f"Using conversation history with {len(sanitized_history)} messages")
                    history_text = self._format_conversation_history(sanitized_history)
                    initial_state.thinking.append(f"Context: {history_text}")
                    
                    # Update the state with sanitized history
                    initial_state.conversation_history = sanitized_history
                except Exception as e:
                    logger.error(f"Error processing conversation history: {e}")
                    initial_state.thinking.append(f"Error processing conversation history: {e}")
                    initial_state.conversation_history = []
            
            # Run the workflow
            logger.info(f"Starting query generation for: {query}")
            result = await self.workflow.ainvoke(initial_state)
            
            # Fix: Extract data from the AddableValuesDict object
            # The actual result is a dictionary, so we need to access it correctly
            generated_query = result.get("generated_query", "// No query generated")
            thinking = result.get("thinking", [])
            
            logger.info(f"Generated query: {generated_query}")
            return generated_query, thinking
        
        except Exception as e:
            logger.error(f"Error generating query: {str(e)}", exc_info=True)
            logger.error(f"Error type: {type(e)}")
            # Return a fallback query and the error as thinking
            return f"// Error generating query: {str(e)}", [f"Error: {str(e)}"]