# app/services/query_generator.py
from typing import Tuple, List, Dict, Any, Optional, Union
import uuid
from langgraph.graph import StateGraph , END
from pydantic import BaseModel, Field
from app.core.logging import logger
from app.services.query_generation.nodes import (
    query_analyzer,
    schema_description_node,
    schema_retriever,
    query_generator_node,
    query_validator,
    query_refiner,
    unified_query_analyzer
)
from app.core.langfuse_client import langfuse_client

class QueryGenerationState(BaseModel):
    """State for the query generation process."""
    query: str = Field(..., description="The original natural language query")
    llm: Any = Field(..., description="The language model to use")
    directives: List[str] = Field(default_factory=list, description="Extracted directives from the query")
    database_type: str = Field(default="kdb", description="Type of database to query")
    entities: List[str] = Field(default_factory=list, description="Extracted entities from the query")
    intent: Optional[str] = Field(default=None, description="Extracted intent from the query")
    intent_type: str = Field(default="query_generation", description="Type of intent detected")
    schema_targets: Dict[str, Any] = Field(default_factory=dict, description="Schema targets for schema description intent")
    help_request: Dict[str, Any] = Field(default_factory=dict, description="Help request details for help intent")
    query_schema: Dict[str, Any] = Field(default_factory=dict, description="Retrieved schema information")
    generated_query: Optional[str] = Field(default=None, description="Generated database query")
    generated_content: Optional[str] = Field(default=None, description="Generated content for non-query intents")
    validation_result: Optional[bool] = Field(default=None, description="Whether the query is valid")
    validation_errors: List[Union[str, Dict[str, Any]]] = Field(default_factory=list, description="Validation errors if any")
    thinking: List[str] = Field(default_factory=list, description="Thinking process during generation")
    conversation_id: Optional[str] = Field(default=None, description="ID of the conversation for context")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")
    no_schema_found: bool = Field(default=False, description="Flag indicating if no relevant schema was found")
    original_query: Optional[str] = Field(default=None, description="Original query before refinement")
    original_errors: List[str] = Field(default_factory=list, description="Errors in the original query")
    refinement_guidance: Optional[str] = Field(default=None, description="Guidance for query refinement")
    refinement_count: int = Field(default=0, description="Number of refinement attempts")
    max_refinements: int = Field(default=2, description="Maximum number of refinement attempts")
    llm_corrected_query: Optional[str] = Field(default=None, description="Corrected query from LLM")
    detailed_feedback: Optional[List[Union[str, Dict[str, Any]]]] = Field(default_factory=list, description="Detailed feedback from the query")
    validation_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validation details")
    use_unified_analyzer: bool = Field(default=False, description="Whether to use the new unified analyzer")
    confidence: str = Field(default="medium", description="Analysis confidence level")
    reasoning: str = Field(default="", description="Analysis reasoning")
    query_complexity: str = Field(default="SINGLE_LINE", description="Query complexity")
    execution_plan: List[str] = Field(default_factory=list, description="Execution plan")
    query_type: str = Field(default="select_basic", description="Query type")

    # NEW: Retry-specific fields
    is_retry_request: bool = Field(default=False, description="Whether this is a retry request")
    original_generated_query: Optional[str] = Field(default=None, description="Previously generated query for retry")
    user_feedback: Optional[str] = Field(default=None, description="User feedback about what was wrong")
    retry_context: Dict[str, Any] = Field(default_factory=dict, description="Additional retry context")

    # NEW: Conversation essence fields
    conversation_essence: Dict[str, Any] = Field(default_factory=dict, description="Conversation essence from database")
    original_intent: Optional[str] = Field(default=None, description="Original user intent from conversation start")
    feedback_trail: List[Dict[str, Any]] = Field(default_factory=list, description="Trail of feedback and corrections")
    current_understanding: Optional[str] = Field(default=None, description="Current system understanding of user needs")
    key_context: List[str] = Field(default_factory=list, description="Key context elements (directives, entities, constraints)")

    # NEW: User context
    user_id: Optional[str] = Field(default=None, description="User ID for personalized context")
    conversation_summary: Optional[str] = Field(default=None, description="Current conversation summary")

# Keep the existing QueryGenerator class structure, just with enhanced state
class QueryGenerator:
    """
    Service for generating database queries from natural language using LangGraph.
    """
    
    def __init__(self, llm,use_unified_analyzer=False):
        self.llm = llm
        self.use_unified_analyzer = use_unified_analyzer
        self.workflow = self._build_workflow()


    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for query generation."""
        # Initialize the workflow with our state
        workflow = StateGraph(QueryGenerationState)
        
        # Add nodes
        # Conditionally add the analyzer based on configuration
        if getattr(self, 'use_unified_analyzer', False):
            workflow.add_node("query_analyzer", unified_query_analyzer.unified_analyze_query)
        else:
            workflow.add_node("query_analyzer", query_analyzer.analyze_query)
        workflow.add_node("schema_retriever", schema_retriever.retrieve_schema)
        workflow.add_node("query_generator", query_generator_node.generate_query)
        workflow.add_node("query_validator", query_validator.validate_query)
        workflow.add_node("query_refiner", query_refiner.refine_query)
        workflow.add_node("schema_description", schema_description_node.generate_schema_description)

        # Set the entrypoint
        workflow.set_entry_point("query_analyzer")

        # Add routing based on intent type
        workflow.add_conditional_edges(
            "query_analyzer",
            lambda state: {
                "query_generation": "schema_retriever",
                "schema_description": "schema_description"
            }.get(state.intent_type, "schema_retriever")
        )
        
        # Add standard edges for query generation path
        workflow.add_edge("schema_retriever", "query_generator")
        workflow.add_edge("query_generator", "query_validator")
        workflow.add_edge("query_refiner", "query_generator")

        # Add schema description path - goes straight to END
        workflow.add_edge("schema_description", END)
        
        # Add conditional edge with a simpler approach
        workflow.add_conditional_edges(
            "query_validator",
            lambda state: END if state.validation_result else (
                END if state.refinement_count >= state.max_refinements else "query_refiner"
            )
        )

        workflow.add_conditional_edges(
            "schema_retriever",
            lambda state: END if state.no_schema_found else "query_generator"
        )
        
        # Compile the workflow
        return workflow.compile()

    # Enhanced generate method to handle conversation essence
    async def generate(
            self,
            query: str,
            database_type: str = "kdb",
            conversation_id: Optional[str] = None,
            conversation_history: Optional[List[Dict[str, Any]]] = None,
            user_id: Optional[str] = None,
            # NEW: Retry-specific parameters
            is_retry: bool = False,
            original_generated_query: Optional[str] = None,
            user_feedback: Optional[str] = None
    ) -> Tuple[Union[str, Dict[str, Any]], List[str]]:
        """
        Generate a database query or other content from natural language.
        Enhanced with conversation essence and retry support.
        """
        try:
            # Initialize the enhanced state
            initial_state = QueryGenerationState(
                query=query,
                llm=self.llm,
                database_type=database_type,
                conversation_id=conversation_id,
                conversation_history=conversation_history or [],
                user_id=user_id,

                # NEW: Retry fields
                is_retry_request=is_retry,
                original_generated_query=original_generated_query,
                user_feedback=user_feedback
            )

            # Add thinking steps
            initial_state.thinking.append(f"Received {'retry' if is_retry else 'initial'} query: {query}")
            if database_type != "kdb":
                initial_state.thinking.append(f"Database type: {database_type}")
            if is_retry:
                initial_state.thinking.append(f"Retry feedback: {user_feedback}")

            # Handle conversation history
            if conversation_history:
                try:
                    sanitized_history = []
                    for msg in conversation_history:
                        msg_dict = msg
                        if not isinstance(msg, dict):
                            try:
                                if hasattr(msg, 'dict'):
                                    msg_dict = msg.dict()
                                elif hasattr(msg, 'model_dump'):
                                    msg_dict = msg.model_dump()
                                else:
                                    logger.warning(f"Skipping invalid message in conversation history: {msg}")
                                    continue
                            except Exception as e:
                                logger.warning(f"Error processing message in conversation history: {e}")
                                continue

                        if 'role' not in msg_dict or 'content' not in msg_dict:
                            logger.warning(f"Skipping message missing required fields: {msg_dict}")
                            continue

                        sanitized_history.append(msg_dict)

                    initial_state.thinking.append(f"Using conversation history with {len(sanitized_history)} messages")
                    initial_state.conversation_history = sanitized_history
                except Exception as e:
                    logger.error(f"Error processing conversation history: {e}")
                    initial_state.thinking.append(f"Error processing conversation history: {e}")
                    initial_state.conversation_history = []

            # Run the enhanced workflow
            logger.info(f"Starting {'retry' if is_retry else 'initial'} processing for query: {query}")
            callbacks = []
            if langfuse_client.get_callback_handler():
                callbacks.append(langfuse_client.get_callback_handler())

            result = await self.workflow.ainvoke(initial_state, config={"callbacks": callbacks})

            # Extract results (same logic as before)
            thinking = result.get("thinking", [])
            intent_type = result.get("intent_type", "query_generation")

            if intent_type == "query_generation":
                if result.get('no_schema_found', False):
                    return {
                        "intent_type": "query_generation",
                        "generated_query": "// I don't know how to generate a query for this request. " +
                                           "I couldn't find any relevant tables in the available schemas. " +
                                           "Please try a different query or upload relevant schemas."
                    }, thinking

                return {
                    "intent_type": "query_generation",
                    "generated_query": result.get("generated_query", "// No query generated")
                }, thinking

            elif intent_type == "schema_description":
                return {
                    "intent_type": "schema_description",
                    "generated_content": result.get("generated_content", "No schema information available.")
                }, thinking

            elif intent_type == "help":
                return {
                    "intent_type": "help",
                    "generated_content": result.get("generated_content", "No help information available.")
                }, thinking

            else:
                logger.warning(f"Unknown intent type: {intent_type}")
                return {
                    "intent_type": "query_generation",
                    "generated_query": "// I'm not sure how to interpret your query. Please try rephrasing it."
                }, thinking

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "intent_type": "error",
                "generated_content": f"I encountered an error while processing your request: {str(e)}"
            }, [f"Error: {str(e)}"]