# app/services/query_generator_state.py - Unified State Model (v3)

"""
v3 Simplifications:
- Removed: escalation_count, max_escalations, needs_reanalysis, escalation_reason
- Removed: recommended_action, specific_guidance (replaced by validation_feedback)
- Removed: needs_schema_reselection, schema_changes_needed, schema_corrections_needed
- Removed: refinement_count, max_refinements, refinement_guidance
- Added: retry_count, validation_feedback, confidence_score

New unified flow uses simple retry logic with validation feedback instead of
complex escalation/refinement mechanisms.
"""

from typing import Tuple, List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class QueryGenerationState(BaseModel):
    """Unified state for the query generation process with thinking/reasoning model (v3)."""

    # Core request information
    query: str = Field(..., description="The original natural language query")
    llm: Any = Field(..., description="The language model to use")
    fast_llm: Optional[Any] = Field(default=None, description="The fast language model to use for simpler tasks")
    database_type: str = Field(default="kdb", description="Type of database to query")

    # Intent classification results
    intent_type: str = Field(default="query_generation", description="Type of intent detected")
    confidence: str = Field(default="medium", description="Classification confidence level")
    is_follow_up: bool = Field(default=False, description="Whether this is a follow-up query")
    classification_reasoning: str = Field(default="", description="Reasoning for intent classification")
    conversation_context_summary: str = Field(default="", description="Summary of conversation context")

    # NEW (v3): Unified generation confidence
    confidence_score: str = Field(default="medium", description="Overall confidence in generated query")

    # Enhanced directive and entity extraction
    directives: List[str] = Field(default_factory=list, description="Extracted directives from the query")
    entities: List[str] = Field(default_factory=list, description="Extracted entities from the query")

    # NEW: Schema-aware analysis results (from intelligent_analyzer)
    query_complexity: str = Field(default="SINGLE_LINE", description="Query complexity level")
    execution_plan: List[str] = Field(default_factory=list, description="Step-by-step execution plan")
    query_type: str = Field(default="select_basic", description="Specific query type")
    reasoning: str = Field(default="", description="Analysis reasoning")
    schema_constraints: str = Field(default="", description="Schema limitations or considerations")

    # Schema and generation
    schema_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from schema retrieval")
    query_schema: Dict[str, Any] = Field(default_factory=dict, description="Retrieved schema information")
    generated_query: Optional[str] = Field(default=None, description="Generated database query")
    generated_content: Optional[str] = Field(default=None, description="Generated content for non-query intents")

    # Validation
    validation_result: Optional[bool] = Field(default=None, description="Whether the query is valid")
    validation_errors: List[Union[str, Dict[str, Any]]] = Field(default_factory=list, description="Validation errors if any")
    detailed_feedback: Optional[List[Union[str, Dict[str, Any]]]] = Field(default_factory=list, description="Detailed feedback from validation")
    validation_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validation details")

    # NEW (v3): Simple retry mechanism
    retry_count: int = Field(default=0, description="Number of retry attempts")
    validation_feedback: str = Field(default="", description="Validation feedback for retry attempts")

    # LLM-driven feedback analysis
    primary_issue_type: str = Field(default="", description="Primary issue type from validator feedback")
    generation_guidance: Dict[str, Any] = Field(default_factory=dict, description="Guidance for query generator")

    # NEW: Retry-specific fields (enhanced)
    is_retry_request: bool = Field(default=False, description="Whether this is a retry request")
    original_generated_query: Optional[str] = Field(default=None, description="Previously generated query for retry")
    user_feedback: Optional[str] = Field(default=None, description="User feedback about what was wrong")
    retry_intent_analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis of retry intent")
    feedback_type: str = Field(default="", description="Type of user feedback")
    feedback_category: str = Field(default="", description="Category of feedback for retry")
    root_cause: str = Field(default="", description="Root cause analysis for retry")
    improvement_strategy: str = Field(default="", description="Strategy for improvement")
    preserve_context: List[str] = Field(default_factory=list, description="Context elements to preserve")
    change_required: str = Field(default="", description="Specific changes required")


    # Conversation context (enhanced)
    conversation_id: Optional[str] = Field(default=None, description="ID of the conversation for context")
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")
    conversation_essence: Dict[str, Any] = Field(default_factory=dict, description="Conversation essence from database")
    original_intent: Optional[str] = Field(default=None, description="Original user intent from conversation start")
    feedback_trail: List[Dict[str, Any]] = Field(default_factory=list, description="Trail of feedback and corrections")
    current_understanding: Optional[str] = Field(default=None, description="Current system understanding of user needs")
    key_context: List[str] = Field(default_factory=list, description="Key context elements")

    # User context
    user_id: Optional[str] = Field(default=None, description="User ID for personalized context")
    conversation_summary: Optional[str] = Field(default=None, description="Current conversation summary")

    # Legacy fields (maintained for compatibility)
    intent: Optional[str] = Field(default=None, description="Extracted intent from the query (legacy)")
    schema_targets: Dict[str, Any] = Field(default_factory=dict, description="Schema targets for schema description intent")
    help_request: Dict[str, Any] = Field(default_factory=dict, description="Help request details for help intent")
    no_schema_found: bool = Field(default=False, description="Flag indicating if no relevant schema was found")
    original_query: Optional[str] = Field(default=None, description="Original query before refinement")
    original_errors: List[str] = Field(default_factory=list, description="Errors in the original query")
    llm_corrected_query: Optional[str] = Field(default=None, description="Corrected query from LLM")

    # Processing metadata
    thinking: List[str] = Field(default_factory=list, description="Thinking process during generation")
    few_shot_examples: List[str] = Field(default_factory=list, description="Few Shot Examples for query generation")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

    class Config:
        arbitrary_types_allowed = True  # Allow Any type for LLM