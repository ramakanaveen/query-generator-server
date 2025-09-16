# app/services/query_generator_state.py - Enhanced State Model

from typing import Tuple, List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

class QueryGenerationState(BaseModel):
    """Enhanced state for the query generation process with LLM-driven decision making."""

    # Core request information
    query: str = Field(..., description="The original natural language query")
    llm: Any = Field(..., description="The language model to use")
    database_type: str = Field(default="kdb", description="Type of database to query")

    # NEW: Intent classification results (from intent_classifier)
    intent_type: str = Field(default="query_generation", description="Type of intent detected")
    confidence: str = Field(default="medium", description="Classification confidence level")
    is_follow_up: bool = Field(default=False, description="Whether this is a follow-up query")
    classification_reasoning: str = Field(default="", description="Reasoning for intent classification")
    conversation_context_summary: str = Field(default="", description="Summary of conversation context")

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
    query_schema: Dict[str, Any] = Field(default_factory=dict, description="Retrieved schema information")
    generated_query: Optional[str] = Field(default=None, description="Generated database query")
    generated_content: Optional[str] = Field(default=None, description="Generated content for non-query intents")

    # Validation and refinement
    validation_result: Optional[bool] = Field(default=None, description="Whether the query is valid")
    validation_errors: List[Union[str, Dict[str, Any]]] = Field(default_factory=list, description="Validation errors if any")
    detailed_feedback: Optional[List[Union[str, Dict[str, Any]]]] = Field(default_factory=list, description="Detailed feedback from validation")
    validation_details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Validation details")

    # NEW: Escalation and reanalysis tracking
    escalation_count: int = Field(default=0, description="Number of complexity escalations")
    max_escalations: int = Field(default=2, description="Maximum allowed escalations")
    needs_reanalysis: bool = Field(default=False, description="Whether intelligent analyzer should re-analyze")
    escalation_reason: str = Field(default="", description="Reason for complexity escalation")

    # NEW: LLM-driven feedback analysis
    primary_issue_type: str = Field(default="", description="Primary issue type from validator feedback")
    recommended_action: str = Field(default="", description="LLM recommended action")
    specific_guidance: str = Field(default="", description="Specific guidance for regeneration")
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

    # NEW: Schema reselection and corrections
    needs_schema_reselection: bool = Field(default=False, description="Whether schema should be reselected")
    schema_changes_needed: str = Field(default="", description="Schema changes needed")
    schema_corrections_needed: str = Field(default="", description="Specific schema corrections")

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
    refinement_guidance: Optional[str] = Field(default=None, description="Guidance for query refinement")
    refinement_count: int = Field(default=0, description="Number of refinement attempts")
    max_refinements: int = Field(default=2, description="Maximum number of refinement attempts")
    llm_corrected_query: Optional[str] = Field(default=None, description="Corrected query from LLM")

    # Processing metadata
    thinking: List[str] = Field(default_factory=list, description="Thinking process during generation")
    few_shot_examples: List[str] = Field(default_factory=list, description="Few Shot Examples for query generation")

    class Config:
        arbitrary_types_allowed = True  # Allow Any type for LLM