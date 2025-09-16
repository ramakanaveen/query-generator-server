# app/services/query_generator.py - Final Enhanced Version
import time
from typing import Tuple, List, Dict, Any, Optional, Union
import uuid
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from app.core.logging import logger
from app.services.caching.cache_manager import get_cache_manager
from app.services.query_generation.nodes import (
    intent_classifier,
    schema_description_node,
    schema_retriever,
    query_generator_node,
    query_validator,
    query_refiner,
    intelligent_analyzer, enhanced_schema_retriever
)
from app.core.langfuse_client import langfuse_client

# Import the enhanced state model
from app.services.query_generator_state import QueryGenerationState

class QueryGenerator:
    """
    Final enhanced service for generating database queries from natural language using LangGraph.

    Features the complete LLM-driven architecture with:
    - Intent classification before schema loading
    - Schema-aware complexity analysis
    - LLM-driven feedback analysis and escalation
    - Intelligent retry handling with conversation context
    - Bounded escalation loops with graceful failure
    """

    def __init__(self, llm, use_unified_analyzer=False , enable_cache=True, use_redis_cache=False):
        self.llm = llm
        self.use_unified_analyzer = use_unified_analyzer  # For backward compatibility
        self.enable_cache = enable_cache
        self.workflow = self._build_complete_enhanced_workflow()

        # Initialize cache manager
        if self.enable_cache:
            self.cache_manager = get_cache_manager(use_redis=use_redis_cache)
            logger.info(f"Cache enabled: {'Redis' if use_redis_cache else 'Local'}")
        else:
            self.cache_manager = None
            logger.info("Cache disabled")
    def _build_complete_enhanced_workflow(self) -> StateGraph:
        """Build the complete enhanced LangGraph workflow with new architecture."""
        # Initialize the workflow with enhanced state
        workflow = StateGraph(QueryGenerationState)

        # Add all the enhanced nodes
        workflow.add_node("intent_classifier", intent_classifier.classify_intent)
        # workflow.add_node("schema_retriever", schema_retriever.retrieve_schema)
        workflow.add_node("schema_retriever", enhanced_schema_retriever.retrieve_schema_with_examples)
        workflow.add_node("intelligent_analyzer", intelligent_analyzer.intelligent_analyze_query)
        workflow.add_node("query_generator", query_generator_node.generate_query)
        workflow.add_node("query_validator", query_validator.validate_query)
        workflow.add_node("query_refiner", query_refiner.refine_query)  # Legacy fallback
        workflow.add_node("schema_description", schema_description_node.generate_schema_description)

        # Set the entrypoint to intent classification
        workflow.set_entry_point("intent_classifier")

        # Add routing based on intent type (from intent_classifier)
        workflow.add_conditional_edges(
            "intent_classifier",
            self._route_after_intent_classification
        )

        # Query generation path: schema → analysis → generation → validation
        workflow.add_edge("schema_retriever", "intelligent_analyzer")
        workflow.add_edge("intelligent_analyzer", "query_generator")
        workflow.add_edge("query_generator", "query_validator")

        # Schema description path: direct to description generation
        workflow.add_edge("schema_description", END)

        # Enhanced validation feedback loops with LLM-driven decisions
        workflow.add_conditional_edges(
            "query_validator",
            self._route_after_enhanced_validation
        )

        # Re-analysis loop (from validator feedback to intelligent analyzer)
        # This creates the intelligent feedback loop

        # Legacy refinement fallback
        workflow.add_edge("query_refiner", "query_generator")

        # Compile the workflow
        return workflow.compile()

    def _route_after_intent_classification(self, state: QueryGenerationState) -> str:
        """Route based on intent classification results."""
        intent_type = state.intent_type

        if intent_type == "schema_description":
            return "schema_description"
        elif intent_type == "help":
            # For help requests, we could add a dedicated help node
            # For now, route to schema_description which can handle help
            return "schema_description"
        else:
            # Default to query generation path
            return "schema_retriever"

    def _route_after_enhanced_validation(self, state: QueryGenerationState) -> str:
        """
        Enhanced routing after validation with complete LLM-driven feedback analysis.

        This implements the intelligent feedback loop that replaces blind retry loops.
        """

        # If validation passed, we're done
        if state.validation_result:
            return END

        # Validation failed - apply intelligent routing based on LLM analysis

        # 1. Check if LLM detected need for re-analysis (from validator)
        if getattr(state, 'needs_reanalysis', False):
            # Check escalation limits
            if state.escalation_count < state.max_escalations:
                state.thinking.append("🔄 Routing to intelligent analyzer for re-analysis")
                return "intelligent_analyzer"
            else:
                state.thinking.append("⚠️ Max escalations reached, trying legacy refinement")
                # Fall through to refinement check below

        # 2. Check if schema reselection is needed
        elif getattr(state, 'needs_schema_reselection', False):
            state.thinking.append("🔄 Routing to schema retriever for reselection")
            return "schema_retriever"

        # 3. Try legacy refinement if we haven't exceeded limits
        elif state.refinement_count < state.max_refinements:
            state.thinking.append("🔧 Routing to legacy refiner")
            state.refinement_count += 1
            return "query_refiner"

        # 4. All options exhausted - graceful failure
        else:
            failure_reason = self._determine_failure_reason(state)
            state.thinking.append(f"❌ All enhancement options exhausted: {failure_reason}")

            # Generate a helpful error message for the user
            state.generated_query = self._generate_graceful_failure_message(state, failure_reason)
            return END

    def _determine_failure_reason(self, state: QueryGenerationState) -> str:
        """Determine the primary reason for failure to provide helpful feedback."""
        validation_errors = state.validation_errors

        if not validation_errors:
            return "Unknown validation issues"

        # Analyze error patterns to provide specific feedback
        error_text = " ".join([str(error) for error in validation_errors])

        if "table not found" in error_text.lower():
            return "Required tables not available in current schema"
        elif "column not found" in error_text.lower():
            return "Required columns not available in current schema"
        elif "syntax" in error_text.lower():
            return "Complex syntax requirements beyond current capabilities"
        elif "too complex" in error_text.lower():
            return "Query requirements exceed current complexity handling"
        else:
            return "Complex query requirements that couldn't be resolved"

    def _generate_graceful_failure_message(self, state: QueryGenerationState, reason: str) -> str:
        """Generate a helpful failure message for the user."""
        base_message = f"// I apologize, but I wasn't able to generate a working query for your request.\n"
        base_message += f"// Issue: {reason}\n"
        base_message += f"// Attempts made: {state.escalation_count} complexity escalations, {state.refinement_count} refinements\n"

        # Add helpful suggestions based on the failure reason
        if "schema" in reason.lower():
            base_message += f"// Suggestion: Please verify that the required tables and columns are available in your schema.\n"
        elif "syntax" in reason.lower():
            base_message += f"// Suggestion: Try simplifying your request or breaking it into smaller parts.\n"
        elif "complex" in reason.lower():
            base_message += f"// Suggestion: Consider rephrasing your request or providing more specific details.\n"

        base_message += f"// Original request: {state.query}"

        return base_message

    def _build_cache_context(
            self,
            database_type: str,
            user_id: Optional[str] = None,
            conversation_id: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """Build context for cache key generation"""
        return {
            'database_type': database_type,
            'user_id': user_id or 'anonymous',
            'conversation_id': conversation_id,
            'model_version': getattr(self.llm, 'model_name', 'unknown'),
            'schema_version': 'latest',  # You might want to track this
            **kwargs
        }

    async def generate(
            self,
            query: str,
            database_type: str = "kdb",
            conversation_id: Optional[str] = None,
            conversation_history: Optional[List[Dict[str, Any]]] = None,
            user_id: Optional[str] = None,
            # Enhanced: Retry-specific parameters
            is_retry: bool = False,
            original_generated_query: Optional[str] = None,
            user_feedback: Optional[str] = None
    ) -> Tuple[Union[str, Dict[str, Any]], List[str]]:
        """
        Generate a database query or other content from natural language.

        Complete enhanced implementation with:
        - LLM-driven decision making throughout the workflow
        - Intelligent feedback analysis and escalation
        - Context-aware retry handling
        - Bounded loops with graceful failure
        """
        generation_start_time = time.time()

        try:
            # Build cache context
            cache_context = self._build_cache_context(
                database_type=database_type,
                user_id=user_id,
                conversation_id=conversation_id,
                is_retry=is_retry
            )

            # 🎯 STEP 1: CHECK CACHE FIRST (Option 1 approach)
            cached_result = None
            if self.enable_cache and not is_retry:  # Don't use cache for retries
                cache_start_time = time.time()
                cached_result = await self.cache_manager.get_cached_response(query, cache_context)
                cache_check_time = time.time() - cache_start_time

                if cached_result:
                    logger.info(f"Cache HIT for query: {query[:50]}... (took {cache_check_time*1000:.1f}ms)")

                    # Add cache info to thinking
                    thinking = cached_result.get('thinking', [])
                    thinking.insert(0, f"💾 Cache HIT - Retrieved cached response in {cache_check_time*1000:.1f}ms")
                    thinking.append(f"⚡ Total time saved by caching: {(time.time() - generation_start_time)*1000:.1f}ms")

                    return cached_result['result'], thinking
                else:
                    logger.info(f"Cache MISS for query: {query[:50]}... (check took {cache_check_time*1000:.1f}ms)")
            # 🎯 STEP 2: CACHE MISS - RUN NORMAL WORKFLOW
            workflow_start_time = time.time()

            # Initialize the enhanced state
            initial_state = QueryGenerationState(
                query=query,
                llm=self.llm,
                database_type=database_type,
                conversation_id=conversation_id,
                conversation_history=conversation_history or [],
                user_id=user_id,

                # Enhanced: Retry fields
                is_retry_request=is_retry,
                original_generated_query=original_generated_query,
                user_feedback=user_feedback
            )

            # Add comprehensive thinking steps
            initial_state.thinking.append(f"🚀 Starting {'retry' if is_retry else 'initial'} query processing with enhanced LLM architecture")
            initial_state.thinking.append(f"📝 Query: {query}")
            if self.enable_cache:
                initial_state.thinking.append("💾 Cache MISS - Generating fresh response")

            if database_type != "kdb":
                initial_state.thinking.append(f"🔧 Database type: {database_type}")
            if is_retry:
                initial_state.thinking.append(f"🔄 Retry feedback: {user_feedback}")
                initial_state.thinking.append(f"🔙 Previous query: {original_generated_query}")

            # Enhanced conversation history handling
            if conversation_history:
                try:
                    sanitized_history = self._sanitize_conversation_history(conversation_history)
                    initial_state.thinking.append(f"📚 Using conversation history with {len(sanitized_history)} messages")
                    initial_state.conversation_history = sanitized_history
                except Exception as e:
                    logger.error(f"Error processing conversation history: {e}")
                    initial_state.thinking.append(f"⚠️ Error processing conversation history: {e}")
                    initial_state.conversation_history = []

            # Run the complete enhanced workflow
            logger.info(f"Starting {'retry' if is_retry else 'initial'} processing with complete enhanced workflow")
            callbacks = []
            if langfuse_client.get_callback_handler():
                callbacks.append(langfuse_client.get_callback_handler())

            result = await self.workflow.ainvoke(initial_state, config={"callbacks": callbacks})
            workflow_time = time.time() - workflow_start_time
            # Extract and format results with enhanced metadata
            thinking = result.get("thinking", [])
            intent_type = result.get("intent_type", "query_generation")

            # Log workflow completion
            escalation_count = result.get("escalation_count", 0)
            refinement_count = result.get("refinement_count", 0)
            logger.info(f"Workflow completed: intent={intent_type}, escalations={escalation_count}, refinements={refinement_count}")

            # Enhanced result processing with more metadata
            if intent_type == "query_generation":
                final_result, final_thinking =  self._process_enhanced_query_generation_result(result, thinking)
            elif intent_type == "schema_description":
                final_result, final_thinking =  self._process_enhanced_schema_description_result(result, thinking)
            elif intent_type == "help":
                final_result, final_thinking =  self._process_enhanced_help_result(result, thinking)
            else:
                logger.warning(f"Unknown intent type: {intent_type}")
                final_result, final_thinking =  self._process_unknown_intent_result(result, thinking)
            # 🎯 STEP 3: POPULATE CACHE (if successful and cacheable)
            if self.enable_cache and not is_retry:
                cache_populate_start = time.time()
                # Determine if result should be cached
                should_cache = await self._should_cache_result(query, final_result, cache_context, workflow_time)

                if should_cache:
                    # Prepare cache payload
                    cache_payload = {
                        'result': final_result,
                        'thinking': final_thinking,
                        'workflow_time': workflow_time,
                        'generation_metadata': {
                            'intent_type': intent_type,
                            'escalation_count': escalation_count,
                            'refinement_count': refinement_count,
                            'timestamp': time.time()
                        }
                    }

                    # Cache the result
                    cache_success = await self.cache_manager.cache_response(query, cache_payload, cache_context)
                    cache_populate_time = time.time() - cache_populate_start

                    if cache_success:
                        logger.info(f"Cached response for query: {query[:50]}... (cache populate took {cache_populate_time*1000:.1f}ms)")
                        final_thinking.append(f"💾 Response cached for future use (took {cache_populate_time*1000:.1f}ms)")
                    else:
                        logger.warning(f"Failed to cache response for query: {query[:50]}...")
                        final_thinking.append("⚠️ Failed to cache response")
                else:
                    logger.info(f"Skipping cache for query: {query[:50]}... (not suitable for caching)")

            total_time = time.time() - generation_start_time
            logger.info(f"⏱️ Total generation time: {total_time:.4f} seconds")
            logger.info(f"✅ Query generated successfully with {len(final_thinking)} thinking steps")

            return final_result, final_thinking

        except Exception as e:
            logger.error(f"Error in complete enhanced query generation: {str(e)}", exc_info=True)
            return {
                "intent_type": "error",
                "generated_content": f"I encountered an error while processing your request: {str(e)}",
                "error_details": {
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            }, [f"❌ Critical Error: {str(e)}"]

    async def _should_cache_result(
            self,
            query: str,
            result: Dict[str, Any],
            context: Dict[str, Any],
            workflow_time: float
    ) -> bool:
        """Determine if the result should be cached"""

        # Don't cache errors
        if result.get('intent_type') == 'error':
            return False

        # Don't cache if no meaningful content generated
        if not result.get('generated_query') and not result.get('generated_content'):
            return False

        # Don't cache if workflow took too long (might be unreliable)
        if workflow_time > 30:
            logger.info(f"Not caching - workflow took too long: {workflow_time:.2f}s")
            return False

        # Don't cache if validation failed repeatedly
        if result.get('query_metadata', {}).get('escalation_count', 0) > 2:
            logger.info("Not caching - too many escalations (potentially unreliable)")
            return False

        # Use cache manager's built-in logic
        return self.cache_manager.should_cache_query(query, context)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self.enable_cache:
            return {'cache_enabled': False}

        return await self.cache_manager.get_stats()

    async def clear_cache(self, pattern: Optional[str] = None) -> Dict[str, Any]:
        """Clear cache with optional pattern matching"""
        if not self.enable_cache:
            return {'cache_enabled': False, 'cleared': 0}

        cleared_count = await self.cache_manager.clear_cache(pattern)
        return {
            'cache_enabled': True,
            'cleared': cleared_count,
            'pattern': pattern or 'all'
        }

    async def switch_to_redis_cache(self, redis_url: Optional[str] = None) -> Dict[str, Any]:
        """Switch from local cache to Redis cache"""
        if not self.enable_cache:
            return {'success': False, 'reason': 'Cache not enabled'}

        try:
            await self.cache_manager.switch_to_redis(redis_url)
            return {
                'success': True,
                'cache_type': 'redis',
                'redis_url': redis_url or 'default'
            }
        except Exception as e:
            logger.error(f"Failed to switch to Redis cache: {e}")
            return {
                'success': False,
                'reason': str(e)
            }

    def _sanitize_conversation_history(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize and validate conversation history."""
        sanitized_history = []

        for msg in conversation_history:
            try:
                msg_dict = msg
                if not isinstance(msg, dict):
                    if hasattr(msg, 'dict'):
                        msg_dict = msg.dict()
                    elif hasattr(msg, 'model_dump'):
                        msg_dict = msg.model_dump()
                    else:
                        logger.warning(f"Skipping invalid message in conversation history: {msg}")
                        continue

                if 'role' not in msg_dict or 'content' not in msg_dict:
                    logger.warning(f"Skipping message missing required fields: {msg_dict}")
                    continue

                sanitized_history.append(msg_dict)

            except Exception as e:
                logger.warning(f"Error processing message in conversation history: {e}")
                continue

        return sanitized_history

    def _process_enhanced_query_generation_result(self, result: Dict[str, Any], thinking: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """Process results for query generation intent with enhanced metadata."""
        if result.get('no_schema_found', False):
            return {
                "intent_type": "query_generation",
                "generated_query": "// I don't know how to generate a query for this request. " +
                                   "I couldn't find any relevant tables in the available schemas. " +
                                   "Please try a different query or upload relevant schemas.",
                "error_reason": "no_schema_found",
                "suggestions": [
                    "Verify that relevant schemas are uploaded",
                    "Try using different @directives",
                    "Rephrase your query to match available data"
                ]
            }, thinking

        # Enhanced result with comprehensive metadata
        return {
            "intent_type": "query_generation",
            "generated_query": result.get("generated_query", "// No query generated"),

            # Enhanced metadata from the new architecture
            "query_metadata": {
                "complexity": result.get("query_complexity", "SINGLE_LINE"),
                "execution_plan": result.get("execution_plan", []),
                "confidence": result.get("confidence", "medium"),
                "escalation_count": result.get("escalation_count", 0),
                "refinement_count": result.get("refinement_count", 0),
                "analysis_reasoning": result.get("reasoning", ""),
                "schema_constraints": result.get("schema_constraints", "")
            },

            # Validation details
            "validation_details": result.get("validation_details", {}),

            # Context preservation for future queries
            "context_preserved": {
                "entities": result.get("entities", []),
                "directives": result.get("directives", []),
                "key_context": result.get("key_context", [])
            }
        }, thinking

    def _process_enhanced_schema_description_result(self, result: Dict[str, Any], thinking: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """Process results for schema description intent with enhanced metadata."""
        return {
            "intent_type": "schema_description",
            "generated_content": result.get("generated_content", "No schema information available."),
            "schema_metadata": {
                "targets": result.get("schema_targets", {}),
                "confidence": result.get("confidence", "medium"),
                "classification_reasoning": result.get("classification_reasoning", "")
            }
        }, thinking

    def _process_enhanced_help_result(self, result: Dict[str, Any], thinking: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """Process results for help intent with enhanced metadata."""
        return {
            "intent_type": "help",
            "generated_content": result.get("generated_content", "No help information available."),
            "help_metadata": {
                "request": result.get("help_request", {}),
                "confidence": result.get("confidence", "medium")
            }
        }, thinking

    def _process_unknown_intent_result(self, result: Dict[str, Any], thinking: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """Process results for unknown intent types."""
        return {
            "intent_type": "query_generation",
            "generated_query": "// I'm not sure how to interpret your query. Please try rephrasing it.",
            "query_metadata": {
                "confidence": "low",
                "analysis_reasoning": "Unknown intent detected, defaulting to query generation",
                "escalation_count": 0,
                "refinement_count": 0
            },
            "suggestions": [
                "Try being more specific about what data you want",
                "Use @directives to specify the data source",
                "Provide examples of the output you're looking for"
            ]
        }, thinking

    # Backward compatibility method
    def set_use_unified_analyzer(self, use_unified: bool):
        """
        Backward compatibility method for toggling between architectures.
        In the enhanced version, this doesn't change behavior since we use the new architecture.
        """
        self.use_unified_analyzer = use_unified
        logger.info(f"Query generator architecture preference set to: {'unified' if use_unified else 'enhanced'}")
        # Note: The enhanced architecture is always used regardless of this setting