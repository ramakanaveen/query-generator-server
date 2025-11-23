"""
Memory Module - Auto-Extraction

Automatically extracts learnings from user feedback and corrections
using LLM analysis.
"""

import json
import logging
from typing import Optional, List, Dict, Any
from uuid import UUID

from ..llm_provider import LLMProvider
from .memory_types import (
    Memory,
    MemoryType,
    SourceType,
    MemoryExtractionRequest
)

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """
    Extracts structured learnings from unstructured feedback.

    Uses LLM to analyze user feedback, corrections, and conversation
    context to automatically create reusable memories.
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize extractor.

        Args:
            llm_provider: LLM provider instance (creates one if not provided)
        """
        self.llm_provider = llm_provider or LLMProvider()

    async def extract_from_feedback(
        self,
        request: MemoryExtractionRequest
    ) -> List[Memory]:
        """
        Extract memories from user feedback.

        Args:
            request: Extraction request with feedback data

        Returns:
            List of extracted Memory objects
        """
        # Build analysis prompt
        prompt = self._build_extraction_prompt(request)

        # Get LLM to analyze
        llm = self.llm_provider.get_model(
            provider="gemini",  # Using Gemini for extraction
            temperature=0.1  # Low temperature for consistent extraction
        )

        try:
            response = await llm.ainvoke(prompt)
            analysis = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))

            # Convert analysis to Memory objects
            memories = self._analysis_to_memories(analysis, request)

            logger.info(f"Extracted {len(memories)} memories from feedback")
            return memories

        except Exception as e:
            logger.error(f"Failed to extract memories: {e}", exc_info=True)
            return []

    def _build_extraction_prompt(self, request: MemoryExtractionRequest) -> str:
        """Build prompt for LLM analysis"""

        prompt = """# Task: Extract Reusable Learnings from User Feedback

You are analyzing user feedback to extract **reusable learnings** that can help improve future query generations.

## Input Data

### Original Query
```
{original_query}
```

### Corrected Query
```
{corrected_query}
```

### User Feedback
```
{user_feedback}
```

### Recent Conversation Context
{conversation_context}

---

## Your Task

Analyze the feedback and extract **specific, actionable learnings** that fit into these categories:

### 1. SYNTAX_CORRECTION
- Corrections to query syntax (e.g., table name should be singular, not plural)
- KDB/Q specific syntax issues
- Common syntax mistakes

### 2. USER_DEFINITION
- User-specific definitions of terms (e.g., "VWAP means...")
- Personal interpretations of domain concepts
- User preferences for terminology

### 3. APPROACH_RECOMMENDATION
- Recommended approaches for certain patterns
- Performance optimization suggestions
- Best practices for specific query types

### 4. SCHEMA_CLARIFICATION
- Clarifications about what columns/tables actually mean
- Hidden relationships not obvious from schema
- Business logic constraints

### 5. ERROR_CORRECTION
- Specific errors and their fixes
- Common mistakes to avoid
- Edge cases to handle

---

## Output Format

Return a JSON array of learnings. Each learning should have:

```json
[
  {{
    "memory_type": "syntax_correction | user_definition | approach_recommendation | schema_clarification | error_correction",
    "learning_description": "Clear, concise description of what was learned",
    "original_context": "The specific context that triggered this learning",
    "corrected_version": "The corrected/improved version (if applicable)",
    "is_user_specific": true/false,
    "confidence": 0.0-1.0,
    "tags": ["tag1", "tag2"],
    "metadata": {{
      "key": "value"
    }}
  }}
]
```

## Guidelines

1. **Be Specific**: Extract concrete, actionable learnings
2. **Generalize When Appropriate**: If a learning applies broadly, mark is_user_specific=false
3. **High Confidence for Clear Corrections**: If the correction is obvious, use high confidence
4. **Low Confidence for Preferences**: Personal preferences should have lower confidence
5. **Include Relevant Metadata**: Add context that helps understand when to apply this learning

## Example

If user said:
- Original: "select from trades where sym=`AAPL"
- Corrected: "select from trade where sym=`AAPL"
- Feedback: "Table name should be trade (singular)"

Extract:
```json
[
  {{
    "memory_type": "syntax_correction",
    "learning_description": "Table name should be 'trade' (singular), not 'trades' (plural) in this KDB/Q schema",
    "original_context": "select from trades where sym=`AAPL",
    "corrected_version": "select from trade where sym=`AAPL",
    "is_user_specific": false,
    "confidence": 0.95,
    "tags": ["table_name", "syntax", "kdb"],
    "metadata": {{
      "table": "trade",
      "common_mistake": "using plural form"
    }}
  }}
]
```

---

Now analyze the feedback above and extract learnings as JSON:
""".format(
            original_query=request.original_query or "N/A",
            corrected_query=request.corrected_query or "N/A",
            user_feedback=request.user_feedback or "N/A",
            conversation_context=self._format_conversation_context(request.conversation_history)
        )

        return prompt

    def _format_conversation_context(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for prompt"""
        if not history:
            return "No conversation context available."

        context_lines = []
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            context_lines.append(f"**{role.upper()}**: {content}")

        return "\n".join(context_lines)

    def _parse_llm_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM JSON response"""
        try:
            # Extract JSON from response (handle markdown code blocks)
            response = response.strip()

            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            response = response.strip()

            # Parse JSON
            learnings = json.loads(response)

            if not isinstance(learnings, list):
                logger.warning("LLM response is not a list, wrapping in array")
                learnings = [learnings]

            return learnings

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response was: {response}")
            return []

    def _analysis_to_memories(
        self,
        analysis: List[Dict[str, Any]],
        request: MemoryExtractionRequest
    ) -> List[Memory]:
        """Convert LLM analysis to Memory objects"""

        memories = []

        for learning in analysis:
            try:
                # Validate required fields
                if 'memory_type' not in learning or 'learning_description' not in learning:
                    logger.warning(f"Skipping learning missing required fields: {learning}")
                    continue

                # Filter by confidence threshold
                confidence = learning.get('confidence', 0.6)
                if confidence < request.min_confidence:
                    logger.debug(f"Skipping low-confidence learning: {confidence:.2f}")
                    continue

                # Parse memory type
                try:
                    memory_type = MemoryType(learning['memory_type'])
                except ValueError:
                    logger.warning(f"Unknown memory type: {learning['memory_type']}")
                    continue

                # Determine user scope
                is_user_specific = learning.get('is_user_specific', False)
                user_id = request.user_id if is_user_specific else None

                # Create Memory object
                memory = Memory(
                    memory_type=memory_type,
                    user_id=user_id,
                    schema_group_id=request.schema_group_id,
                    original_context=learning.get('original_context', request.original_query),
                    learning_description=learning['learning_description'],
                    corrected_version=learning.get('corrected_version', request.corrected_query),
                    metadata=learning.get('metadata', {}),
                    source_type=SourceType.FEEDBACK,
                    source_conversation_id=request.conversation_id,
                    source_feedback_id=request.feedback_id,
                    confidence_score=confidence,
                    is_validated=request.auto_validate and confidence >= 0.9,
                    tags=learning.get('tags', [])
                )

                memories.append(memory)

            except Exception as e:
                logger.error(f"Failed to create memory from learning: {e}", exc_info=True)
                continue

        return memories

    async def extract_from_correction(
        self,
        original_query: str,
        corrected_query: str,
        user_id: Optional[str] = None,
        schema_group_id: Optional[UUID] = None,
        conversation_id: Optional[UUID] = None
    ) -> List[Memory]:
        """
        Extract learnings from a query correction.

        Simplified version for direct corrections without full feedback.

        Args:
            original_query: Original query that had issues
            corrected_query: Corrected version
            user_id: Optional user ID
            schema_group_id: Optional schema group
            conversation_id: Optional conversation ID

        Returns:
            List of extracted memories
        """
        request = MemoryExtractionRequest(
            conversation_id=conversation_id,
            user_id=user_id,
            schema_group_id=schema_group_id,
            original_query=original_query,
            corrected_query=corrected_query,
            user_feedback=f"Query corrected from:\n{original_query}\n\nTo:\n{corrected_query}",
            min_confidence=0.7  # Higher threshold for automatic corrections
        )

        return await self.extract_from_feedback(request)