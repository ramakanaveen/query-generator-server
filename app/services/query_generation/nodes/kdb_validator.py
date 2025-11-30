"""
KDB Query Validator

LLM-based validation for KDB+/q queries.
No rule-based validation - relies entirely on LLM understanding.
"""

import json
import re
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from langchain.prompts import ChatPromptTemplate

from app.core.profiling import timeit
from app.core.logging import logger
from app.services.query_generation.prompts.validator_prompts import KDB_VALIDATION_PROMPT


@dataclass
class ValidationResult:
    """Result from validation"""
    valid: bool
    critical_errors: List[str]
    logical_issues: List[str]
    improvement_suggestions: List[str]
    corrected_query: str = None


class KDBValidator:
    """LLM-based validator for KDB+/q queries"""

    def __init__(self, llm):
        """
        Initialize KDB validator.

        Args:
            llm: Language model for validation
        """
        self.llm = llm

    @timeit
    async def validate(
        self,
        user_query: str,
        generated_query: str,
        schema_summary: str,
        complexity: str = "SINGLE_LINE"
    ) -> ValidationResult:
        """
        Validate a KDB+/q query using LLM.

        Args:
            user_query: Original user request
            generated_query: Generated KDB query
            schema_summary: Formatted schema information
            complexity: Query complexity level

        Returns:
            ValidationResult object
        """
        try:
            logger.info(f"Validating KDB query (complexity: {complexity})")

            # Prepare validation prompt
            prompt = ChatPromptTemplate.from_template(KDB_VALIDATION_PROMPT)
            chain = prompt | self.llm

            # Invoke LLM for validation
            response = await chain.ainvoke({
                "user_query": user_query,
                "generated_query": generated_query,
                "schema_summary": schema_summary,
                "complexity": complexity
            })

            # Parse validation result
            result = self._parse_validation_response(response.content.strip())

            # Log validation result
            if result.valid:
                logger.info("✅ Query validation passed")
            else:
                logger.warning(f"❌ Query validation failed: {len(result.critical_errors)} errors")
                for error in result.critical_errors:
                    logger.warning(f"  - {error}")

            return result

        except Exception as e:
            logger.error(f"Error in KDB validation: {str(e)}", exc_info=True)

            # Return safe default on error - assume valid to avoid blocking
            return ValidationResult(
                valid=True,  # Fail open on validation errors
                critical_errors=[],
                logical_issues=[f"Validation error: {str(e)}"],
                improvement_suggestions=["Validation failed - query may need manual review"]
            )

    def _parse_validation_response(self, response_text: str) -> ValidationResult:
        """
        Parse LLM validation response into ValidationResult.

        Args:
            response_text: Raw LLM response

        Returns:
            ValidationResult object

        Raises:
            ValueError: If parsing fails completely
        """
        try:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON object found in validation response")

            # Parse JSON
            data = json.loads(json_str)

            # Extract fields with defaults
            return ValidationResult(
                valid=data.get('valid', True),  # Default to valid
                critical_errors=data.get('critical_errors', []),
                logical_issues=data.get('logical_issues', []),
                improvement_suggestions=data.get('improvement_suggestions', []),
                corrected_query=data.get('corrected_query')
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in validation response: {str(e)}")
            logger.error(f"Response: {response_text[:500]}...")

            # Try manual extraction of valid field
            valid_match = re.search(r'"valid":\s*(true|false)', response_text, re.IGNORECASE)
            if valid_match:
                is_valid = valid_match.group(1).lower() == 'true'
                logger.info(f"Extracted valid={is_valid} from malformed response")

                return ValidationResult(
                    valid=is_valid,
                    critical_errors=["Validation response parsing error"],
                    logical_issues=[],
                    improvement_suggestions=[]
                )

            # Complete parsing failure - fail open
            logger.warning("Complete validation parse failure - defaulting to valid")
            return ValidationResult(
                valid=True,  # Fail open
                critical_errors=[],
                logical_issues=["Validation parsing failed"],
                improvement_suggestions=["Manual review recommended"]
            )


@timeit
async def validate_kdb_query(state):
    """
    Node function to validate KDB query from state.

    Args:
        state: Query generation state

    Returns:
        Updated state with validation results
    """
    try:
        # Create validator
        validator = KDBValidator(state.llm)

        # Format schema for validation
        schema_summary = format_schema_for_validation(state.query_schema)

        # Validate
        result = await validator.validate(
            user_query=state.query,
            generated_query=state.generated_query,
            schema_summary=schema_summary,
            complexity=state.query_complexity
        )

        # Update state
        state.validation_result = result.valid
        state.validation_errors = result.critical_errors + result.logical_issues

        # Store validation feedback for potential retry
        if not result.valid:
            feedback_parts = []
            if result.critical_errors:
                feedback_parts.append("Critical Errors:")
                feedback_parts.extend([f"  - {err}" for err in result.critical_errors])
            if result.logical_issues:
                feedback_parts.append("Logical Issues:")
                feedback_parts.extend([f"  - {issue}" for issue in result.logical_issues])
            if result.corrected_query:
                feedback_parts.append(f"\nSuggested correction:\n{result.corrected_query}")

            state.validation_feedback = "\n".join(feedback_parts)

        # Log to thinking
        if result.valid:
            state.thinking.append("✅ Validation passed")
        else:
            state.thinking.append(f"❌ Validation failed: {len(result.critical_errors)} errors")
            for error in result.critical_errors[:3]:  # Show first 3 errors
                state.thinking.append(f"   - {error}")

        return state

    except Exception as e:
        logger.error(f"Error in validation node: {str(e)}", exc_info=True)
        state.thinking.append(f"⚠️ Validation error: {str(e)}")

        # Fail open - assume valid
        state.validation_result = True
        state.validation_errors = []

        return state


def format_schema_for_validation(schema: Dict) -> str:
    """
    Format schema for validation prompt.

    Args:
        schema: Schema dictionary

    Returns:
        Formatted schema string
    """
    if not schema:
        return "No schema available"

    lines = []
    for table_name, table_info in schema.items():
        lines.append(f"Table: {table_name}")

        columns = table_info.get('columns', {})
        if columns:
            col_names = list(columns.keys())
            lines.append(f"  Columns: {', '.join(col_names)}")
        else:
            lines.append("  Columns: (not detailed)")

    return "\n".join(lines)
