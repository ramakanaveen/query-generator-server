# app/services/query_generation/nodes/query_validator.py
import re
from typing import Dict, Any, List, Tuple
from langchain.prompts import ChatPromptTemplate

from app.core.logging import logger
from app.core.profiling import timeit

# LLM validation prompts for different database types
KDB_VALIDATION_PROMPT = """
You are an expert KDB+/q validator analyzing a generated query. Your goal is to provide detailed feedback to help improve the query.

Database Schema:
{schema}

Follow these steps to analyze the query:

1. Syntax Check: Identify any KDB+/q syntax errors
2. Schema Validation: Verify tables and columns exist in the schema
3. Semantic Analysis: Check if the query logic matches the user's intent
4. KDB Best Practices: Evaluate if the query follows KDB+/q idioms and best practices

Important KDB+/q considerations:
- KDB uses backticks for symbols: `symbol
- Tables are accessed directly: select from table (not SELECT * FROM table)
- Sorting uses xasc/xdesc: `column xdesc tableName OR select ... | xdesc `column
- Date comparisons often use .z.d (today), .z.d-1 (yesterday)
- For top N: select[N] or top N, not LIMIT

Provide a JSON-formatted response with these fields:
- valid: true/false
- critical_errors: [] (syntax or schema errors that prevent execution)
- logical_issues: [] (problems with query logic that would return incorrect results)
- improvement_suggestions: [] (ideas for better implementations)
- corrected_query: (optional improved version if available)

Based on the above information provide detailed feedback to help improve the query
Original User Query: {query}
Generated KDB Query: {generated_query}

Be specific with line numbers, error locations, and suggest corrections.
"""

SQL_VALIDATION_PROMPT = """
You are an expert SQL validator analyzing a generated query. Your goal is to provide detailed feedback to help improve the query.

Original User Query: {query}
Generated SQL Query: {generated_query}
Database Schema:
{schema}

Follow these steps to analyze the query:

1. Syntax Check: Identify any SQL syntax errors
2. Schema Validation: Verify tables and columns exist in the schema
3. Semantic Analysis: Check if the query logic matches the user's intent
4. SQL Best Practices: Evaluate if the query follows SQL idioms and best practices

Important SQL considerations:
- Table and column names should match schema exactly (case sensitivity depends on engine)
- JOIN conditions should match appropriate keys
- WHERE clauses should use appropriate operators
- Proper use of GROUP BY, HAVING, ORDER BY
- Watch for function usage and aggregation logic

Provide a JSON-formatted response with these fields:
- valid: true/false
- critical_errors: [] (syntax or schema errors that prevent execution)
- logical_issues: [] (problems with query logic that would return incorrect results)
- improvement_suggestions: [] (ideas for better implementations)
- corrected_query: (optional improved version if available)

Be specific with line numbers, error locations, and suggest corrections.
"""

# Add more prompts for other database types here


class ValidationResult:
    """Class to hold validation results with detailed feedback."""

    def __init__(self):
        self.valid = True
        self.critical_errors = []
        self.logical_issues = []
        self.improvement_suggestions = []
        self.corrected_query = None

    def add_critical_error(self, error: str):
        """Add a critical error that prevents execution."""
        self.valid = False
        self.critical_errors.append(error)

    def add_logical_issue(self, issue: str):
        """Add a logical issue that may lead to incorrect results."""
        self.logical_issues.append(issue)

    def add_improvement_suggestion(self, suggestion: str):
        """Add a suggestion for improving the query."""
        self.improvement_suggestions.append(suggestion)

    def set_corrected_query(self, query: str):
        """Set a corrected version of the query if available."""
        self.corrected_query = query

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easier processing."""
        return {
            "valid": self.valid,
            "critical_errors": self.critical_errors,
            "logical_issues": self.logical_issues,
            "improvement_suggestions": self.improvement_suggestions,
            "corrected_query": self.corrected_query
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create from dictionary."""
        result = cls()
        result.valid = data.get("valid", True)
        result.critical_errors = data.get("critical_errors", [])
        result.logical_issues = data.get("logical_issues", [])
        result.improvement_suggestions = data.get("improvement_suggestions", [])
        result.corrected_query = data.get("corrected_query")
        return result


class QueryValidator:
    """Base class for query validators."""

    def __init__(self):
        pass

    async def validate(self, query: str, schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate a query against a schema.

        Args:
            query: The query to validate
            schema: The schema to validate against

        Returns:
            ValidationResult with detailed feedback
        """
        raise NotImplementedError("Subclasses must implement validate")


class KDBValidator(QueryValidator):
    """Validator for KDB+/q queries."""

    def __init__(self):
        super().__init__()

    async def validate(self, query: str, schema: Dict[str, Any]) -> ValidationResult:
        """Validate a KDB+/q query."""
        result = ValidationResult()

        # Rule-based validation
        self._validate_syntax(query, result)
        self._validate_schema(query, schema, result)

        return result

    def _validate_syntax(self, query: str, result: ValidationResult):
        """Validate KDB+/q syntax."""
        if not query or query.isspace():
            result.add_critical_error("Generated query is empty")
            return

        # Check for unbalanced parentheses
        if query.count('(') != query.count(')'):
            result.add_critical_error("KDB/Q query has unbalanced parentheses")

        # Check for SQL-style ORDER BY (not valid in KDB/Q)
        if re.search(r'\border\s+by\b', query, re.IGNORECASE):
            result.add_critical_error("KDB/Q does not support SQL-style 'ORDER BY'. Use 'by column desc' or 'xdesc `column' instead")

        # Check for common SQL syntax that's not valid in KDB/Q
        sql_patterns = [
            (r'\bGROUP\s+BY\b', "KDB/Q does not support SQL-style 'GROUP BY'. Use 'select ... by column' instead"),
            (r'\bLIMIT\b', "KDB/Q does not support SQL-style 'LIMIT'. Use 'select[N] ...' or 'N#' instead"),
            (r'\bJOIN\b', "KDB/Q does not support SQL-style 'JOIN'. Use ',' or 'lj' for joins"),
            (r'\bAS\b', "KDB/Q does not support SQL-style column aliasing with 'AS'. Use colName:expression")
        ]

        for pattern, error_msg in sql_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                result.add_critical_error(error_msg)

        # Check for dangerous patterns
        dangerous_patterns = [
            (r'system\s*"', "KDB/Q query contains potentially unsafe system command"),
            (r'hopen\s*":', "KDB/Q query contains potentially unsafe port handling"),
            (r'read1\s*`:', "KDB/Q query contains potentially unsafe file reading"),
            (r'set\s*`:', "KDB/Q query contains potentially unsafe file writing")
        ]

        for pattern, error_msg in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                result.add_critical_error(error_msg)

    def _validate_schema(self, query: str, schema: Dict[str, Any], result: ValidationResult):
        """Validate query against schema."""
        # Skip if schema is not provided or query already has critical errors
        if not schema or result.critical_errors:
            return

        # Extract table names from the query using regex
        # This is a simplified approach; a proper parser would be better
        table_pattern = r'(?:from|join)\s+(\w+)'
        referenced_tables = re.findall(table_pattern, query, re.IGNORECASE)

        # Extract column references (simplified approach)
        column_pattern = r'(?:select|where|by|asc|desc)\s+([^,;\s]+)'
        potential_columns = re.findall(column_pattern, query, re.IGNORECASE)

        # Clean up potential columns (remove backticks, etc.)
        referenced_columns = []
        for col in potential_columns:
            col = col.strip('`')
            if col and not col.startswith('.') and not col.isdigit():
                referenced_columns.append(col)

        # Check if referenced tables exist in schema
        schema_tables = schema.get("tables", {})
        for table in referenced_tables:
            if table not in schema_tables:
                result.add_critical_error(f"Table '{table}' not found in schema. Available tables: {', '.join(schema_tables.keys())}")

        # For each referenced table that exists, check if referenced columns exist
        for table in referenced_tables:
            if table in schema_tables:
                table_schema = schema_tables[table]

                # Get schema columns
                schema_columns = []
                if isinstance(table_schema, dict) and "columns" in table_schema:
                    schema_columns = [col.get("name") for col in table_schema["columns"] if isinstance(col, dict) and "name" in col]

                # Check if columns exist in this table
                for col in referenced_columns:
                    if col not in schema_columns and col != 'i':  # 'i' is a special KDB/Q column
                        result.add_logical_issue(f"Column '{col}' might not exist in table '{table}'. Available columns: {', '.join(schema_columns)}")


class SQLValidator(QueryValidator):
    """Validator for SQL queries."""

    def __init__(self):
        super().__init__()

    async def validate(self, query: str, schema: Dict[str, Any]) -> ValidationResult:
        """Validate an SQL query."""
        # This would be implemented similarly to KDBValidator but with SQL-specific rules
        result = ValidationResult()

        # For now, just add a placeholder suggestion
        result.add_improvement_suggestion("SQL validation is currently a placeholder and should be implemented with SQL-specific rules")

        return result


class LLMValidator:
    """LLM-powered validator for queries using advanced language models."""

    def __init__(self, llm):
        """
        Initialize with a language model.

        Args:
            llm: Language model from LLMProvider
        """
        self.llm = llm
        self.validation_prompts = {
            "kdb": KDB_VALIDATION_PROMPT,
            "sql": SQL_VALIDATION_PROMPT,
            # Add more DB types here
        }

    async def validate(self, query: str, generated_query: str,
                       database_type: str, schema: Dict[str, Any]) -> ValidationResult:
        """
        Validate a query using LLM.

        Args:
            query: The original natural language query
            generated_query: The generated database query
            database_type: The type of database (kdb, sql, etc.)
            schema: The schema to validate against

        Returns:
            ValidationResult with detailed feedback
        """
        try:
            # Select appropriate prompt based on database type
            prompt_template = self.validation_prompts.get(
                database_type.lower(),
                self.validation_prompts["kdb"]  # Default to KDB if no specific prompt
            )

            # Format schema for the prompt
            schema_str = self._format_schema_for_prompt(schema)

            # Create prompt
            prompt = ChatPromptTemplate.from_template(prompt_template)

            # Get response from LLM
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "query": query,
                "generated_query": generated_query,
                "schema": schema_str
            })

            # Parse JSON response
            import json
            import re

            # Try to extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', response.content, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group(1))
            else:
                # Try without code block markers
                try:
                    validation_data = json.loads(response.content)
                except json.JSONDecodeError:
                    # If we can't parse JSON, create a basic result with the raw response
                    result = ValidationResult()
                    result.add_improvement_suggestion(f"LLM analysis: {response.content}")
                    return result

            # Create ValidationResult from the parsed data
            return ValidationResult.from_dict(validation_data)

        except Exception as e:
            logger.error(f"Error in LLM validation: {str(e)}", exc_info=True)

            # Return a basic result with the error
            result = ValidationResult()
            result.add_critical_error(f"LLM validation error: {str(e)}")
            return result

    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """Format schema information for the LLM prompt."""
        if not schema:
            return "No schema information available."

        formatted_schema = []

        # Process tables
        if "tables" in schema:
            formatted_schema.append("Tables:")

            for table_name, table_data in schema["tables"].items():
                formatted_schema.append(f"- Table: {table_name}")

                if isinstance(table_data, dict):
                    if "description" in table_data:
                        formatted_schema.append(f"  Description: {table_data['description']}")

                    if "columns" in table_data and isinstance(table_data["columns"], list):
                        formatted_schema.append("  Columns:")

                        for column in table_data["columns"]:
                            if isinstance(column, dict):
                                col_name = column.get("name", "Unknown")
                                col_type = column.get("type", column.get("kdb_type", "Unknown"))
                                col_desc = column.get("description", column.get("column_desc", ""))

                                formatted_schema.append(f"    - {col_name} ({col_type}): {col_desc}")

        return "\n".join(formatted_schema)


@timeit
async def validate_query(state):
    """
    Validate the generated query using both rule-based and LLM validation.

    Args:
        state: The current state of the workflow

    Returns:
        Updated state with validation result
    """
    try:
        generated_query = state.generated_query
        database_type = state.database_type
        query_schema = state.query_schema
        original_query = state.query
        llm = state.llm

        # Add thinking step
        state.thinking.append("Validating generated query...")

        # Initialize validation result and errors list
        validation_result = True
        validation_errors = []
        detailed_feedback = []

        # 1. First do basic rule-based validation
        if database_type.lower() == "kdb":
            validator = KDBValidator()
        elif database_type.lower() == "sql":
            validator = SQLValidator()
        else:
            # Default to KDB for now
            validator = KDBValidator()
            validation_errors.append(f"Using default KDB validator for database type: {database_type}")

        # Perform rule-based validation
        rule_validation = await validator.validate(generated_query, query_schema)

        # If there are critical errors, we know the validation failed
        if rule_validation.critical_errors:
            validation_result = False
            validation_errors.extend(rule_validation.critical_errors)
            detailed_feedback.append("Critical errors found during rule-based validation:")
            for error in rule_validation.critical_errors:
                detailed_feedback.append(f"- {error}")

        # Add logical issues to the feedback
        if rule_validation.logical_issues:
            detailed_feedback.append("Potential logical issues:")
            for issue in rule_validation.logical_issues:
                detailed_feedback.append(f"- {issue}")

        # Add improvement suggestions
        if rule_validation.improvement_suggestions:
            detailed_feedback.append("Improvement suggestions:")
            for suggestion in rule_validation.improvement_suggestions:
                detailed_feedback.append(f"- {suggestion}")

        # 2. Now perform LLM-based validation for deeper analysis
        llm_validator = LLMValidator(llm)
        llm_validation = await llm_validator.validate(
            query=original_query,
            generated_query=generated_query,
            database_type=database_type,
            schema=query_schema
        )

        # Update validation result based on LLM validation
        if not llm_validation.valid:
            validation_result = False
            validation_errors.extend(llm_validation.critical_errors)

        # Add LLM validation feedback to the detailed feedback
        if llm_validation.critical_errors:
            detailed_feedback.append("Critical issues identified by LLM:")
            for error in llm_validation.critical_errors:
                detailed_feedback.append(f"- {error}")

        if llm_validation.logical_issues:
            detailed_feedback.append("Logical issues identified by LLM:")
            for issue in llm_validation.logical_issues:
                detailed_feedback.append(f"- {issue}")

        if llm_validation.improvement_suggestions:
            detailed_feedback.append("Improvement suggestions from LLM:")
            for suggestion in llm_validation.improvement_suggestions:
                detailed_feedback.append(f"- {suggestion}")

        if llm_validation.corrected_query:
            detailed_feedback.append("LLM suggested correction:")
            detailed_feedback.append(llm_validation.corrected_query)
            # Store corrected query for potential use by the refiner
            state.llm_corrected_query = llm_validation.corrected_query

        # Update state with validation results
        state.validation_result = validation_result
        state.validation_errors = validation_errors
        state.detailed_feedback = detailed_feedback
        state.validation_details = {
            "rule_validation": rule_validation.to_dict(),
            "llm_validation": llm_validation.to_dict()
        }

        if validation_result:
            state.thinking.append("Query validation passed")
        else:
            state.thinking.append(f"Query validation failed: {', '.join(validation_errors)}")
            state.thinking.append("Detailed feedback:\n" + "\n".join(detailed_feedback))

        return state

    except Exception as e:
        logger.error(f"Error in query validator: {str(e)}", exc_info=True)
        state.thinking.append(f"Error validating query: {str(e)}")
        # Fail validation on error
        state.validation_result = False
        state.validation_errors = [str(e)]
        return state