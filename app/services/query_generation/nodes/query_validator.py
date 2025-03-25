# app/services/query_generation/nodes/query_validator.py
import re
from typing import Dict, Any
from app.core.logging import logger

async def validate_query(state):
    """
    Validate the generated query for syntax and safety.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Updated state with validation result
    """
    try:
        generated_query = state.generated_query
        database_type = state.database_type
        
        # Add thinking step
        state.thinking.append("Validating generated query...")
        
        # Initialize validation result
        validation_result = True
        validation_errors = []
        
        # Basic validation
        if not generated_query or generated_query.isspace():
            validation_result = False
            validation_errors.append("Generated query is empty")
        
        # For KDB/Q, do specific syntax checks
        if database_type == "kdb":
            # Check for SQL-style ORDER BY (not valid in KDB/Q)
            if re.search(r'\border\s+by\b', generated_query, re.IGNORECASE):
                validation_result = False
                validation_errors.append("KDB/Q does not support SQL-style 'ORDER BY'. Use 'by column desc' or '| xdesc `column' instead")
            
            # Check for unbalanced parentheses
            if generated_query.count('(') != generated_query.count(')'):
                validation_result = False
                validation_errors.append("KDB/Q query has unbalanced parentheses")
            
            # Check for common SQL syntax that's not valid in KDB/Q
            sql_patterns = [
                r'\bGROUP\s+BY\b',
                r'\bLIMIT\b',
                r'\bJOIN\b',
                r'\bAS\b',
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, generated_query, re.IGNORECASE):
                    validation_result = False
                    validation_errors.append(f"KDB/Q does not support SQL-style '{re.search(pattern, generated_query, re.IGNORECASE).group(0)}' syntax")
            
            # Check for common security issues
            dangerous_patterns = [
                r'system\s*"',        # system command
                r'hopen\s*":',        # opening handles to ports
                r'read1\s*`:',        # reading files
                r'set\s*`:',          # writing files
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, generated_query, re.IGNORECASE):
                    validation_result = False
                    validation_errors.append(f"KDB/Q query contains potentially unsafe pattern: {pattern}")
        
        # Update state with validation result
        state.validation_result = validation_result
        state.validation_errors = validation_errors
        
        if validation_result:
            state.thinking.append("Query validation passed")
        else:
            state.thinking.append(f"Query validation failed: {validation_errors}")
        
        return state
    
    except Exception as e:
        logger.error(f"Error in query validator: {str(e)}")
        state.thinking.append(f"Error validating query: {str(e)}")
        # Fail validation on error
        state.validation_result = False
        state.validation_errors = [str(e)]
        return state