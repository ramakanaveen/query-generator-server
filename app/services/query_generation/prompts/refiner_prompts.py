# app/services/query_generation/prompts/refiner_prompts.py
REFINER_PROMPT_TEMPLATE = """
You are an expert in {database_type} queries. The following query has some issues that need to be fixed:

Query: {query}

Validation Errors:
{errors}

Please refine the query to fix these issues while maintaining the original intent.
Only provide the refined query itself, no explanations or comments.
"""
