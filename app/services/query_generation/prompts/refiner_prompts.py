# app/services/query_generation/prompts/refiner_prompts.py
REFINER_PROMPT_TEMPLATE = """
You are an expert in {database_type} queries. The following query was generated but has validation issues:

Original Query: {query}

Validation Errors:
{errors}

Please provide guidance on how to generate a better query that avoids these issues.
Your guidance will be used to generate a new query from scratch.
Focus on explaining what needs to be fixed and any specific syntax requirements.
"""