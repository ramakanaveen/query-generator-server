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

ENHANCED_REFINER_PROMPT_TEMPLATE = """
You are an expert in {database_type} query generation. Your task is to refine a previously generated query
that has validation issues. You'll use the detailed feedback to create an improved query.

Original User Query: {query}
Generated Query with Issues:
```
{generated_query}
```

Validation Feedback:
{detailed_feedback}

Consider all the validation feedback carefully. Create a new, improved query that:
1. Addresses all critical errors
2. Fixes logical issues
3. Incorporates improvement suggestions
4. Maintains the original intent of the user query

{llm_correction_guidance}

Your response should ONLY contain the corrected query, no explanations or comments.
"""