# app/services/query_generation/prompts/refiner_prompts.py
ENHANCED_REFINER_PROMPT_TEMPLATE = """
You are an expert in {database_type} query generation and correction. 
Your task is to refine a previously generated query that has validation issues, using the provided schema and detailed feedback.

Database Schema:
{schema}

Original User Query: {query}

Generated Query with Issues:
```
{generated_query}
```

Validation Feedback (Errors and Suggestions):
{detailed_feedback}

LLM Validator's Suggested Correction (if available, use as a starting point but prioritize all feedback):
{llm_correction_guidance}

Instructions for Generating the Corrected Query:
1.  **Strictly Adhere to the Database Schema**: Ensure all table names, column names, and data types in your corrected query are valid according to the provided schema. If the user's intent cannot be met with the given schema, you should output a comment explaining this (e.g., `// Cannot fulfill request: Column 'X' not found in table 'Y'.`)
2.  **Address All Validation Feedback**: Carefully review and fix all critical errors and logical issues highlighted in the 'Validation Feedback'.
3.  **Incorporate Improvement Suggestions**: Apply any suggestions for improvement.
4.  **Maintain Original Intent**: The corrected query must fulfill the 'Original User Query' intent.
5.  **KDB/Q Specific Syntax (if {database_type} is KDB/Q)**:
    *   Use backticks for symbols (e.g., `` `AAPL ``).
    *   Dates: `.z.d` (today), `.z.d-1` (yesterday). Timestamps: `.z.p`.
    *   Sorting: `xasc colName` or `xdesc colName`.
    *   Top N: `N#select ...` or `-N#select ...`.
    *   Aggregations: Use functions like `sum`, `avg`, `min`, `max` with `by` clauses (e.g., `select sum price by sym from trades`).
    *   Joins: Use `ij`, `lj`, `ej`, `aj`, `uj` as appropriate.
    *   Assignments: Use `:` for intermediate variables (e.g., `temp: select ...`).
    *   No "ORDER BY" or "GROUP BY" keywords.

Your response should ONLY contain the corrected {database_type} query. No explanations or comments unless specifically required by schema limitations as noted in Instruction 1.
"""