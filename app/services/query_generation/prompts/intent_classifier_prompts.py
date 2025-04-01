# app/services/query_generation/prompts/intent_classifier_prompts.py

INTENT_CLASSIFIER_PROMPT_TEMPLATE = """
Determine the intent type of this user query: "{query}"

The query optionally includes these directives: {directives}

Choose exactly ONE of the following intent types:
1. query_generation - The user wants to generate a database query to fetch or analyze data
2. schema_description - The user wants information about the database schema, tables, or columns
3. help - The user is asking for help on how to use the system

Only output one of these exact strings: "query_generation", "schema_description", or "help".
Do not include any explanation or additional text.
"""