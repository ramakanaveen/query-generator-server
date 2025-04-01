# app/services/query_generation/prompts/generator_prompts.py
GENERATOR_PROMPT_TEMPLATE = """
You are an expert in generating {database_type} queries from natural language. Your task is to generate a valid
{database_type} query based on the following information:

Natural Language Query: {query}
Directives: {directives}
Entities: {entities}
Intent: {intent}
Database Schema:
{schema}

Examples of similar queries:
{examples}

{conversation_context}

IMPORTANT NOTES FOR KDB/Q QUERIES:
- KDB/Q does not use "ORDER BY". Instead, use:
  - For sorting within select: `select ... by col desc` or `select ... by col asc`
  - For sorting after selection: `select ... | xdesc `column` or `select ... | xasc `column`
- For date comparisons, use .z.d for today's date
- For symbols, prefix with backtick (`): `AAPL
- Top N queries use: `select[N] ...` or `select top N ...`
- Common time operations: .z.d (today), .z.d-1 (yesterday), .z.d-7 (a week ago)

Conversation Context Tips:
- Use the previous conversation to understand the context of the current query
- If the user is asking a follow-up question, maintain the context from previous queries
- If the user mentions "top N", ensure you update the number compared to previous queries
- If the user is refining a previous query, keep the relevant parts and modify as requested

Generate a valid {database_type} query that satisfies the user's request.
Only provide the query itself, no explanations or comments.
"""

REFINED_PROMPT_TEMPLATE = """
            You are an expert in {database_type} queries. 
            
            User query: {query}
            
            Schema information:
            {schema}
            
            Previous query had these issues:
            {original_errors}
            
            Guidance for improvement:
            {refinement_guidance}
            
            Generate a valid {database_type} query that addresses these issues.
            Only provide the query itself, no explanations or comments.
            """