# app/services/query_generation/prompts/schema_description_prompts.py

SCHEMA_DESCRIPTION_PROMPT_TEMPLATE = """
I'm going to provide you with database schema information. Please generate a clear, 
well-organized description of this schema information.

Schema Information:
{schema_info}

Detail Level: {detail_level}
Database Type: {database_type}

The user asked: "{query}"

IMPORTANT: 
- Format your response in markdown with appropriate headers, lists, and code blocks for readability.
- Focus only on the schemas, tables, and columns specified in the request.
- When providing example queries, use {database_type} syntax.
- For KDB/Q queries (if database_type is "kdb"):
  - Use `.z.d` for today's date
  - Use backtick (`) for symbols: `` `AAPL ``
  - Tables are referenced directly without FROM: `select from table where ...`
  - Use `select from table_name` for KDB/Q syntax instead of SQL's `SELECT * FROM table_name`
- For SQL queries (if database_type is "sql"):
  - Use standard SQL syntax: `SELECT * FROM table_name WHERE condition`
  - Include column names and proper SQL keywords
"""