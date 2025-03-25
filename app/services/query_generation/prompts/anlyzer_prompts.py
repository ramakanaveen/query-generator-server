# app/services/query_generation/prompts/analyzer_prompts.py
ANALYZER_PROMPT_TEMPLATE = """
You are an expert in analyzing database queries. Your task is to analyze the following natural language query
and extract the key entities and the intent of the query.

Natural Language Query: {query}
Directives: {directives}
Database Type: {database_type}

Extract the entities (tables, columns, values, etc.) and the intent (select, count, aggregate, etc.) from the query.

Format your response as follows:
Entities: entity1, entity2, ...
Intent: intent
"""
