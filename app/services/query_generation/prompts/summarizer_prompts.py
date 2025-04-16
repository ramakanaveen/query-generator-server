# app/services/query_generation/prompts/summarizer_prompts.py

CONVERSATION_SUMMARY_PROMPT_TEMPLATE = """
Below is a conversation about database queries. Please provide a structured summary 
        with the following components:

        1. DIRECTIVES: List any directive names (like @SPOT, @STIRT) that were used
        2. TABLES: List any database tables mentioned or queried
        3. COLUMNS: List any specific columns that were discussed
        4. TIME_CONTEXT: Note any date ranges or time periods discussed
        5. QUERY_TYPES: Note what kinds of operations were performed (select, filter, aggregate)
        
        CONVERSATION:
        {conversation_text}
        
        SUMMARY:
"""