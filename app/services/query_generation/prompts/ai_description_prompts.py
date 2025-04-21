column_description_prompt = """
        Generate {count} concise but informative descriptions for a KDB database column named "{column_name}" in a table named "{table_name}".
        
        Full Table Context:
        Table: {table_name}
        Schema:  {schema_name}
        Table Description: {table_description}
        Column Type: {column_type}
        
        All Columns in the Table:
        {formatted_columns}
        
        Each description for the "{column_name}" column should:
        - Be 1-2 sentences long
        - Clearly explain what the column represents in the context of {table_name}
        - Mention its data type ({column_type}) and format if relevant
        - Be suitable for database documentation
        - Consider the column's relationship to other columns in the table
        
        Focus specifically on financial/trading data concepts if applicable.
        Format your response as a JSON array of strings with just the descriptions.
        """
table_description_prompt = """
        Generate {count} concise but informative descriptions for a KDB database table named "{table_name}".
        
        Context:
        Schema: {schema_name}
        
        Columns in the Table:
        {formatted_columns}
        
        Each description should:
        - Be 1-2 sentences long
        - Clearly explain what data this table stores and its purpose
        - Mention key columns or relationships if evident from the column list
        - Be suitable for database documentation
            - Focus on financial/trading concepts if applicable
        
        Format your response as a JSON array of strings with just the descriptions.
"""
examples_prompt = """
        Generate {count} example queries for a KDB/q database table named "{table_name}" with the following schema:

        Table: {table_name}
        Description: {table_description}
        
        Columns:
        {columns_text}
        
        For each example, provide:
        1. A natural language description of what the query does
        2. The corresponding KDB/q query

        Important KDB/q Syntax Notes:
        - For symbols, use backtick notation: `AAPL
        - For date filtering, use .z.d for today
        - Tables are referenced directly: select from {table_name}
        - For ordering: `column xdesc or xasc select ... from ...
        
        Format your response as a JSON array, where each item contains:
        {{
            "natural_language": "Description of the query",
            "query": "The KDB/q query code"
        }}
        """