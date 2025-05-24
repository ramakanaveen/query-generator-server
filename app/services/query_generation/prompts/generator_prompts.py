GENERATOR_PROMPT_TEMPLATE = """
### Task Overview
You are an expert in generating accurate {database_type} queries (using the {database_type} syntax) from natural language inputs. 
Your objective is to produce **clear, logically structured,{database_type} queries** aimed at handling complex data transformations.
use  multi-line {database_type} queries for the non-trivial queries.
---

### Contextual Information for Query Generation
1. **Database Schema**:
   Use the schema below to structure your query. Ensure adherence to table structure and column naming conventions:
   {schema}

2. **Instructions/Directives**:
   Follow these rules to guide query creation:
   {directives}

3. **Entities**:
   Focus on these key entities when generating the query:
   {entities}

4. **Query Intent**:
   The user’s objective or purpose for this query:
   {intent}

5. **Examples of Similar Patterns**:
   Use these examples to understand how to break down complex queries into multiple steps:
   {examples}

---

### Guidelines for Multi-Line Query Generation:

1. **Break Down the Query into Logical Steps**:
   - Separate different stages of computation into **distinct variables** or **intermediate queries**.
   - Avoid combining too many operations in a single line.

2. **Start with the Base Data**:
   Begin by selecting **raw or filtered data** (e.g., filtering by date or symbol) into an intermediate variable or table.

3. **Perform Aggregations or Transformations Next**:
   - Use intermediate results to compute necessary summaries, averages, or aggregations.
   - Ensure each step reuses prior transformations logically.

4. **Include Sorting, Ranking, or Final Transformations Last**:
   - Apply final operations like sorting (`xdesc`/`xasc`) or selecting “top N rows” (`N#select`) in the output step.

---

### Additional Notes for KDB/Q Queries

1. **Sorting**:
   Use `xdesc` or `xasc` instead of "ORDER BY".

2. **Dates**:
   For date manipulations, use `.z.d` for today’s date and calculate relative dates as needed:
   - `.z.d`: Current date
   - `.z.d-1`: Yesterday
   - `.z.d-7`: A week ago

3. **Symbols**:
   Prefix all symbols with backticks (e.g., `AAPL`).

4. **Top N Queries**:
   Use `N#select` or `-N#select` to extract the Top N rows, based on specific conditions.

---

### Natural Language Query from User
The user’s natural language query is as follows:
- **User Query**: {query}

---

### Expectations:
- **Output Format**: Your output must be a valid, single-line/ multi-line KDB query suitable for the described task.
- **Include Intermediate Steps**: Use intermediate variables to structure the logic into clear, modular steps.
- **Focus on Clarity**: Write queries that are easy to read and troubleshoot.
- **Output Only the Query**: Exclude comments or explanations.
"""

REFINED_PROMPT_TEMPLATE = """
You are a specialized language model trained to refine and correct {database_type} queries.

Database Schema:
{schema}

Issues with the Previous Query:
{original_errors}

Detailed Validation Feedback:
{detailed_feedback}

User Request: {query}

Instructions:
- Focus on resolving the identified errors.
- Follow the provided feedback and incorporate necessary changes into the query.
- Ensure the query is logical, efficient, and executable.

Generate the corrected query. Do not include any additional remarks or explanations.
"""