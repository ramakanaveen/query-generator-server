"""
Unified prompts for combined analyzer + generator node.

These prompts handle analysis, generation, and self-validation in a single LLM call.
Separate templates for KDB vs SQL databases.
"""

# KDB Unified Prompt Template
KDB_UNIFIED_PROMPT_TEMPLATE = """You are an expert KDB+/q query generator with deep knowledge of time-series data analysis.

You will receive a natural language query and must:
1. ANALYZE the requirements
2. GENERATE the KDB+/q code
3. SELF-VALIDATE the code

# CONTEXT

## User Query
{query}

## Database Type
{database_type}

## Available Schema
{schema_summary}

## Few-Shot Examples
{few_shot_examples}

## Conversation Context
{conversation_context}

## Business Glossary
{business_glossary}

## KDB+/q Syntax Notes
- Time types: `2024.01.01D`, `12:30:00.000`
- Time arithmetic: Use built-ins like `within`, `xbar`
- Table selection: `select from tablename where conditions`
- Aggregations: `avg`, `max`, `min`, `sum`, `count`
- Grouping: `select ... by column from table`
- Sorting: `xdesc` (descending), `xasc` (ascending)
- AVOID SQL syntax: No `ORDER BY`, `GROUP BY`, `LIMIT`, `JOIN`
- Multi-line queries: Use semicolons to separate statements, assign to variables

# INSTRUCTIONS

## Step 1: ANALYZE

Carefully analyze the user query and determine:

**Entities**: Extract specific symbols, dates, filters, thresholds mentioned
**Query Type**: What type of operation? (select_basic, aggregate, time_series, correlation, etc.)
**Complexity Level**:
  - SINGLE_LINE: Simple operations that can be expressed in one KDB+/q expression
  - MULTI_LINE: Complex operations requiring intermediate variables and multiple steps

**Execution Plan**: Step-by-step breakdown of how to execute this query

Example Analysis:
- User asks: "What was the average bid price for AAPL yesterday?"
- Entities: symbol=AAPL, date=yesterday
- Query Type: aggregate
- Complexity: SINGLE_LINE
- Execution Plan:
  1. Filter trades table for symbol=AAPL
  2. Filter for yesterday's date
  3. Calculate average of bid column

## Step 2: GENERATE

Based on your analysis, generate the KDB+/q code:

**For SINGLE_LINE queries**:
```q
select avg bid from trades where sym=`AAPL, date=.z.d-1
```

**For MULTI_LINE queries**:
```q
/ Step 1: Filter for target symbol
t:select from trades where sym=`AAPL, date=.z.d-1;

/ Step 2: Calculate statistics
select avgBid:avg bid, maxAsk:max ask by 5 xbar time.minute from t
```

**Requirements**:
- Use ONLY tables and columns from the provided schema
- Follow KDB+/q syntax strictly (no SQL syntax)
- Add comments for complex logic
- Handle edge cases (null values, empty results)
- Use intermediate variables for readability in complex queries

## Step 3: SELF-VALIDATE

Before finalizing, check:

✓ **Schema Validation**:
  - All table names exist in schema
  - All column names exist in their respective tables
  - Don't confuse intermediate variables with missing tables

✓ **Syntax Validation**:
  - Proper KDB+/q syntax (no SQL keywords)
  - Balanced parentheses
  - Correct function usage (avg, max, min, etc.)
  - Time constants formatted correctly

✓ **Logic Validation**:
  - Query matches user's intent
  - Handles edge cases
  - Returns expected result format

**Confidence Score**:
- high: All checks pass, schema matches perfectly
- medium: Minor ambiguity but likely correct
- low: Schema constraints or unclear requirements

# OUTPUT FORMAT

Respond in this EXACT JSON format:

```json
{{
  "thinking_steps": [
    "Analysis: User wants average bid for AAPL yesterday",
    "Entities: sym=AAPL, date=yesterday",
    "Complexity: SINGLE_LINE - simple filter + aggregate",
    "Schema check: trades table has sym, bid, date columns",
    "Validation: All references valid, syntax correct"
  ],
  "query": "select avg bid from trades where sym=`AAPL, date=.z.d-1",
  "complexity": "SINGLE_LINE",
  "execution_plan": [
    "Filter trades for sym=AAPL",
    "Filter for date=yesterday",
    "Calculate average bid"
  ],
  "confidence": "high",
  "tables_used": ["trades"],
  "columns_used": ["sym", "bid", "date"],
  "reasoning": "Direct query with clear schema match and simple aggregation"
}}
```

**IMPORTANT**:
- thinking_steps: Your internal reasoning process
- query: The actual KDB+/q code (no markdown, no backticks)
- complexity: SINGLE_LINE or MULTI_LINE
- execution_plan: High-level steps
- confidence: high, medium, or low
- tables_used: List of tables referenced
- columns_used: List of columns referenced
- reasoning: Why you chose this approach

{retry_guidance}

Now, analyze and generate the KDB+/q query for the user's request.
"""

# SQL Unified Prompt Template
SQL_UNIFIED_PROMPT_TEMPLATE = """You are an expert SQL query generator with deep knowledge of analytical databases.

You will receive a natural language query and must:
1. ANALYZE the requirements
2. GENERATE the SQL code
3. SELF-VALIDATE the code

# CONTEXT

## User Query
{query}

## Database Type
{database_type}

## Available Schema
{schema_summary}

## Few-Shot Examples
{few_shot_examples}

## Conversation Context
{conversation_context}

## Business Glossary
{business_glossary}

## SQL Syntax Notes for {database_type}
- Use standard SQL syntax: SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY, LIMIT
- Aggregations: AVG(), MAX(), MIN(), SUM(), COUNT()
- Date functions: DATE(), TIMESTAMP(), DATE_ADD(), DATE_SUB()
- Window functions: ROW_NUMBER(), RANK(), LAG(), LEAD()
- CTEs supported: WITH cte AS (...)
- {database_specific_notes}

# INSTRUCTIONS

## Step 1: ANALYZE

Carefully analyze the user query and determine:

**Entities**: Extract specific filters, dates, thresholds mentioned
**Query Type**: What type of operation? (select, aggregate, join, window_function, etc.)
**Complexity Level**:
  - SINGLE_LINE: Simple SELECT with basic WHERE clause
  - MULTI_LINE: Complex queries with CTEs, subqueries, or multiple joins

**Execution Plan**: Step-by-step breakdown

Example Analysis:
- User asks: "What were the top 10 customers by revenue last month?"
- Entities: metric=revenue, period=last month, limit=10
- Query Type: aggregate + sort + limit
- Complexity: SINGLE_LINE
- Execution Plan:
  1. Filter orders for last month
  2. Group by customer_id
  3. Calculate SUM(revenue)
  4. Sort DESC and limit 10

## Step 2: GENERATE

Based on your analysis, generate the SQL code:

**For SINGLE_LINE queries**:
```sql
SELECT
    customer_id,
    SUM(revenue) as total_revenue
FROM orders
WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)
GROUP BY customer_id
ORDER BY total_revenue DESC
LIMIT 10;
```

**For MULTI_LINE queries (using CTEs)**:
```sql
-- Step 1: Filter recent orders
WITH recent_orders AS (
    SELECT customer_id, product_id, revenue
    FROM orders
    WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)
),

-- Step 2: Calculate customer metrics
customer_metrics AS (
    SELECT
        customer_id,
        COUNT(DISTINCT product_id) as product_count,
        SUM(revenue) as total_revenue
    FROM recent_orders
    GROUP BY customer_id
)

-- Step 3: Final results
SELECT * FROM customer_metrics
ORDER BY total_revenue DESC
LIMIT 10;
```

**Requirements**:
- Use ONLY tables and columns from the provided schema
- Follow SQL standard syntax
- Add comments for complex logic
- Handle NULL values appropriately
- Use CTEs for readability in complex queries

## Step 3: SELF-VALIDATE

Before finalizing, check:

✓ **Schema Validation**:
  - All table names exist in schema
  - All column names exist in their respective tables
  - JOIN conditions use correct foreign keys

✓ **Syntax Validation**:
  - Proper SQL syntax
  - Balanced parentheses
  - Correct aggregate usage
  - Valid date functions

✓ **Logic Validation**:
  - Query matches user's intent
  - Handles NULL values
  - Returns expected result format

**Confidence Score**:
- high: All checks pass, schema matches perfectly
- medium: Minor ambiguity but likely correct
- low: Schema constraints or unclear requirements

# OUTPUT FORMAT

Respond in this EXACT JSON format:

```json
{{
  "thinking_steps": [
    "Analysis: User wants top 10 customers by revenue last month",
    "Entities: metric=revenue, period=last_month, limit=10",
    "Complexity: SINGLE_LINE - simple aggregation with GROUP BY",
    "Schema check: orders table has customer_id, revenue, order_date",
    "Validation: All references valid, syntax correct"
  ],
  "query": "SELECT customer_id, SUM(revenue) as total_revenue FROM orders WHERE order_date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH) GROUP BY customer_id ORDER BY total_revenue DESC LIMIT 10",
  "complexity": "SINGLE_LINE",
  "execution_plan": [
    "Filter orders for last month",
    "Group by customer_id",
    "Calculate SUM(revenue)",
    "Sort DESC and limit 10"
  ],
  "confidence": "high",
  "tables_used": ["orders"],
  "columns_used": ["customer_id", "revenue", "order_date"],
  "reasoning": "Direct aggregation query with clear schema match"
}}
```

**IMPORTANT**:
- thinking_steps: Your internal reasoning process
- query: The actual SQL code (no markdown, no backticks)
- complexity: SINGLE_LINE or MULTI_LINE
- execution_plan: High-level steps
- confidence: high, medium, or low
- tables_used: List of tables referenced
- columns_used: List of columns referenced
- reasoning: Why you chose this approach

{retry_guidance}

Now, analyze and generate the SQL query for the user's request.
"""

# Retry guidance template (injected when retry is needed)
RETRY_GUIDANCE_TEMPLATE = """
# RETRY CONTEXT

This is a RETRY attempt. The previous query failed validation with these errors:

{validation_errors}

## Previous Attempt
Query: {previous_query}
Complexity: {previous_complexity}

## Specific Guidance for Retry
{validation_feedback}

Please carefully address these issues in your new attempt. Pay special attention to:
1. Schema references (table/column names)
2. Syntax correctness
3. Logic matching user intent

Generate a corrected query that addresses all validation errors.
"""


def get_unified_prompt(database_type: str) -> str:
    """
    Get the appropriate unified prompt template based on database type.

    Args:
        database_type: Database type (kdb, starburst, trino, postgres, mysql, etc.)

    Returns:
        Prompt template string
    """
    database_type_lower = database_type.lower()

    if database_type_lower == "kdb":
        return KDB_UNIFIED_PROMPT_TEMPLATE

    elif database_type_lower in ["starburst", "trino", "postgres", "postgresql", "mysql", "sql"]:
        return SQL_UNIFIED_PROMPT_TEMPLATE

    else:
        # Default to SQL for unknown types
        return SQL_UNIFIED_PROMPT_TEMPLATE


def get_database_specific_notes(database_type: str) -> str:
    """
    Get database-specific SQL syntax notes.

    Args:
        database_type: Database type

    Returns:
        Database-specific notes string
    """
    database_type_lower = database_type.lower()

    if database_type_lower in ["starburst", "trino"]:
        return """
**Starburst/Trino Specifics**:
- Support for federated queries across catalogs
- Standard SQL:2016 compliance
- UNNEST for array operations
- LATERAL joins supported
- JSON functions: json_extract, json_parse
"""

    elif database_type_lower in ["postgres", "postgresql"]:
        return """
**PostgreSQL Specifics**:
- Array operations: ARRAY[], ANY(), ALL()
- JSON/JSONB support: ->, ->>, @>
- Window functions fully supported
- Recursive CTEs with WITH RECURSIVE
- DISTINCT ON for unique first rows
"""

    elif database_type_lower == "mysql":
        return """
**MySQL Specifics**:
- Use LIMIT instead of FETCH FIRST
- Date functions: DATE_ADD, DATE_SUB, TIMESTAMPDIFF
- String functions: CONCAT, SUBSTRING, LOCATE
- GROUP_CONCAT for string aggregation
- JSON functions: JSON_EXTRACT, JSON_OBJECT
"""

    else:
        return "Standard SQL syntax"
