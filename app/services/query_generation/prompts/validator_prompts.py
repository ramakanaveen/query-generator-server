"""
Validation prompts for KDB and SQL queries.

LLM-based validation only (no rule-based validation).
"""

# KDB Validation Prompt
KDB_VALIDATION_PROMPT = """You are an expert KDB+/q validator. Validate the generated query for correctness.

# USER'S ORIGINAL REQUEST
{user_query}

# GENERATED KDB+/q QUERY
{generated_query}

# AVAILABLE SCHEMA
{schema_summary}

# QUERY COMPLEXITY
{complexity}

# VALIDATION INSTRUCTIONS

Carefully validate the generated query and check for:

## 1. Schema Validation
- ✓ All table names exist in the provided schema
- ✓ All column names exist in their respective tables
- ✓ **IMPORTANT**: Understand intermediate variables in multi-line queries
  - Variables assigned with `:` are NOT missing tables
  - Example: `t:select from trades where...` - 't' is a variable, not a table
- ✓ Don't confuse built-in KDB functions with column names
  - Functions like `max`, `min`, `avg`, `sum`, `count` are built-ins
  - Example: `select max bid` means MAX function on 'bid' column

## 2. Syntax Validation
- ✓ Proper KDB+/q syntax (not SQL syntax)
- ✓ No SQL keywords: ORDER BY, GROUP BY, LIMIT, JOIN, AS
- ✓ Balanced parentheses and brackets
- ✓ Correct function usage
- ✓ Time constants formatted correctly (`2024.01.01D`, `12:30:00.000`)
- ✓ Symbol notation correct (backtick for symbols: `AAPL)

## 3. Logic Validation
- ✓ Query actually answers the user's question
- ✓ Proper aggregations and groupings
- ✓ Time-based operations correct (xbar, within, etc.)
- ✓ Handles edge cases (null values, empty results)

## 4. Common Pitfalls to Check

**Multi-line queries**:
```q
t:select from trades where sym=`AAPL;  / t is a variable, not a missing table!
select avg bid from t                  / Referencing variable t is CORRECT
```

**Built-in functions**:
```q
select max bid from trades  / max is a built-in function, not a column!
```

**Time bucketing**:
```q
select avg bid by 5 xbar time.minute from trades  / xbar is correct syntax
```

# OUTPUT FORMAT

Respond in this EXACT JSON format:

```json
{{
  "valid": true,
  "critical_errors": [],
  "logical_issues": [],
  "improvement_suggestions": [],
  "corrected_query": null
}}
```

OR if there are errors:

```json
{{
  "valid": false,
  "critical_errors": [
    "Table 'prices' does not exist in schema (available: trades, quotes)",
    "Column 'price' not found in trades table (available: bid, ask, sym, date)"
  ],
  "logical_issues": [
    "Query groups by sym but user asked for overall average"
  ],
  "improvement_suggestions": [
    "Consider adding error handling for null values",
    "Add date filter to limit results"
  ],
  "corrected_query": "select avg bid from trades where sym=`AAPL, date=.z.d"
}}
```

**Field Descriptions**:
- `valid`: true if query is correct and can be executed, false otherwise
- `critical_errors`: Errors that prevent execution (syntax, schema mismatches)
- `logical_issues`: Query runs but may not match user intent
- `improvement_suggestions`: Optional enhancements (not errors)
- `corrected_query`: If you can fix the query, provide corrected version (otherwise null)

**IMPORTANT RULES**:
1. Be lenient with multi-line queries - intermediate variables are NOT errors
2. Understand KDB functions vs columns - don't flag built-ins as missing columns
3. Only mark as invalid if it ACTUALLY won't work or gives wrong results
4. If query is valid but could be improved, set valid=true and use improvement_suggestions

Now, validate the generated KDB+/q query.
"""

# SQL Validation Prompt
SQL_VALIDATION_PROMPT = """You are an expert SQL validator. Validate the generated query for correctness.

# USER'S ORIGINAL REQUEST
{user_query}

# GENERATED SQL QUERY
{generated_query}

# DATABASE TYPE
{database_type}

# AVAILABLE SCHEMA
{schema_summary}

# VALIDATION INSTRUCTIONS

Carefully validate the generated query and check for:

## 1. Schema Validation
- ✓ All table names exist in the provided schema
- ✓ All column names exist in their respective tables
- ✓ JOIN conditions use correct foreign keys
- ✓ **IMPORTANT**: Understand CTEs and subqueries
  - CTEs defined with WITH...AS are NOT missing tables
  - Subquery aliases are valid table references
- ✓ Understand aggregate functions vs columns
  - AVG(), MAX(), MIN(), SUM(), COUNT() are functions

## 2. Syntax Validation
- ✓ Proper SQL syntax for {database_type}
- ✓ Balanced parentheses
- ✓ Correct aggregate usage
- ✓ Valid date functions
- ✓ Proper GROUP BY with aggregations
- ✓ HAVING clause only with GROUP BY

## 3. Logic Validation
- ✓ Query actually answers the user's question
- ✓ Proper aggregations and groupings
- ✓ JOIN conditions correct
- ✓ WHERE clause filters correct
- ✓ ORDER BY and LIMIT usage appropriate

## 4. Common Pitfalls to Check

**CTEs (Common Table Expressions)**:
```sql
WITH recent_orders AS (
    SELECT * FROM orders WHERE order_date >= '2024-01-01'
)
SELECT * FROM recent_orders;  -- recent_orders is a CTE, not a missing table!
```

**Aggregate functions**:
```sql
SELECT MAX(price) FROM products;  -- MAX is a function, not a column!
```

**Subquery aliases**:
```sql
SELECT *
FROM (SELECT id, name FROM users) AS u  -- 'u' is a valid alias
WHERE u.id > 100;
```

# OUTPUT FORMAT

Respond in this EXACT JSON format:

```json
{{
  "valid": true,
  "critical_errors": [],
  "logical_issues": [],
  "improvement_suggestions": [],
  "corrected_query": null
}}
```

OR if there are errors:

```json
{{
  "valid": false,
  "critical_errors": [
    "Table 'products' does not exist in schema (available: orders, customers)",
    "Column 'total_amount' not found in orders table (available: order_id, customer_id, amount)"
  ],
  "logical_issues": [
    "Query uses WHERE with aggregates - should use HAVING instead"
  ],
  "improvement_suggestions": [
    "Add index hint for better performance",
    "Consider adding LIMIT clause to restrict results"
  ],
  "corrected_query": "SELECT customer_id, SUM(amount) as total FROM orders GROUP BY customer_id HAVING total > 1000"
}}
```

**Field Descriptions**:
- `valid`: true if query is correct and can be executed, false otherwise
- `critical_errors`: Errors that prevent execution (syntax, schema mismatches)
- `logical_issues`: Query runs but may not match user intent
- `improvement_suggestions`: Optional enhancements (not errors)
- `corrected_query`: If you can fix the query, provide corrected version (otherwise null)

**IMPORTANT RULES**:
1. Understand CTEs and subqueries - they create temporary tables
2. Recognize aggregate functions vs columns
3. Only mark as invalid if it ACTUALLY won't work or gives wrong results
4. If query is valid but could be improved, set valid=true and use improvement_suggestions
5. Be aware of {database_type}-specific syntax differences

Now, validate the generated SQL query.
"""
