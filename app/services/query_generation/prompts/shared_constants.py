# app/services/query_generation/prompts/shared_constants.py - Updated with Syntax Corrections

# New shared syntax corrections that will be used by both single-line and multi-line
KDB_SYNTAX_CORRECTIONS = """

---

### 🚨 CRITICAL KDB+/Q Syntax Corrections:

**These are the most common mistakes that MUST be avoided:**

#### 1. BY Clause Rules (CRITICAL):
- **Columns used in BY clause must NOT appear in SELECT clause**
- ❌ **WRONG**: `select sum price, sym by sym from trades`
- ❌ **WRONG**: `select sum size, customer by customer from trades`
- ✅ **CORRECT**: `select sum price by sym from trades`
- ✅ **CORRECT**: `select sum size by customer from trades`
- ✅ **CORRECT**: `select sum price, avg size by sym from trades` (different columns in select)

#### 2. WITHIN Clause Rules (CRITICAL):
- **Syntax**: `column_name within (start_value;end_value)`
- **Use semicolon (;) to separate range values, NOT +til operations**
- ❌ **WRONG**: `within .z.d-7 +til 7`
- ❌ **WRONG**: `date within (.z.d-7) + til 7`
- ❌ **WRONG**: `within (.z.d-7) til .z.d`
- ✅ **CORRECT**: `date within (.z.d-7;.z.d)`
- ✅ **CORRECT**: `time within (09:00:00.000;17:00:00.000)`
- ✅ **CORRECT**: `price within (100;200)`

#### 3. Date Range Best Practices:
- **For date ranges**: Use `.z.d-N` for N days ago
- **For time ranges**: Use specific times with semicolon separation
- ✅ **CORRECT**: `date within (.z.d-30;.z.d)` (last 30 days)
- ✅ **CORRECT**: `timestamp within (2023.01.01D00:00:00;2023.12.31D23:59:59)`

#### 4. Aggregation + Grouping Rules:
- **When using BY, only aggregated expressions in SELECT**
- ✅ **CORRECT**: `select count i, max price, min price by sym from trades`
- ✅ **CORRECT**: `select sum size by sym, date from trades`
- ❌ **WRONG**: `select sym, sum size by sym from trades` (sym appears in both)

---

"""

# Enhanced single-line KDB notes with syntax corrections
KDB_NOTES_SINGLE = f"""

---

### Guidelines for Single-Line KDB+/Q Query Generation:

**CRITICAL**: Generate ONE efficient KDB+/q expression that accomplishes the entire request without intermediate variables.

1.  **Combine Operations Efficiently**:
    -   Merge filtering, aggregation, and sorting into a single expression
    -   Use proper operator precedence to chain operations
    -   Example: `5#\`size xdesc select from trades where date=.z.d, sym=\`EURUSD`

2.  **Direct Column Operations**:
    -   Apply functions directly in select clauses: `select avg bid, max ask from table`
    -   Use `by` clauses for grouping: `select sum price by sym from trades where date=.z.d`

3.  **Efficient Filtering and Sorting**:
    -   Combine where clauses: `where date=.z.d, sym in` `\`EURUSD\`GBPUSD`
    -   Sort with xdesc/xasc: `\`price xdesc select from trades`
    -   Top N with direct syntax: `10#select from trades`

4.  **Avoid These in Single-Line Mode**:
    -   Intermediate variables (no `variable:` assignments)
    -   Multiple statements separated by semicolons
    -   Unnecessary complexity that can be simplified

{KDB_SYNTAX_CORRECTIONS}

### Essential KDB/Q Syntax for Single-Line Queries:

1.  **Basic Selection**: `select from table where conditions`
2.  **Aggregation**: `select sum price, avg size by sym from table where date=.z.d`
3.  **Sorting**: `\`column xdesc select from table` or `\`column xasc select from table`
4.  **Top N**: `N#select from table` (e.g., `5#select from trades`)
5.  **Symbols**: Always use backticks \`EURUSD`, \`1M`, \`MSFT`
6.  **Dates**: `.z.d` (today), `.z.d-1` (yesterday), `.z.d-7` (7 days ago)
7.  **Time Filtering**: `where date>.z.d-7` (last 7 days)
8.  **Multiple Conditions**: `where date=.z.d, sym=\`EURUSD, size>1000`

### Single-Line Patterns:
- **Simple Filter**: `select from market_price where sym=\`EURUSD, date=.z.d`
- **Aggregation**: `select avg bid, max ask by sym from market_price where date=.z.d`
- **Top N**: `5#\`size xdesc select from market_price where date=.z.d`
- **Time Range**: `select from market_price where date within (.z.d-7;.z.d), sym=\`GBPUSD`

**Focus**: Create one clear, efficient expression that reads naturally and accomplishes the full request.

---

"""

# Enhanced multi-line KDB notes with syntax corrections
KDB_NOTES_MULTI = f"""

---

### Guidelines for Multi-Line KDB+/Q Query Generation:

1.  **Break Down the Query into Logical Steps using Intermediate Variables**:
    -   Separate different stages of computation into **distinct variables** using the KDB+/Q assignment operator (e.g., `step1: select ... from ... where ...`, `step2: update ... from step1`).
    -   Avoid combining too many operations in a single complex expression.

2.  **Start with Base Data Selection/Filtering**:
    Begin by selecting **raw or filtered data** into an intermediate variable. (e.g., `filteredData: select from tableName where date within (.z.d-7;.z.d), sym=\`AAPL`).

3.  **Perform Joins, Aggregations, or Transformations Next**:
    -   Use intermediate results to perform joins (e.g., `ej`, `ij`, `lj`), compute necessary summaries (e.g., `sum`, `avg`, `dev`), or apply transformations.
    -   Ensure each step reuses prior intermediate variables logically.

4.  **Include Sorting, Ranking, or Final Transformations Last**:
    -   Apply final operations like sorting (`xdesc colName`, `xasc colName`) or selecting "top N rows" (`N#select ...`, `-N#select ...`) in the final step or as part of the output.

5.  **Self-Critique before Output**:
    - Before providing the query, review it against the provided schema.
    - **Verify all table and column names.**
    - **Check data type compatibility for operations.**
    - **Ensure KDB/Q syntax is correct.**
    If you find any discrepancies, correct them. If the user's request cannot be fulfilled with the given schema, explicitly state what is missing or problematic.

{KDB_SYNTAX_CORRECTIONS}

### Additional Notes for KDB/Q Queries

1.  **Assignment**: Use `:` for variable assignment (e.g., `myVar: select from table`).
2.  **Table Selection**: Always use `select from table` syntax - the `from` keyword IS required.
3.  **Sorting**: Use `xdesc columnName` or `xasc columnName` for sorting (e.g., `price xdesc select from trades`).
4.  **Dates & Times**:
    -   `.z.d`: Current date (e.g., `2023.10.26`)
    -   `.z.t`: Current time (e.g., `10:30:00.000`)
    -   `.z.p`: Current timestamp (e.g., `2023.10.26D10:30:00.000000000`)
    -   Relative dates: `.z.d-1` (yesterday), `.z.d-7` (a week ago).
    -   Casting: `"D"$"2023-10-26"` for string to date.
5.  **Symbols**: Prefix all symbols with a backtick (e.g., \`AAPL`, \`MSFT`).
6.  **Top N Queries**: Use `N#select ...` (top N) or `-N#select ...` (bottom N) (e.g., `10#select from trades where sym=\`GOOG`).
7.  **Aggregations**:
    -   Common functions: `sum`, `avg`, `min`, `max`, `dev`, `var`, `count`.
    -   Group by: `select sum price, avg size by sym from trades`.
    -   `by` clauses are critical for aggregations.
8.  **Joins**:
    -   `ij` (inner join), `lj` (left join), `ej` (equi-join), `aj` (asof join), `uj` (union join).
    -   Example: `ij[trades; quotes; \`sym\`time]`
9.  **Conditional Logic**:
    -   `$[condition; true_expr; false_expr]` (ternary operator).
    -   `?[table; filters; by; aggregates]` (functional select).
10. **Functional Forms**: Many operations have functional forms (e.g., `select` can be `?[tableName; (); (); ()]`). Prefer the more readable query-like syntax unless a functional form is more idiomatic for a specific task.

### Time Bucketing in KDB+/Q:
- Hourly buckets: `1h xbar time` or `0D01:00:00 xbar time`
- Minute buckets: `1m xbar time` or `0D00:01:00 xbar time`
- Custom buckets: `0D00:15:00 xbar time` (15-minute buckets)

---

"""

# Legacy constant for backward compatibility (keeping exactly the same)
KDB_NOTES = KDB_NOTES_MULTI