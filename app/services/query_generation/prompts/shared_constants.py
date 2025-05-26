# app/services/query_generation/prompts/shared_constants.py

# Shared KDB+/q syntax rules and guidelines used across all prompts

KDB_NOTES = """

---

### Guidelines for Multi-Line KDB+/Q Query Generation:

1.  **Break Down the Query into Logical Steps using Intermediate Variables**:
    -   Separate different stages of computation into **distinct variables** using the KDB+/Q assignment operator (e.g., `step1: select ... from ... where ...`, `step2: update ... from step1`).
    -   Avoid combining too many operations in a single complex expression.

2.  **Start with Base Data Selection/Filtering**:
    Begin by selecting **raw or filtered data** into an intermediate variable. (e.g., `filteredData: select from tableName where date = .z.d-1, sym = `AAPL`).

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

---

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
5.  **Symbols**: Prefix all symbols with a backtick (e.g., `AAPL`, `MSFT`).
6.  **Top N Queries**: Use `N#select ...` (top N) or `-N#select ...` (bottom N) (e.g., `10#select from trades where sym=`GOOG`).
7.  **Aggregations**:
    -   Common functions: `sum`, `avg`, `min`, `max`, `dev`, `var`, `count`.
    -   Group by: `select sum price, avg size by sym from trades`.
    -   `by` clauses are critical for aggregations.
8.  **Joins**:
    -   `ij` (inner join), `lj` (left join), `ej` (equi-join), `aj` (asof join), `uj` (union join).
    -   Example: `ij[trades; quotes; `sym`time]`
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