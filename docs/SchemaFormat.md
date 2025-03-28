# Schema Format Guide

This guide explains the expected format for schema files used in the Query Generator application. Well-structured schemas are crucial for accurate query generation.

## Overview

The schema files provide the system with information about database tables, columns, and example queries. This information is used to:

1. Provide context to the LLM when generating queries
2. Enable vector search to find relevant tables
3. Supply example queries for few-shot learning

## File Format

Schema files should be JSON files with the following high-level structure:

```json
{
  "group": "GroupName",
  "schemas": [
    {
      "schema": [
        {
          "name": "schema_name",
          "description": "Schema description",
          "tables": [
            // Table definitions
          ]
        }
      ]
    }
  ]
}
```

## Hierarchy

The schema system follows a hierarchical structure:

1. **Group**: Top-level organization (e.g., "MarketData", "RiskAnalytics")
2. **Schemas**: Collection of related schemas within a group
3. **Schema**: Definition of a specific database schema
4. **Table**: Definition of a table within a schema
5. **Column**: Definition of a column within a table
6. **Example**: Examples of natural language queries and their corresponding database queries

## Detailed Structure

### Group Level

```json
{
  "group": "MarketData",
  "schemas": [
    // Schema definitions
  ]
}
```

### Schema Level

```json
{
  "schema": [
    {
      "name": "spot",
      "description": "Spot market trading data",
      "tables": [
        // Table definitions
      ]
    }
  ]
}
```

### Table Level

```json
{
  "kdb_table_name": "market_price",
  "description": "Spot market price data",
  "columns": [
    // Column definitions
  ],
  "examples": [
    // Example queries specific to this table
  ]
}
```

### Column Level

```json
{
  "name": "date",
  "kdb_type": "d",
  "type": "Date",
  "column_desc": "Trade date",
  "references": null
}
```

Columns should include:
- `name`: Column name
- `type`: Data type (e.g., "Date", "Symbol", "Float")
- `kdb_type`: KDB+ data type (e.g., "d" for date, "s" for symbol)
- `column_desc`: Description of the column
- `references` (optional): Reference to another table for foreign keys

### Example Level

```json
{
  "natural_language": "Show me the top 5 AAPL trades by size today",
  "query": "xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL"
}
```

Examples should include:
- `natural_language`: The natural language query
- `query`: The corresponding database query

## Complete Example

Here's a complete example of a schema file:

```json
{
  "group": "MarketData",
  "schemas": [
    {
      "schema": [
        {
          "name": "spot",
          "description": "Spot market trading data",
          "tables": [
            {
              "kdb_table_name": "market_price",
              "description": "Spot market price",
              "columns": [
                {
                  "name": "date",
                  "kdb_type": "d",
                  "type": "Date",
                  "column_desc": "Trade date",
                  "references": null
                },
                {
                  "name": "time",
                  "kdb_type": "t",
                  "type": "Time",
                  "column_desc": "Trade time",
                  "references": null
                },
                {
                  "name": "sym",
                  "kdb_type": "s",
                  "type": "Symbol",
                  "column_desc": "Ticker symbol",
                  "references": null
                },
                {
                  "name": "price",
                  "kdb_type": "f",
                  "type": "Float",
                  "column_desc": "Trade price",
                  "references": null
                },
                {
                  "name": "size",
                  "kdb_type": "j",
                  "type": "Long",
                  "column_desc": "Trade size",
                  "references": null
                },
                {
                  "name": "side",
                  "kdb_type": "s",
                  "type": "Symbol",
                  "column_desc": "Buy or sell",
                  "references": null
                }
              ],
              "examples": [
                {
                  "natural_language": "Show me the top 5 AAPL trades by size today",
                  "query": "xdesc `size select top 5 from trades where date=.z.d, sym=`AAPL"
                },
                {
                  "natural_language": "What was the average price of MSFT trades yesterday?",
                  "query": "select avg price from trades where date=.z.d-1, sym=`MSFT"
                },
                {
                  "natural_language": "Count all Google trades by side",
                  "query": "select count i by side from trades where date=.z.d, sym=`GOOGL"
                },
                {
                  "natural_language": "Show me the largest trades today",
                  "query": "xdesc `size select from trades where date=.z.d"
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

## Vector Embedding Considerations

When the system processes schema files, it generates vector embeddings for tables using:

1. Table name
2. Table description
3. Column names and descriptions

To improve vector search results:

- Use descriptive table and column names
- Provide detailed table and column descriptions
- Use consistent terminology
- Include domain-specific terms that might appear in user queries

## Examples Best Practices

Good examples significantly improve query generation. Follow these best practices:

1. **Cover Common Query Types**:
   - Simple selections
   - Filtering
   - Aggregations
   - Sorting
   - Limiting results

2. **Include Domain-Specific Queries**:
   - Use relevant entity names (e.g., stock symbols)
   - Include industry terminology
   - Cover domain-specific operations

3. **Demonstrate Syntax**:
   - Show proper KDB/Q syntax
   - Include date/time handling
   - Show symbol notation
   - Demonstrate sorting

4. **Variety in Phrasing**:
   - Use different ways to ask for the same information
   - Include both formal and casual phrasing
   - Vary the complexity of natural language

## Example Queries

Here are additional examples for different query types:

### Selection Queries

```json
{
  "natural_language": "Get all IBM trades from yesterday",
  "query": "select from trades where date=.z.d-1, sym=`IBM"
}
```

### Aggregation Queries

```json
{
  "natural_language": "Calculate average, minimum, and maximum TSLA prices today",
  "query": "select avg price, min price, max price from trades where date=.z.d, sym=`TSLA"
}
```

### Grouping Queries

```json
{
  "natural_language": "Count trades by symbol for today",
  "query": "select count i by sym from trades where date=.z.d"
}
```

### Sorting Queries

```json
{
  "natural_language": "Show trades sorted by price in descending order",
  "query": "xdesc `price select from trades where date=.z.d"
}
```

### Time-Based Queries

```json
{
  "natural_language": "Get trades between 9:30 AM and 10:00 AM today",
  "query": "select from trades where date=.z.d, time within 09:30:00.000 10:00:00.000"
}
```

## Schema Relationships

The schema system supports relationships between tables:

1. **Foreign Keys**: Use the `references` field in column definitions:
   ```json
   {
     "name": "account_id",
     "kdb_type": "s",
     "type": "Symbol",
     "column_desc": "Account identifier",
     "references": "accounts"
   }
   ```

2. **Cross-Schema Examples**: Examples that span multiple tables:
   ```json
   {
     "natural_language": "Show trades and their corresponding account information",
     "query": "select t.time, t.sym, t.price, t.size, a.name from trades t, accounts a where t.account_id=a.id, t.date=.z.d"
   }
   ```

## Importing Schemas

You can import schemas using:

1. **API Upload**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/schemas/upload" \
     -F "file=@./my_schema.json" \
     -F "name=SPOT" \
     -F "description=Spot market schema"
   ```

2. **Import Script**:
   ```bash
   python scripts/import_schema.py --file path/to/schema.json
   ```

3. **Batch Import**:
   ```bash
   python scripts/import_schema.py --directory path/to/schemas
   ```

## Schema Version Management

The system supports versioning of schemas:

- Each schema can have multiple versions
- Only one version can be active at a time
- When you upload a new version, you can activate it

The versions are managed in the database in these tables:
- `schema_definitions`: Basic schema information
- `schema_versions`: Version-specific information
- `active_schemas`: Tracks which version is currently active

## Schema Debugging

If your schemas aren't working as expected:

1. Check the structure matches the expected format
2. Validate the JSON syntax
3. Test vector search with specific queries:
   ```bash
   python scripts/test_vector_search.py "Your query here"
   ```
4. Examine the logs when a query is processed
5. Try adjusting the similarity threshold:
   ```
   SCHEMA_SIMILARITY_THRESHOLD=0.5
   ```

## Schema Migration

To migrate schemas from old formats:

1. Review the `scripts/db_scripts/migration_script.sql` file
2. Use the import script with appropriate transformations
3. Validate that all tables and columns are correctly imported
