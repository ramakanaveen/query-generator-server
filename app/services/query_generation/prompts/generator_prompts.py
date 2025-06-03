# app/services/query_generation/prompts/generator_prompts.py
from app.services.query_generation.prompts.shared_constants import KDB_NOTES_SINGLE, KDB_NOTES_MULTI, KDB_NOTES

def get_complexity_guidance(query_complexity="SINGLE_LINE", execution_plan=None):
    """
    Get complexity-specific guidance and KDB notes.

    Args:
        query_complexity: "SINGLE_LINE" or "MULTI_LINE"
        execution_plan: List of execution steps (for multi-line queries)

    Returns:
        Tuple of (complexity_guidance, kdb_notes)
    """
    if query_complexity == "SINGLE_LINE":
        complexity_guidance = """
### 🎯 SINGLE-LINE QUERY APPROACH

**CRITICAL**: Generate ONE efficient KDB+/q expression that accomplishes the entire request.

**Strategy**:
- Combine all operations (filtering, aggregation, sorting) into a single expression
- Avoid intermediate variables or multiple statements
- Focus on clarity and efficiency in one line
- Use proper KDB+/q syntax patterns shown below

"""
        return complexity_guidance, KDB_NOTES_SINGLE

    else:  # MULTI_LINE
        if execution_plan:
            plan_text = "\n".join([f"   {i+1}. {step}" for i, step in enumerate(execution_plan)])
            complexity_guidance = f"""
### 🔧 MULTI-LINE QUERY APPROACH

**CRITICAL**: Break this request into logical steps using intermediate variables.

**Execution Plan**:
{plan_text}

**Strategy**:
- Each step should correspond to the plan above
- Use meaningful variable names (e.g., `filteredData`, `bucketedPrices`, `correlationResults`)
- Build complexity incrementally through logical steps
- Final step should produce the requested output

"""
        else:
            complexity_guidance = """
### 🔧 MULTI-LINE QUERY APPROACH

**CRITICAL**: Break this complex request into logical steps using intermediate variables.

**Strategy**:
- Separate different stages of computation into distinct steps
- Use meaningful variable names for each intermediate result
- Build complexity incrementally through logical steps
- Final step should produce the requested output

"""
        return complexity_guidance, KDB_NOTES_MULTI

GENERATOR_PROMPT_TEMPLATE = """
You are a world-class KDB+/q expert specializing in financial trading systems. This task is extremely important for generating accurate market data queries that will be used in production systems.

Take a deep breath and work on this problem step-by-step.

### Task Overview
You are an expert in generating accurate {database_type} queries from natural language inputs. 
Your objective is to produce **clear, logically structured, and syntactically correct {database_type} queries** aimed at handling complex data transformations.

{complexity_guidance}

---

### Contextual Information for Query Generation
1.  **Database Schema**:
    Use the schema below to structure your query. **Strictly adhere** to table names, column names, and data types. If the user query implies operations on columns or tables not present, you MUST state that limitation clearly instead of inventing them.
    {schema}

2.  **Instructions/Directives**:
    Focus on these key directive groups when generating the query:
    {directives}

3.  **Entities**:
    Focus on these key entities extracted from the user query:
    {entities}

4.  **Query Intent**:
    The user's objective or purpose for this query:
    {intent}

5.  **Examples of Similar Patterns**:
    Use these examples to understand how to break down complex queries into multiple steps using intermediate assignments:
    {examples}

6.  **Conversation Context**:
    Consider previous interactions for better context understanding:
    {conversation_context}

{kdb_notes}

### Critical Success Factors for {database_type}:
- **Production-Ready**: Query must execute without errors in real trading systems
- **Performance-Optimized**: Use efficient KDB+/q patterns and vector operations
- **Schema-Compliant**: All references must match the provided schema exactly
- **Intent-Accurate**: Query logic must precisely match the user's request
- **Complexity-Appropriate**: Follow the {query_complexity} approach as specified above

### Pre-Generation Validation Checklist:
Before outputting your query, mentally verify:
- [ ] All table names exist in the provided schema
- [ ] All column names are spelled correctly and exist
- [ ] All symbols use proper backtick notation (`AAPL`, `MSFT`)
- [ ] Date operations use KDB+ conventions (.z.d, .z.d-1, etc.)
- [ ] Sorting uses xdesc/xasc syntax, not ORDER BY
- [ ] The `from` keyword is included in select statements
- [ ] Query logic matches the user's stated intent
- [ ] Approach matches the specified complexity: {query_complexity}

### Natural Language Query from User
The user's natural language query is as follows:
- **User Query**: {query}

### Your Mission:
Generate a production-ready {database_type} query that perfectly fulfills the user's request following the {query_complexity} approach.

**Output Only the Query** - no comments, explanations, or markdown formatting.
"""

REFINED_PROMPT_TEMPLATE = """
You are a specialized KDB+/q query refinement expert with deep expertise in error correction and optimization.

This is a critical query refinement task - take a deep breath and approach this systematically.

### Refinement Context
**Database Type**: {database_type}
**Original User Request**: {query}

### Analysis of the Problematic Query
The following query was generated but failed validation:
```
{original_query}
```

### Comprehensive Error Analysis
**Issues Identified in the Previous Query:**
{original_errors}

**Detailed Validation Feedback and Correction Guidance:**
{detailed_feedback}

### Database Schema Reference
Use this schema to ensure all corrections are accurate:
{schema}

{kdb_notes}

### Systematic Refinement Approach:

#### Step 1: Error Classification
Categorize each error as:
- **Syntax Error**: Incorrect KDB+/q syntax that prevents execution
- **Schema Error**: References to non-existent tables or columns  
- **Logic Error**: Query doesn't match user intent
- **Performance Issue**: Inefficient patterns that should be optimized

#### Step 2: Correction Strategy
For each identified error:
1. **Root Cause Analysis**: Why did the original query fail?
2. **Schema Verification**: Confirm correct table/column references
3. **Syntax Correction**: Apply proper KDB+/q syntax rules
4. **Logic Validation**: Ensure corrected logic matches user intent

#### Step 3: Quality Assurance
Before finalizing the corrected query:
- [ ] All syntax follows KDB+/q conventions
- [ ] All schema references are validated against provided schema
- [ ] Query logic precisely addresses the original user request
- [ ] Performance is optimized using KDB+/q best practices
- [ ] Complex operations are properly structured with intermediate variables

### Critical Correction Points:
- **Symbol Notation**: Ensure all symbols use backticks (`AAPL`, `MSFT`)
- **Table References**: Verify all table names exist in schema
- **Column References**: Confirm all column names are correct
- **Date Operations**: Use KDB+ date functions (.z.d, .z.d-1, etc.)
- **Sorting Syntax**: Use xdesc/xasc, not SQL-style ORDER BY
- **Select Syntax**: Always include `from` keyword in select statements

### Instructions for Correction:
1.  **Re-evaluate the entire query against the provided Database Schema.** Ensure every table, column, and data type reference is accurate and valid.
2.  **Address all specific issues** listed in the error feedback systematically.
3.  **Verify KDB/Q Syntax meticulously**: Pay close attention to operators, function names, symbol notation (backticks), date/time handling, and join syntax.
4.  **Maintain the original query intent** as described in the user request.
5.  If the feedback points to schema limitations (e.g., a requested column doesn't exist), modify the query to work within the schema constraints or explain why it's impossible.
6.  The refined query should be efficient and logically sound.

### Your Task:
Generate the completely corrected {database_type} query that resolves all identified issues while maintaining the original user intent.

**Output Only the Corrected Query** - no additional remarks, explanations, or formatting.
"""

