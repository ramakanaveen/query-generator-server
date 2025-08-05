# app/services/query_generation/prompts/retry_prompts.py - NEW FILE

FEEDBACK_ANALYSIS_PROMPT = """
You are an expert at analyzing user feedback on database queries. Analyze what went wrong and provide structured guidance.

ORIGINAL USER REQUEST: {original_query}

PREVIOUSLY GENERATED QUERY:
```
{generated_query}
```

USER FEEDBACK: {user_feedback}

CONVERSATION CONTEXT:
Original Intent: {original_intent}
Current Understanding: {current_understanding}
Previous Corrections: {previous_corrections}

Analyze the feedback and categorize the issue:

1. **Issue Type** (choose primary category):
   - SYNTAX_ERROR: KDB+/q syntax was incorrect
   - SCHEMA_ERROR: Wrong table or column references
   - LOGIC_ERROR: Query doesn't match user intent
   - MISSING_FEATURE: User wants something not included
   - TIME_RANGE_ERROR: Wrong date/time handling
   - SYMBOL_ERROR: Wrong symbols or symbol format
   - AGGREGATION_ERROR: Wrong aggregation logic
   - FILTER_ERROR: Missing or incorrect filters

2. **Specific Problem**: What exactly was wrong?

3. **User's Actual Intent**: What does the user really want?

4. **Correction Strategy**: How should this be fixed?

5. **Learning Point**: What should be remembered to avoid this mistake again?

Format your response as:
Issue_Type: [category]
Specific_Problem: [detailed description]
User_Intent: [what user actually wants]
Correction_Strategy: [how to fix it]
Learning_Point: [lesson for future queries]
"""

RETRY_GENERATION_PROMPT = """
You are a world-class KDB+/q expert fixing a query based on user feedback and conversation context.

## Context Overview
**Original User Request**: {original_query}
**Original Intent**: {original_intent}
**Current Understanding**: {current_understanding}

## Previous Attempt Analysis
**Previously Generated Query**:
```
{original_generated_query}
```

**User Feedback**: {user_feedback}

**Analysis of What Went Wrong**:
{feedback_analysis}

## Learning from History
**Previous Corrections Applied**:
{feedback_trail}

**Key Context to Maintain**:
{key_context}

## Current Schema Context
{schema}

## Few-Shot Learning Examples
{few_shot_examples}

## Retry Generation Guidelines

### Critical Success Factors:
1. **Address the specific feedback** provided by the user
2. **Maintain the original intent** from the conversation start
3. **Learn from previous corrections** to avoid repeating mistakes
4. **Use proper schema references** from the provided context
5. **Apply KDB+/q best practices** for syntax and performance

### KDB+/q Syntax Reminders:
- Symbols: Use backticks (`EURUSD`, `AAPL`)
- Dates: `.z.d` (today), `.z.d-1` (yesterday)
- Sorting: `column xdesc table` or `column xasc table`
- Top N: `N#select from table`
- Aggregations: `select sum price, avg size by sym from table`
- Time buckets: `1h xbar time`, `0D01:00:00 xbar time`

### Feedback Integration Process:
1. **Understand the core issue** from the feedback analysis
2. **Identify what to preserve** from the original query
3. **Apply the specific correction** requested
4. **Validate against schema** and conversation context
5. **Ensure consistency** with previous corrections

## Your Task
Generate an improved KDB+/q query that:
- Addresses the user's feedback completely
- Maintains the original intent and context
- Incorporates lessons from previous corrections
- Uses correct schema references and syntax
- Follows KDB+/q best practices

**Output only the corrected query** - no explanations or markdown formatting.
"""

RETRY_CONTEXT_BUILDER_PROMPT = """
Build comprehensive context for retry query generation.

CONVERSATION ESSENCE: {conversation_essence}
RECENT MESSAGES: {recent_messages}
DIRECTIVES USED: {directives}

Extract and structure the following:
1. Original Intent: What did the user initially want?
2. Evolution: How has the intent evolved?
3. Key Constraints: What requirements must be preserved?
4. Context Elements: Important directives, symbols, timeframes
5. Success Patterns: What has worked in this conversation?

Format as structured context for query generation.
"""

# Additional helper prompts
INTENT_EVOLUTION_PROMPT = """
Analyze how user intent has evolved through conversation:

ORIGINAL REQUEST: {original_query}
FEEDBACK TRAIL: {feedback_trail}
CURRENT REQUEST: {current_feedback}

Determine:
1. Is this a refinement of original intent or new direction?
2. What core elements should be preserved?
3. What new elements are being added?
4. What previous attempts should inform this retry?

Output: Intent evolution analysis
"""

CONTEXT_PRESERVATION_PROMPT = """
Determine what context must be preserved in retry:

CONVERSATION ESSENCE: {essence}
FEEDBACK: {feedback}

Key decisions:
1. Schema/directives to maintain: {directives}
2. Time ranges to preserve: {time_context}
3. Symbols/entities to keep: {entities}
4. Aggregation patterns to maintain: {aggregations}

Output: Context preservation guide
"""