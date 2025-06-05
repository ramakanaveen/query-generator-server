# app/services/query_generation/prompts/intent_classifier_prompts.py

INTENT_CLASSIFICATION_PROMPT = """
You are an expert intent classifier for database query systems. Your job is to understand what the user wants and classify their intent accurately.

## User Request Analysis

**User Query**: {query}
**Available Directives**: {directives}
**Database Type**: {database_type}
**Conversation Context**: {conversation_context}
**Is Retry Request**: {is_retry}
**User Feedback** (if retry): {user_feedback}

## Intent Classification Task

Classify the user's primary intent into exactly ONE of these categories:

### 1. query_generation
The user wants to generate a database query to retrieve, filter, calculate, or analyze data.

**Examples:**
- "Show me EURUSD prices today"
- "Calculate correlation between currencies"
- "Top 10 trades by volume"
- "Average bid price for AAPL"
- "Filter trades where size > 1000"

**Indicators:**
- Requests for specific data
- Calculations or aggregations
- Filtering or sorting requests
- Data analysis queries
- Follow-up questions about data

### 2. schema_description  
The user wants to understand the database structure, available tables, columns, or data organization.

**Examples:**
- "What tables do you have?"
- "Show me the schema structure"
- "What market data is available?"
- "Describe the trades table"
- "What columns are in market_price?"

**Indicators:**
- Questions about tables, columns, schema
- Requests for data structure information
- "What" questions about available data
- Schema exploration requests

### 3. help
The user needs assistance with using the system, syntax, or general guidance.

**Examples:**
- "How do I query data?"
- "Help me with KDB syntax"
- "What can I do with this system?"
- "How do I use directives?"
- "Explain how to write queries"

**Indicators:**
- Requests for help or guidance
- Questions about system capabilities
- Syntax help requests
- Tutorial or instruction requests

## Special Considerations

### Follow-up Detection
Determine if this is a follow-up to previous queries by looking for:
- Continuation words: "also", "and", "additionally"
- Modification words: "change", "modify", "instead", "but"
- Reference to previous results: "that", "those", "the same"
- Time modifiers: "yesterday instead", "different symbol"

### Retry Request Analysis
If this is a retry request (is_retry=True), analyze the user feedback to understand:
- What went wrong with the previous attempt
- What the user actually wants
- Whether the intent type should remain the same or change

### Context-Aware Classification
Consider the conversation context to:
- Maintain consistency with previous interactions
- Understand abbreviated requests in context
- Detect intent shifts in the conversation

## Output Format

Provide your analysis in this exact format:

```
INTENT_TYPE: [query_generation | schema_description | help]
CONFIDENCE: [high | medium | low]
IS_FOLLOW_UP: [true | false]
REASONING: [Brief explanation of your classification decision]
CONVERSATION_CONTEXT_SUMMARY: [Key context from conversation that influenced your decision]
RETRY_ANALYSIS: [If retry: what went wrong and what user wants now]
```

## Classification Guidelines

- **High Confidence**: Very clear what the user wants (>90% certain)
- **Medium Confidence**: Reasonably clear but some ambiguity (70-90% certain)
- **Low Confidence**: Ambiguous or unclear request (<70% certain)

- **Default to query_generation** if the intent is unclear but involves data
- **Only classify as schema_description** if explicitly asking about structure
- **Only classify as help** if explicitly asking for guidance or instructions

## Examples

**Example 1:**
User: "Show me EURUSD trades today"
→ INTENT_TYPE: query_generation (clear data request)

**Example 2:**
User: "What tables are available for FX data?"
→ INTENT_TYPE: schema_description (asking about structure)

**Example 3:**
User: "How do I filter by date in KDB?"
→ INTENT_TYPE: help (asking for guidance)

**Example 4:**
User: "Change that to GBPUSD instead" (with previous EURUSD query)
→ INTENT_TYPE: query_generation, IS_FOLLOW_UP: true

Now classify the user's request above.
"""

RETRY_INTENT_ANALYSIS_PROMPT = """
You are analyzing a retry request where the user provided feedback about a previously generated query.

## Context
**Original User Request**: {original_query}
**Previously Generated Query**: {previous_query}
**User Feedback**: {user_feedback}
**Previous Intent**: {previous_intent}
**Conversation History**: {conversation_context}

## Analysis Task

Determine:

1. **Intent Preservation**: Should the intent type remain the same or change?
   - Did the user's fundamental goal change?
   - Is this still about the same type of request?

2. **Feedback Classification**: What type of feedback is this?
   - correction: "Wrong table/column"
   - modification: "Change X to Y"  
   - clarification: "I meant Z not W"
   - complexity_feedback: "Too complex" or "Need more detail"
   - completely_different: "Actually I want something else"

3. **Context Preservation**: What should be preserved from the original request?
   - Schema context (tables, directives)
   - Time ranges or filters
   - Basic intent and entities

## Output Format

```
INTENT_SHOULD_CHANGE: [true | false]
NEW_INTENT_TYPE: [query_generation | schema_description | help | same]
FEEDBACK_TYPE: [correction | modification | clarification | complexity_feedback | completely_different]
PRESERVE_CONTEXT: [list of elements to preserve]
CHANGE_REQUIRED: [specific changes needed]
REASONING: [explanation of your analysis]
```

Analyze the retry request above.
"""