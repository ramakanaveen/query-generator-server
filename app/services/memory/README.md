```markdown
# Long-Term Memory Module

A reusable, independent memory system that learns from user interactions across conversations.

## 🎯 Overview

The Memory Module automatically captures and retrieves learnings from:
- **Syntax corrections**: "Table should be 'trade' not 'trades'"
- **User definitions**: "VWAP means sum(price*volume)/sum(volume)"
- **Approach recommendations**: "Filter by date before aggregating for better performance"
- **Schema clarifications**: "Column 'px' means 'price' not 'pixel'"
- **Error corrections**: "Don't use WHERE with aggregations"

These learnings persist across conversations and can be:
- **User-specific**: Personal preferences and interpretations
- **Global**: Learnings that benefit all users

## 🏗️ Architecture

```
Memory Module (Independent & Reusable)
├── memory_manager.py       → Public API (main interface)
├── memory_storage.py       → Database operations
├── memory_extractor.py     → Auto-extraction from feedback (uses LLM)
├── memory_types.py         → Data structures and enums
└── README.md              → This file

Integration Points
├── memory_retrieval_node.py → LangGraph node
└── Your existing services   → Use MemoryManager API
```

## 📦 Installation

### Step 1: Initialize Database Schema

```bash
# Run the SQL script to create memory schema
psql -U your_user -d your_database -f scripts/init_memory_schema.sql
```

This creates:
- `memory.entries` - Main memory table
- `memory.tags` - Memory tags
- `memory.usage_log` - Usage analytics
- Indexes for fast retrieval
- Helper functions and views

### Step 2: Update Requirements

The memory module uses existing dependencies:
- `asyncpg` - Database access
- `langchain` - LLM integration
- Your existing `EmbeddingProvider` and `LLMProvider`

No additional packages needed!

### Step 3: Verify Installation

```python
from app.services.memory import MemoryManager

# Test initialization
memory_manager = MemoryManager()
stats = await memory_manager.get_stats()
print(f"Memory system ready! {stats.total_memories} memories stored.")
```

## 🚀 Usage

### Basic API Usage

```python
from app.services.memory import MemoryManager, MemoryType

# Initialize (singleton)
memory_manager = MemoryManager()

# 1. Store a memory manually
memory_id = await memory_manager.store(
    memory_type=MemoryType.SYNTAX_CORRECTION,
    original_context="select from trades where sym=`AAPL",
    learning="Table name should be 'trade' (singular), not 'trades'",
    corrected_version="select from trade where sym=`AAPL",
    confidence_score=0.9,
    tags=["table_name", "syntax", "kdb"]
)

# 2. Retrieve relevant memories
memories = await memory_manager.retrieve(
    query="Show me trades for AAPL",
    user_id="john@example.com",
    limit=5
)

for result in memories:
    print(f"Learning: {result.memory.learning_description}")
    print(f"Confidence: {result.memory.confidence_score}")
    print(f"Similarity: {result.similarity_score}")

# 3. Format for LLM prompt
memory_context = memory_manager.format_memories_for_prompt(memories)
# Add this to your prompt
```

### Auto-Extraction from Feedback

```python
# Automatically extract learnings from user feedback
memory_ids = await memory_manager.extract_and_store_from_feedback(
    feedback_id=feedback_id,
    conversation_id=conversation_id,
    original_query="select from trades",
    corrected_query="select from trade",
    user_feedback="Table name should be singular",
    user_id="john@example.com",
    schema_group_id=schema_group_id,
    conversation_history=[...]  # Recent messages
)

print(f"Extracted {len(memory_ids)} learnings from feedback")
```

### Track Memory Effectiveness

```python
# After using a memory, track if it was helpful
await memory_manager.mark_helpful(memory_id, helpful=True)

# This:
# - Increases confidence score
# - Increments success count
# - Improves future rankings
```

## 🔗 Integration with Query Generation Pipeline

### Option 1: Add to LangGraph (Recommended)

Update `app/services/query_generator.py`:

```python
from app.services.query_generation.nodes.memory_retrieval_node import memory_retrieval_node

def _build_graph(self) -> StateGraph:
    """Build LangGraph DAG with memory retrieval"""
    graph = StateGraph(QueryGenerationState)

    # Existing nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("schema_retriever", schema_retriever_node)

    # NEW: Add memory retrieval
    graph.add_node("memory_retriever", memory_retrieval_node)

    graph.add_node("intelligent_analyzer", intelligent_analyzer_node)
    graph.add_node("query_generator", query_generator_node)
    graph.add_node("query_validator", query_validator_node)
    graph.add_node("query_refiner", query_refiner_node)

    # Update edges
    graph.set_entry_point("intent_classifier")
    graph.add_edge("intent_classifier", "schema_retriever")

    # Insert memory retrieval after schema retrieval
    graph.add_edge("schema_retriever", "memory_retriever")
    graph.add_edge("memory_retriever", "intelligent_analyzer")

    # Rest of the pipeline...
    graph.add_edge("intelligent_analyzer", "query_generator")
    # ...

    return graph.compile()
```

### Option 2: Direct Integration

Integrate directly in your query generation logic:

```python
from app.services.memory import MemoryManager

async def generate_query(user_query: str, user_id: str):
    # Retrieve relevant memories
    memory_manager = MemoryManager()
    memories = await memory_manager.retrieve(
        query=user_query,
        user_id=user_id,
        limit=5
    )

    # Add to prompt context
    memory_context = memory_manager.format_memories_for_prompt(memories)

    prompt = f"""
    User Query: {user_query}

    {memory_context}

    Schema: ...

    Generate query:
    """

    # Your LLM call with enhanced prompt
    query = await llm.ainvoke(prompt)

    return query
```

## 🔄 Auto-Extraction Integration

### Hook into Feedback System

Update `app/routes/feedback.py` or `app/services/feedback_manager.py`:

```python
from app.services.memory import MemoryManager

@router.post("/api/v1/feedback")
async def submit_feedback(request: FeedbackRequest):
    # 1. Store feedback (existing code)
    feedback_id = await feedback_manager.store_feedback(...)

    # 2. NEW: Auto-extract learnings
    if _is_feedback_substantial(request):
        memory_manager = MemoryManager()

        # Get conversation history for context
        conversation_history = await conversation_manager.get_messages(
            request.conversation_id,
            limit=5
        )

        # Extract and store memories
        memory_ids = await memory_manager.extract_and_store_from_feedback(
            feedback_id=feedback_id,
            conversation_id=request.conversation_id,
            original_query=request.original_query,
            corrected_query=request.corrected_query,
            user_feedback=request.feedback_text,
            user_id=request.user_id,
            schema_group_id=request.schema_group_id,
            conversation_history=conversation_history,
            auto_validate=False  # Require manual review for now
        )

        logger.info(f"Auto-extracted {len(memory_ids)} memories from feedback")

    return {"status": "success", "feedback_id": feedback_id}

def _is_feedback_substantial(request: FeedbackRequest) -> bool:
    """Check if feedback is worth extracting memories from"""
    # Has a correction or detailed feedback
    return (
        request.corrected_query is not None or
        (request.feedback_text and len(request.feedback_text) > 20)
    )
```

## 📊 Monitoring & Analytics

### Get Memory Statistics

```python
# Overall stats
stats = await memory_manager.get_stats()
print(f"Total memories: {stats.total_memories}")
print(f"Active: {stats.active_memories}")
print(f"Avg confidence: {stats.avg_confidence}")
print(f"Success rate: {stats.avg_success_rate}")

# User-specific stats
user_stats = await memory_manager.get_stats(user_id="john@example.com")
print(f"User has {user_stats.user_specific_memories} personal memories")
```

### View Memory by Type

```python
print(f"By type: {stats.memories_by_type}")
# Output: {
#   MemoryType.SYNTAX_CORRECTION: 15,
#   MemoryType.USER_DEFINITION: 8,
#   MemoryType.APPROACH_RECOMMENDATION: 5
# }
```

## 🔧 Maintenance

### Automated Maintenance (Recommended)

Set up a daily cron job to maintain memory quality:

```python
# scripts/memory_maintenance.py
import asyncio
from app.services.memory import MemoryManager

async def run_maintenance():
    memory_manager = MemoryManager()
    await memory_manager.apply_maintenance()

    # This will:
    # 1. Apply temporal decay to old memories
    # 2. Archive low-quality memories
    # 3. Log statistics

if __name__ == "__main__":
    asyncio.run(run_maintenance())
```

Add to crontab:
```bash
# Run daily at 2 AM
0 2 * * * /path/to/python /path/to/scripts/memory_maintenance.py
```

### Manual Maintenance

```python
# Apply temporal decay
decayed_count = await memory_manager.storage.apply_temporal_decay(decay_rate=0.01)

# Archive low-quality memories
archived_count = await memory_manager.storage.archive_low_quality(min_quality=0.3)

# Deactivate specific memory
await memory_manager.deactivate(memory_id)
```

## 🎨 Example: Complete End-to-End Flow

```python
# Day 1: User gives feedback
user_query = "Show me latest trades"
generated_query = "select from trade"
user_feedback = "By 'latest' I mean today's trades only"
corrected_query = "select from trade where date=.z.d"

# Auto-extract and store
memory_ids = await memory_manager.extract_and_store_from_feedback(
    feedback_id=feedback_id,
    conversation_id=conv_id,
    original_query=user_query,
    corrected_query=corrected_query,
    user_feedback=user_feedback,
    user_id="john@example.com"
)
# → Creates memory: "For this user, 'latest' means 'today'"

# Day 5: Same user, new conversation
new_query = "Show me latest orders"

# Retrieve memories
memories = await memory_manager.retrieve(
    query=new_query,
    user_id="john@example.com"
)
# → Finds: "For this user, 'latest' means 'today'" (high similarity)

# Add to prompt
memory_context = memory_manager.format_memories_for_prompt(memories)
enhanced_prompt = f"""
Query: {new_query}

{memory_context}

Generate query:
"""

# LLM generates: select from order where date=.z.d
# ✅ Correct on first try!

# Track success
await memory_manager.mark_helpful(memories[0].memory.id, helpful=True)
# → Confidence increases, will rank higher next time
```

## 🔐 Privacy & Data Management

### User-Specific vs Global Memories

```python
# User-specific (only for this user)
await memory_manager.store(
    memory_type=MemoryType.USER_DEFINITION,
    original_context="Show me VWAP",
    learning="User defines VWAP as sum(price*volume)/sum(volume)",
    user_id="john@example.com"  # ← User-specific
)

# Global (benefits all users)
await memory_manager.store(
    memory_type=MemoryType.SYNTAX_CORRECTION,
    original_context="select from trades",
    learning="Table name is 'trade' (singular)",
    user_id=None  # ← Global
)
```

### Delete User Data (GDPR Compliance)

```python
# Deactivate all memories for a user
async with memory_manager.db_pool.acquire() as conn:
    await conn.execute("""
        UPDATE memory.entries
        SET is_active = false
        WHERE user_id = $1
    """, user_id)
```

## 🧪 Testing

```python
# tests/test_memory_integration.py
import pytest
from app.services.memory import MemoryManager, MemoryType

@pytest.mark.asyncio
async def test_store_and_retrieve():
    manager = MemoryManager()

    # Store
    memory_id = await manager.store(
        memory_type=MemoryType.SYNTAX_CORRECTION,
        original_context="test query",
        learning="test learning"
    )

    # Retrieve
    memories = await manager.retrieve(
        query="test query",
        limit=1
    )

    assert len(memories) == 1
    assert memories[0].memory.id == memory_id
```

## 📝 Configuration

Customize behavior via `MemoryConfig`:

```python
from app.services.memory import MemoryManager
from app.services.memory.memory_types import MemoryConfig

config = MemoryConfig(
    default_limit=10,  # Retrieve top 10 by default
    min_similarity_threshold=0.75,  # Higher threshold
    include_global_memories=True,
    temporal_decay_rate=0.02,  # Faster decay
    min_confidence_threshold=0.4,  # Lower floor
    archive_after_days=60  # Archive after 60 days
)

memory_manager = MemoryManager(config=config)
```

## 🆘 Troubleshooting

### "No memories retrieved"
- Check similarity threshold (lower it to see more results)
- Verify embeddings are being generated
- Check database connectivity

### "Auto-extraction not working"
- Verify LLM provider is configured
- Check logs for extraction errors
- Ensure feedback has sufficient detail

### "Memories not improving over time"
- Call `mark_helpful()` to reinforce good memories
- Run maintenance to decay old memories
- Check confidence scores and success rates

## 🚀 Next Steps

1. ✅ Initialize database schema
2. ✅ Integrate into LangGraph pipeline
3. ✅ Hook up auto-extraction from feedback
4. [ ] Set up maintenance cron job
5. [ ] Build admin UI to review/validate memories (optional)
6. [ ] Monitor effectiveness metrics

## 📚 API Reference

See inline documentation in:
- `memory_manager.py` - Main API
- `memory_types.py` - Data structures
- `memory_storage.py` - Database operations
- `memory_extractor.py` - Auto-extraction logic

## 🤝 Support

Questions? Check:
- This README
- Inline code documentation
- Database schema comments in `init_memory_schema.sql`
- Architecture documentation in `docs/ARCHITECTURE.md`
```