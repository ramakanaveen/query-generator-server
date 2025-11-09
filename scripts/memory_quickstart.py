#!/usr/bin/env python3
"""
Memory Module Quick Start

Run this script to test the memory system and see it in action.

Usage:
    python scripts/memory_quickstart.py
"""

import asyncio
import logging
from uuid import uuid4

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Demonstrate memory module functionality"""

    print("=" * 70)
    print("MEMORY MODULE QUICK START")
    print("=" * 70)

    try:
        from app.services.memory import MemoryManager, MemoryType

        # Initialize
        print("\n1. Initializing Memory Manager...")
        memory_manager = MemoryManager()
        print("✅ Memory Manager initialized")

        # Get initial stats
        print("\n2. Getting current stats...")
        stats = await memory_manager.get_stats()
        print(f"   Total memories: {stats.total_memories}")
        print(f"   Active memories: {stats.active_memories}")
        print(f"   Global memories: {stats.global_memories}")
        print(f"   User-specific: {stats.user_specific_memories}")

        # Store a global syntax correction
        print("\n3. Storing a global syntax correction...")
        memory_id_1 = await memory_manager.store(
            memory_type=MemoryType.SYNTAX_CORRECTION,
            original_context="select from trades where sym=`AAPL",
            learning="Table name should be 'trade' (singular), not 'trades' (plural) in KDB/Q schema",
            corrected_version="select from trade where sym=`AAPL",
            confidence_score=0.95,
            tags=["table_name", "syntax", "kdb", "common_mistake"],
            metadata={
                "table": "trade",
                "common_mistake": "using plural form"
            }
        )
        print(f"✅ Stored memory: {memory_id_1}")

        # Store a user-specific definition
        print("\n4. Storing a user-specific definition...")
        memory_id_2 = await memory_manager.store(
            memory_type=MemoryType.USER_DEFINITION,
            original_context="Show me VWAP for AAPL",
            learning="User defines VWAP as: sum(price * volume) % sum(volume)",
            corrected_version="select sym, sum(price*size)%sum size from trade where sym=`AAPL",
            user_id="demo_user@example.com",
            confidence_score=0.8,
            tags=["VWAP", "calculation", "user_preference"],
            metadata={
                "term": "VWAP",
                "formula": "sum(price*volume)%sum(volume)"
            }
        )
        print(f"✅ Stored user memory: {memory_id_2}")

        # Store an approach recommendation
        print("\n5. Storing an approach recommendation...")
        memory_id_3 = await memory_manager.store(
            memory_type=MemoryType.APPROACH_RECOMMENDATION,
            original_context="Queries with large time ranges and aggregations",
            learning="For date-range queries with aggregations, always filter by date first before calculating aggregates to improve performance",
            confidence_score=0.9,
            tags=["performance", "optimization", "time_series"],
            metadata={
                "optimization": "filter_before_aggregate",
                "applies_to": "time_series_with_aggregations"
            }
        )
        print(f"✅ Stored approach memory: {memory_id_3}")

        # Retrieve memories for a query
        print("\n6. Retrieving memories for: 'Show me trades for AAPL'...")
        memories = await memory_manager.retrieve(
            query="Show me trades for AAPL",
            user_id="demo_user@example.com",
            limit=5
        )
        print(f"   Found {len(memories)} relevant memories:")
        for i, result in enumerate(memories, 1):
            mem = result.memory
            print(f"\n   {i}. Type: {mem.memory_type.value}")
            print(f"      Learning: {mem.learning_description[:80]}...")
            print(f"      Confidence: {mem.confidence_score:.2f}")
            print(f"      Similarity: {result.similarity_score:.2f}")
            print(f"      User-specific: {mem.is_user_specific}")

        # Format for prompt
        print("\n7. Formatting memories for LLM prompt...")
        memory_context = memory_manager.format_memories_for_prompt(memories)
        print("   Formatted context:")
        print("   " + "-" * 66)
        for line in memory_context.split('\n')[:15]:  # First 15 lines
            print(f"   {line}")
        print("   ... (truncated)")
        print("   " + "-" * 66)

        # Mark a memory as helpful
        print("\n8. Marking memory as helpful...")
        if memories:
            await memory_manager.mark_helpful(memories[0].memory.id, helpful=True)
            print(f"✅ Marked memory {memories[0].memory.id} as helpful")

            # Retrieve it again to see updated stats
            updated_memory = await memory_manager.get_by_id(memories[0].memory.id)
            print(f"   Success count: {updated_memory.success_count}")
            print(f"   Updated confidence: {updated_memory.confidence_score:.2f}")

        # Get updated stats
        print("\n9. Getting updated stats...")
        updated_stats = await memory_manager.get_stats()
        print(f"   Total memories: {updated_stats.total_memories}")
        print(f"   By type:")
        for mem_type, count in updated_stats.memories_by_type.items():
            print(f"     - {mem_type.value}: {count}")

        # Test auto-extraction
        print("\n10. Testing auto-extraction from feedback...")
        print("    (This uses LLM to analyze feedback)")

        try:
            extracted_ids = await memory_manager.extract_and_store_from_feedback(
                feedback_id=uuid4(),
                conversation_id=uuid4(),
                original_query="select top 10 from trades",
                corrected_query="select[10] from trade",
                user_feedback="Use select[N] syntax instead of 'select top N' in KDB/Q. Also, table name is 'trade' not 'trades'.",
                user_id="demo_user@example.com",
                conversation_history=[
                    {"role": "user", "content": "Show me top 10 trades"},
                    {"role": "assistant", "content": "select top 10 from trades"}
                ]
            )

            print(f"✅ Auto-extracted {len(extracted_ids)} memories")
            for mem_id in extracted_ids:
                mem = await memory_manager.get_by_id(mem_id)
                print(f"   - {mem.memory_type.value}: {mem.learning_description[:60]}...")

        except Exception as e:
            print(f"⚠️  Auto-extraction skipped (requires LLM): {e}")

        print("\n" + "=" * 70)
        print("QUICK START COMPLETE!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Check the database: SELECT * FROM memory.entries;")
        print("2. Integrate into your query pipeline")
        print("3. Hook up auto-extraction from feedback")
        print("4. Set up maintenance cron job")
        print("\nSee app/services/memory/README.md for full documentation")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Error during quick start: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure database schema is initialized: psql -f scripts/init_memory_schema.sql")
        print("2. Check database connection in .env")
        print("3. Verify all dependencies are installed")


if __name__ == "__main__":
    asyncio.run(main())