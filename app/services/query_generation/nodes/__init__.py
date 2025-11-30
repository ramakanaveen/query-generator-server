# app/services/query_generation/nodes/__init__.py

"""
Unified query generation nodes with thinking/reasoning model.

New Simplified Architecture (v3):
1. intent_classifier - LLM-based intent classification
2. schema_retriever - Enhanced schema retrieval with context awareness
3. unified_analyzer_generator - Combined analysis + generation in single LLM call
4. kdb_validator / sql_validator - LLM-based validators (database-specific)
5. schema_description_node - Schema information generation

The workflow follows this simplified pattern:
intent_classifier → schema_retriever → unified_analyzer_generator → validator (KDB/SQL)
                                                      ↑                    ↓
                                                      └── (retry on fail) ─┘
"""

# Import all the nodes for the unified workflow
from . import intent_classifier
from . import schema_retriever
from . import schema_description_node
from . import initial_processor
from . import enhanced_schema_retriever

# Import unified nodes (v3)
from .unified_analyzer_generator import unified_analyze_and_generate
from .kdb_validator import validate_kdb_query
from .sql_validator import validate_sql_query

# Legacy imports for backward compatibility
from . import intelligent_analyzer  # v2 - will be deprecated
from . import query_generator_node  # v2 - will be deprecated
from . import query_validator  # v2 - will be deprecated
from . import query_refiner  # v2 - will be deprecated
from . import query_analyzer  # v1 - deprecated
from . import unified_query_analyzer  # v1 - deprecated

__all__ = [
    # v3 unified nodes
    'unified_analyze_and_generate',
    'validate_kdb_query',
    'validate_sql_query',

    # Core nodes (used across versions)
    'intent_classifier',
    'schema_retriever',
    'enhanced_schema_retriever',
    'schema_description_node',
    'initial_processor',

    # v2 nodes (for backward compatibility)
    'intelligent_analyzer',
    'query_generator_node',
    'query_validator',
    'query_refiner',

    # v1 nodes (deprecated)
    'query_analyzer',
    'unified_query_analyzer'
]