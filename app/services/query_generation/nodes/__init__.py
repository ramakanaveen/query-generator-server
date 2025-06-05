# app/services/query_generation/nodes/__init__.py

"""
Enhanced query generation nodes with LLM-driven decision making.

New Architecture:
1. intent_classifier - LLM-based intent classification
2. schema_retriever - Enhanced schema retrieval with context awareness
3. intelligent_analyzer - Schema-aware complexity and execution planning
4. query_generator_node - Context-aware query generation
5. query_validator - Enhanced validation with escalation triggers
6. query_refiner - Legacy refinement for fallback scenarios
7. schema_description_node - Schema information generation

The workflow follows this enhanced pattern:
intent_classifier → schema_retriever → intelligent_analyzer → query_generator → validator
                                           ↑                                      ↓
                                           └── (LLM feedback analysis) ──────────┘
"""

# Import all the nodes for the enhanced workflow
from . import intent_classifier
from . import schema_retriever
from . import intelligent_analyzer
from . import query_generator_node
from . import query_validator
from . import query_refiner
from . import schema_description_node

# Legacy imports for backward compatibility
from . import query_analyzer  # Will be deprecated
from . import unified_query_analyzer  # Will be deprecated

__all__ = [
    # New enhanced nodes
    'intent_classifier',
    'intelligent_analyzer',

    # Enhanced existing nodes
    'schema_retriever',
    'query_generator_node',
    'query_validator',

    # Existing nodes
    'query_refiner',
    'schema_description_node',

    # Legacy nodes (for backward compatibility)
    'query_analyzer',
    'unified_query_analyzer'
]