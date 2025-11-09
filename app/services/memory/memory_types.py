"""
Memory Module - Type Definitions

Data classes and enums for the memory system.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID


class MemoryType(str, Enum):
    """Types of memories that can be stored"""

    SYNTAX_CORRECTION = "syntax_correction"
    """Corrections to query syntax or structure"""

    USER_DEFINITION = "user_definition"
    """User-specific definitions or interpretations of terms"""

    APPROACH_RECOMMENDATION = "approach_recommendation"
    """Recommended approaches for certain query patterns"""

    QUERY_PATTERN = "query_pattern"
    """Successful query patterns that can be reused"""

    ERROR_CORRECTION = "error_correction"
    """Corrections for common errors"""

    SCHEMA_CLARIFICATION = "schema_clarification"
    """Clarifications about schema structure or meaning"""


class SourceType(str, Enum):
    """Source of the memory"""

    FEEDBACK = "feedback"
    """Extracted from user feedback"""

    CORRECTION = "correction"
    """Derived from user corrections"""

    MANUAL = "manual"
    """Manually added by admin"""

    AUTO_DETECTED = "auto_detected"
    """Automatically detected pattern"""


@dataclass
class Memory:
    """
    Represents a single memory entry.

    This is the core data structure for a memory, containing all
    information about a learning extracted from user interactions.
    """

    # Identity
    id: Optional[UUID] = None

    # Classification
    memory_type: MemoryType = MemoryType.SYNTAX_CORRECTION

    # Scope
    user_id: Optional[str] = None
    """If None, this is a global memory applicable to all users"""

    schema_group_id: Optional[UUID] = None
    """If None, this applies to all schema groups"""

    # Content
    original_context: str = ""
    """The original query or context that triggered this learning"""

    learning_description: str = ""
    """Description of what was learned"""

    corrected_version: Optional[str] = None
    """The corrected/improved version (if applicable)"""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Flexible metadata for additional context"""

    # Provenance
    source_type: SourceType = SourceType.AUTO_DETECTED
    source_conversation_id: Optional[UUID] = None
    source_feedback_id: Optional[UUID] = None

    # Quality metrics
    confidence_score: float = 0.5
    """How confident we are in this memory (0-1)"""

    success_count: int = 0
    """Number of times this memory was helpful"""

    failure_count: int = 0
    """Number of times this memory was not helpful"""

    # Temporal
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0

    # Lifecycle
    is_active: bool = True
    is_validated: bool = False
    """Whether this memory has been validated by an admin"""

    # Tags
    tags: List[str] = field(default_factory=list)

    @property
    def is_user_specific(self) -> bool:
        """Check if this is a user-specific memory"""
        return self.user_id is not None

    @property
    def is_global(self) -> bool:
        """Check if this is a global memory"""
        return self.user_id is None

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Neutral if no data
        return self.success_count / total

    @property
    def quality_score(self) -> float:
        """
        Calculate overall quality score combining:
        - Confidence (40%)
        - Success rate (40%)
        - Recency (20%)
        """
        # Confidence component
        confidence_component = self.confidence_score * 0.4

        # Success rate component
        success_component = self.success_rate * 0.4

        # Recency component (last 90 days)
        if self.created_at:
            age_days = (datetime.utcnow() - self.created_at).days
            recency_component = max(0, 1 - (age_days / 90)) * 0.2
        else:
            recency_component = 0.0

        return confidence_component + success_component + recency_component


@dataclass
class MemoryRetrievalResult:
    """
    Result of a memory retrieval operation.

    Contains the memory plus additional scoring information
    used during retrieval.
    """

    memory: Memory
    """The retrieved memory"""

    similarity_score: float
    """Semantic similarity score (0-1)"""

    combined_score: float
    """Combined ranking score considering similarity, quality, and recency"""

    relevance_explanation: Optional[str] = None
    """Optional explanation of why this memory is relevant"""

    def __lt__(self, other: 'MemoryRetrievalResult') -> bool:
        """Enable sorting by combined score"""
        return self.combined_score < other.combined_score


@dataclass
class MemoryStats:
    """
    Statistics about stored memories.

    Useful for monitoring and analytics.
    """

    total_memories: int = 0
    active_memories: int = 0
    global_memories: int = 0
    user_specific_memories: int = 0

    # By type
    memories_by_type: Dict[MemoryType, int] = field(default_factory=dict)

    # Quality metrics
    avg_confidence: float = 0.0
    avg_success_rate: float = 0.0

    # Usage metrics
    total_access_count: int = 0
    total_success_count: int = 0
    total_failure_count: int = 0

    # Temporal
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None


@dataclass
class MemoryExtractionRequest:
    """
    Request to extract memories from feedback or corrections.
    """

    # Source data
    feedback_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None
    user_id: Optional[str] = None
    schema_group_id: Optional[UUID] = None

    # Content
    original_query: str = ""
    corrected_query: Optional[str] = None
    user_feedback: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Options
    auto_validate: bool = False
    """Automatically mark as validated if confidence is high"""

    min_confidence: float = 0.6
    """Minimum confidence threshold to create memory"""


@dataclass
class MemoryConfig:
    """
    Configuration for memory system behavior.
    """

    # Retrieval settings
    default_limit: int = 5
    """Default number of memories to retrieve"""

    min_similarity_threshold: float = 0.7
    """Minimum similarity score for retrieval"""

    include_global_memories: bool = True
    """Whether to include global memories by default"""

    # Ranking weights
    similarity_weight: float = 0.5
    quality_weight: float = 0.3
    recency_weight: float = 0.2

    # Lifecycle settings
    temporal_decay_rate: float = 0.01
    """Rate at which old memories decay (daily)"""

    min_confidence_threshold: float = 0.3
    """Memories below this confidence are archived"""

    archive_after_days: int = 90
    """Archive unused memories after this many days"""

    # Extraction settings
    auto_extract_from_feedback: bool = True
    extraction_min_confidence: float = 0.6

    def validate(self):
        """Validate configuration values"""
        assert 0 <= self.min_similarity_threshold <= 1
        assert 0 <= self.temporal_decay_rate <= 1
        assert 0 <= self.min_confidence_threshold <= 1
        assert self.default_limit > 0
        assert self.archive_after_days > 0
        # Weights should sum to 1.0
        total_weight = self.similarity_weight + self.quality_weight + self.recency_weight
        assert abs(total_weight - 1.0) < 0.01, f"Weights must sum to 1.0, got {total_weight}"