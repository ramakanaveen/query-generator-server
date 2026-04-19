"""
SSE event types and helpers for the v2 agent streaming responses.
Step 1: Defines event types and JSON serialisation.
Step 2: Will add FastAPI StreamingResponse helpers.
"""
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional


class EventType(str, Enum):
    THINKING = "thinking"
    SCHEMA_LOOKUP = "schema_lookup"
    CLARIFICATION_NEEDED = "clarification_needed"
    QUERY_GENERATED = "query_generated"
    EXECUTING = "executing"
    RESULTS = "results"
    ANALYSIS = "analysis"
    SCHEMA_INFO = "schema_info"
    EXPLANATION = "explanation"
    DOWNLOAD_READY = "download_ready"
    ERROR = "error"
    DONE = "done"


@dataclass
class SSEEvent:
    event_type: EventType
    data: dict[str, Any]
    id: Optional[str] = None

    def to_sse_bytes(self) -> bytes:
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        lines.append(f"event: {self.event_type.value}")
        lines.append(f"data: {json.dumps(self.data)}")
        lines.append("")  # blank line terminates SSE event
        return "\n".join(lines).encode("utf-8") + b"\n"

    def to_dict(self) -> dict:
        return {"event": self.event_type.value, "data": self.data}
