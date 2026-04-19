"""
Clarification tool — pause/resume state for vague-query flow.
Stores the in-progress message history so the agent can resume after the user answers.
"""
from __future__ import annotations

import uuid

# In-memory store: clarification_id → {"messages": [...], "request_snapshot": {...}}
# Can be backed by Redis later.
_store: dict[str, dict] = {}


def save_context(messages: list[dict], request_snapshot: dict) -> str:
    """Save agent message history and return a clarification_id."""
    cid = str(uuid.uuid4())
    _store[cid] = {"messages": list(messages), "request_snapshot": request_snapshot}
    return cid


def load_context(clarification_id: str) -> dict | None:
    """Load saved context. Returns None if id is unknown or expired."""
    return _store.get(clarification_id)


def delete_context(clarification_id: str) -> None:
    _store.pop(clarification_id, None)
