from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Message(BaseModel):
    id: str = Field(..., description="Message ID")
    role: str = Field(..., description="Message role (user or assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class Conversation(BaseModel):
    id: str = Field(..., description="Conversation ID")
    messages: List[Message] = Field(default=[], description="Conversation messages")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


# Share-related schemas
class ShareCreate(BaseModel):
    access_level: str = Field(default="view", description="Access level: 'view' or 'edit'")
    expires_at: Optional[datetime] = Field(default=None, description="Optional expiry datetime")
    shared_with: Optional[str] = Field(default=None, description="Optional specific user ID")


class ShareResponse(BaseModel):
    id: int = Field(..., description="Share ID")
    share_token: str = Field(..., description="Unique share token")
    conversation_id: str = Field(..., description="Conversation ID")
    shared_by: str = Field(..., description="User ID who created the share")
    access_level: str = Field(..., description="Access level: 'view' or 'edit'")
    shared_with: Optional[str] = Field(default=None, description="Specific user ID or None for public")
    is_active: bool = Field(..., description="Whether share is active")
    created_at: datetime = Field(..., description="When share was created")
    expires_at: Optional[datetime] = Field(default=None, description="When share expires")
    access_count: int = Field(default=0, description="Number of times accessed")
    last_accessed_at: Optional[datetime] = Field(default=None, description="Last access time")


class AccessInfo(BaseModel):
    has_access: bool = Field(..., description="Whether user has access")
    access_level: Optional[str] = Field(default=None, description="Access level: 'owner', 'view', or 'edit'")
    can_view: bool = Field(..., description="Whether user can view")
    can_edit: bool = Field(..., description="Whether user can edit")
    can_add_messages: bool = Field(..., description="Whether user can add messages")
    is_owner: bool = Field(..., description="Whether user is the owner")
    error: Optional[str] = Field(default=None, description="Error message if access denied")


class ConversationWithAccess(Conversation):
    access_info: AccessInfo = Field(..., description="Access information for current user")
    shared_by: Optional[str] = Field(default=None, description="User ID who shared (if shared)")
    owner_name: Optional[str] = Field(default=None, description="Owner's name (if available)")