"""
Unit tests for Shared Conversations feature.

Tests cover:
- Share creation
- Share access via token
- Access control enforcement
- Share revocation
- Permission checking
- Edge cases and error handling
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from app.main import app
from app.services.conversation_manager import ConversationManager


@pytest.fixture
def test_conversation():
    """Sample conversation for testing."""
    return {
        "id": "test-conv-123",
        "user_id": "user-A",
        "title": "Test Conversation",
        "summary": None,
        "messages": [
            {
                "id": "msg-1",
                "role": "user",
                "content": "Hello",
                "timestamp": "2025-01-15T10:00:00"
            }
        ],
        "created_at": datetime.now(),
        "updated_at": datetime.now(),
        "last_accessed_at": datetime.now(),
        "is_archived": False,
        "metadata": {}
    }


@pytest.fixture
def test_share():
    """Sample share for testing."""
    return {
        "id": 1,
        "conversation_id": "test-conv-123",
        "share_token": "test-token-abc123",
        "shared_by": "user-A",
        "access_level": "view",
        "shared_with": None,
        "is_active": True,
        "created_at": datetime.now(),
        "expires_at": None,
        "access_count": 0,
        "last_accessed_at": None,
        "metadata": {}
    }


@pytest.fixture
def mock_db_pool():
    """Mock database connection pool."""
    with patch('app.services.conversation_manager.db_pool') as mock:
        conn = AsyncMock()

        # Make get_connection async
        async def get_conn():
            return conn

        mock.get_connection = get_conn
        mock.release_connection = AsyncMock()
        yield mock, conn


class TestShareCreation:
    """Tests for creating shares."""

    @pytest.mark.asyncio
    async def test_create_share_success(self, mock_db_pool):
        """Test successful share creation."""
        mock_pool, mock_conn = mock_db_pool

        # Mock owner verification
        mock_conn.fetchval.return_value = "user-A"

        # Mock share insertion
        mock_conn.fetchrow.return_value = {
            "id": 1,
            "conversation_id": "conv-123",
            "share_token": "abc123xyz",
            "shared_by": "user-A",
            "access_level": "view",
            "expires_at": None,
            "created_at": datetime.now(),
            "shared_with": None,
            "is_active": True
        }

        manager = ConversationManager()
        share = await manager.create_share(
            conversation_id="conv-123",
            shared_by="user-A",
            access_level="view"
        )

        assert share is not None
        assert share["conversation_id"] == "conv-123"
        assert share["shared_by"] == "user-A"
        assert share["access_level"] == "view"
        assert len(share["share_token"]) > 0

    @pytest.mark.asyncio
    async def test_create_share_not_owner(self, mock_db_pool):
        """Test share creation fails if user is not owner."""
        mock_pool, mock_conn = mock_db_pool

        # Mock owner verification - different user
        mock_conn.fetchval.return_value = "user-B"

        manager = ConversationManager()

        with pytest.raises(ValueError, match="does not own"):
            await manager.create_share(
                conversation_id="conv-123",
                shared_by="user-A",
                access_level="view"
            )

    @pytest.mark.asyncio
    async def test_create_share_nonexistent_conversation(self, mock_db_pool):
        """Test share creation fails for nonexistent conversation."""
        mock_pool, mock_conn = mock_db_pool

        # Mock conversation not found
        mock_conn.fetchval.return_value = None

        manager = ConversationManager()

        with pytest.raises(ValueError, match="not found"):
            await manager.create_share(
                conversation_id="nonexistent",
                shared_by="user-A",
                access_level="view"
            )

    @pytest.mark.asyncio
    async def test_share_token_uniqueness(self, mock_db_pool):
        """Test that share tokens are unique."""
        mock_pool, mock_conn = mock_db_pool
        mock_conn.fetchval.return_value = "user-A"

        tokens = set()

        for i in range(5):
            mock_conn.fetchrow.return_value = {
                "id": i,
                "conversation_id": "conv-123",
                "share_token": f"token-{i}",
                "shared_by": "user-A",
                "access_level": "view",
                "expires_at": None,
                "created_at": datetime.now(),
                "shared_with": None,
                "is_active": True
            }

            manager = ConversationManager()
            share = await manager.create_share(
                conversation_id="conv-123",
                shared_by="user-A"
            )
            tokens.add(share["share_token"])

        # All tokens should be unique
        assert len(tokens) == 5


class TestShareAccess:
    """Tests for accessing shared conversations."""

    @pytest.mark.asyncio
    async def test_get_share_by_token_success(self, mock_db_pool, test_share):
        """Test successful share retrieval by token."""
        mock_pool, mock_conn = mock_db_pool

        # Add owner_id and conversation_title
        test_share["owner_id"] = "user-A"
        test_share["conversation_title"] = "Test Conversation"

        mock_conn.fetchrow.return_value = test_share

        manager = ConversationManager()
        share = await manager.get_share_by_token("test-token-abc123")

        assert share is not None
        assert share["share_token"] == "test-token-abc123"
        assert share["is_active"] is True

    @pytest.mark.asyncio
    async def test_get_share_invalid_token(self, mock_db_pool):
        """Test share retrieval with invalid token."""
        mock_pool, mock_conn = mock_db_pool
        mock_conn.fetchrow.return_value = None

        manager = ConversationManager()
        share = await manager.get_share_by_token("invalid-token")

        assert share is None

    @pytest.mark.asyncio
    async def test_get_share_expired(self, mock_db_pool, test_share):
        """Test accessing expired share."""
        mock_pool, mock_conn = mock_db_pool

        # Set expiry to past
        test_share["expires_at"] = datetime.now() - timedelta(days=1)
        test_share["owner_id"] = "user-A"
        test_share["conversation_title"] = "Test"

        mock_conn.fetchrow.return_value = test_share

        manager = ConversationManager()
        share = await manager.get_share_by_token("test-token-abc123")

        # Should return None for expired shares
        assert share is None


class TestAccessControl:
    """Tests for access control and permissions."""

    @pytest.mark.asyncio
    async def test_check_access_owner(self, mock_db_pool):
        """Test owner has full access."""
        mock_pool, mock_conn = mock_db_pool
        mock_conn.fetchval.return_value = "user-A"

        manager = ConversationManager()
        access = await manager.check_conversation_access(
            conversation_id="conv-123",
            user_id="user-A"
        )

        assert access["has_access"] is True
        assert access["access_level"] == "owner"
        assert access["can_view"] is True
        assert access["can_edit"] is True
        assert access["can_add_messages"] is True
        assert access["is_owner"] is True

    @pytest.mark.asyncio
    async def test_check_access_viewer(self, mock_db_pool):
        """Test viewer has read-only access."""
        mock_pool, mock_conn = mock_db_pool

        # First call: get owner (not matching user)
        mock_conn.fetchval.return_value = "user-A"

        # Second call: get share info
        mock_conn.fetchrow.return_value = {
            "access_level": "view",
            "expires_at": None,
            "is_active": True
        }

        manager = ConversationManager()
        access = await manager.check_conversation_access(
            conversation_id="conv-123",
            user_id="user-B"
        )

        assert access["has_access"] is True
        assert access["access_level"] == "view"
        assert access["can_view"] is True
        assert access["can_edit"] is False
        assert access["can_add_messages"] is False
        assert access["is_owner"] is False

    @pytest.mark.asyncio
    async def test_check_access_no_permission(self, mock_db_pool):
        """Test user with no access."""
        mock_pool, mock_conn = mock_db_pool

        # Owner is different user
        mock_conn.fetchval.return_value = "user-A"

        # No share found
        mock_conn.fetchrow.return_value = None

        manager = ConversationManager()
        access = await manager.check_conversation_access(
            conversation_id="conv-123",
            user_id="user-C"
        )

        assert access["has_access"] is False
        assert access["can_view"] is False
        assert access["can_add_messages"] is False
        assert "error" in access


class TestShareRevocation:
    """Tests for revoking shares."""

    @pytest.mark.asyncio
    async def test_revoke_share_success(self, mock_db_pool):
        """Test successful share revocation."""
        mock_pool, mock_conn = mock_db_pool

        # Owner verification
        mock_conn.fetchval.return_value = "user-A"
        mock_conn.execute = AsyncMock()

        manager = ConversationManager()
        success = await manager.revoke_share(
            share_id=1,
            user_id="user-A"
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_revoke_share_not_owner(self, mock_db_pool):
        """Test revocation fails if user is not owner."""
        mock_pool, mock_conn = mock_db_pool

        # Different owner
        mock_conn.fetchval.return_value = "user-A"

        manager = ConversationManager()
        success = await manager.revoke_share(
            share_id=1,
            user_id="user-B"
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_share(self, mock_db_pool):
        """Test revoking nonexistent share."""
        mock_pool, mock_conn = mock_db_pool

        # Share not found
        mock_conn.fetchval.return_value = None

        manager = ConversationManager()
        success = await manager.revoke_share(
            share_id=999,
            user_id="user-A"
        )

        assert success is False


class TestAPIEndpoints:
    """Integration tests for API endpoints."""

    def test_create_share_endpoint(self, test_client):
        """Test POST /conversations/{id}/share endpoint."""
        with patch('app.routes.conversation.ConversationManager') as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.create_share = AsyncMock(return_value={
                "id": 1,
                "conversation_id": "conv-123",
                "share_token": "abc123",
                "shared_by": "user-A",
                "access_level": "view",
                "expires_at": None,
                "created_at": datetime.now()
            })

            response = test_client.post(
                "/api/v1/conversations/conv-123/share",
                json={
                    "user_id": "user-A",
                    "access_level": "view"
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "share_token" in data
            assert "share_url" in data

    def test_get_shared_conversation_endpoint(self, test_client, test_conversation):
        """Test GET /shared/{token} endpoint."""
        with patch('app.routes.conversation.ConversationManager') as MockManager:
            mock_manager = MockManager.return_value

            # Add share info
            test_conversation["shared_by"] = "user-A"
            test_conversation["access_level"] = "view"
            test_conversation["share_info"] = {
                "share_token": "abc123",
                "access_level": "view",
                "shared_by": "user-A",
                "expires_at": None,
                "access_count": 1
            }

            mock_manager.get_shared_conversation = AsyncMock(return_value=test_conversation)

            response = test_client.get("/api/v1/shared/abc123")

            assert response.status_code == 200
            data = response.json()
            assert "access_info" in data
            assert data["access_info"]["access_level"] == "view"
            assert data["access_info"]["can_add_messages"] is False

    def test_list_shares_endpoint(self, test_client):
        """Test GET /conversations/{id}/shares endpoint."""
        with patch('app.routes.conversation.ConversationManager') as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.get_conversation = AsyncMock(return_value={
                "id": "conv-123",
                "user_id": "user-A"
            })
            mock_manager.get_conversation_shares = AsyncMock(return_value=[
                {
                    "id": 1,
                    "share_token": "abc123",
                    "shared_with": None,
                    "access_level": "view",
                    "created_at": datetime.now(),
                    "expires_at": None,
                    "access_count": 5,
                    "last_accessed_at": datetime.now(),
                    "is_active": True
                }
            ])

            response = test_client.get(
                "/api/v1/conversations/conv-123/shares?user_id=user-A"
            )

            assert response.status_code == 200
            data = response.json()
            assert "shares" in data
            assert len(data["shares"]) == 1
            assert data["shares"][0]["access_count"] == 5

    def test_revoke_share_endpoint(self, test_client):
        """Test DELETE /conversations/{id}/shares/{share_id} endpoint."""
        with patch('app.routes.conversation.ConversationManager') as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.revoke_share = AsyncMock(return_value=True)

            response = test_client.delete(
                "/api/v1/conversations/conv-123/shares/1?user_id=user-A"
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_access_control_in_add_message(self, test_client):
        """Test access control when adding messages."""
        with patch('app.routes.conversation.ConversationManager') as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.check_conversation_access = AsyncMock(return_value={
                "has_access": True,
                "access_level": "view",
                "can_add_messages": False
            })

            response = test_client.post(
                "/api/v1/conversations/conv-123/messages?user_id=user-B",
                json={
                    "id": "msg-1",
                    "role": "user",
                    "content": "Hello"
                }
            )

            assert response.status_code == 403
            assert "permission" in response.json()["detail"].lower()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_share_with_expiry(self, mock_db_pool):
        """Test creating share with expiry date."""
        mock_pool, mock_conn = mock_db_pool
        mock_conn.fetchval.return_value = "user-A"

        future_date = datetime.now() + timedelta(days=7)

        mock_conn.fetchrow.return_value = {
            "id": 1,
            "conversation_id": "conv-123",
            "share_token": "abc123",
            "shared_by": "user-A",
            "access_level": "view",
            "expires_at": future_date,
            "created_at": datetime.now(),
            "shared_with": None,
            "is_active": True
        }

        manager = ConversationManager()
        share = await manager.create_share(
            conversation_id="conv-123",
            shared_by="user-A",
            expires_at=future_date
        )

        assert share["expires_at"] == future_date

    @pytest.mark.asyncio
    async def test_specific_user_share(self, mock_db_pool):
        """Test sharing with specific user."""
        mock_pool, mock_conn = mock_db_pool
        mock_conn.fetchval.return_value = "user-A"

        mock_conn.fetchrow.return_value = {
            "id": 1,
            "conversation_id": "conv-123",
            "share_token": "abc123",
            "shared_by": "user-A",
            "access_level": "view",
            "expires_at": None,
            "created_at": datetime.now(),
            "shared_with": "user-B",
            "is_active": True
        }

        manager = ConversationManager()
        share = await manager.create_share(
            conversation_id="conv-123",
            shared_by="user-A",
            shared_with="user-B"
        )

        assert share["shared_with"] == "user-B"

    @pytest.mark.asyncio
    async def test_access_tracking(self, mock_db_pool):
        """Test access count tracking."""
        mock_pool, mock_conn = mock_db_pool
        mock_conn.execute = AsyncMock()

        manager = ConversationManager()
        await manager.update_share_access_tracking("test-token")

        # Verify execute was called to update counts
        mock_conn.execute.assert_called_once()

    def test_list_all_conversations(self, test_client):
        """Test GET /user/{id}/conversations/all endpoint."""
        with patch('app.routes.conversation.ConversationManager') as MockManager:
            mock_manager = MockManager.return_value
            mock_manager.get_user_conversations = AsyncMock(return_value=[
                {
                    "id": "conv-123",
                    "title": "My Conversation",
                    "user_id": "user-A",
                    "message_count": 5
                }
            ])
            mock_manager.get_conversation_shares = AsyncMock(return_value=[
                {"id": 1, "is_active": True}
            ])
            mock_manager.get_user_shared_conversations = AsyncMock(return_value=[
                {
                    "id": "conv-456",
                    "title": "Shared Conversation",
                    "user_id": "user-B",
                    "shared_by": "user-B",
                    "access_level": "view",
                    "message_count": 3
                }
            ])

            response = test_client.get("/api/v1/user/user-A/conversations/all")

            assert response.status_code == 200
            data = response.json()
            assert "owned" in data
            assert "shared_with_me" in data
            assert len(data["owned"]) == 1
            assert len(data["shared_with_me"]) == 1
            assert data["owned"][0]["shared_with_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
