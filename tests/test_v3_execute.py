"""
Test suite for /v3/execute endpoint

Tests the new universal database execution endpoint with connector factory.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.main import app

client = TestClient(app)


class TestV3ExecuteEndpoint:
    """Test cases for /api/v1/execute/v3 endpoint"""

    def test_v3_execute_endpoint_exists(self):
        """Verify that /v3/execute endpoint exists"""
        # Test with invalid payload to check endpoint exists (will get 422, not 404)
        response = client.post("/api/v1/execute/v3", json={})
        assert response.status_code != 404, "V3 execute endpoint should exist"

    @patch('app.services.connectors.kdb_connector.KDBConnector.execute_paginated')
    async def test_v3_execute_kdb_basic(self, mock_execute):
        """Test basic KDB query execution via v3 endpoint"""
        # Mock the execute_paginated method
        mock_execute.return_value = (
            [{"col1": "value1", "col2": 100}],  # results
            {"row_count": 1, "execution_time": 0.1},  # metadata
            1  # total_count
        )

        response = client.post("/api/v1/execute/v3", json={
            "query": "select from table",
            "execution_id": "test-123",
            "database_type": "kdb",
            "pagination": {"page": 0, "page_size": 100}
        })

        # Should return 200 or 503 (if KDB not available)
        assert response.status_code in [200, 503]

    def test_v3_execute_unsupported_database(self):
        """Test v3 endpoint with unsupported database type"""
        response = client.post("/api/v1/execute/v3", json={
            "query": "SELECT * FROM users",
            "execution_id": "test-456",
            "database_type": "unsupported_db",
            "pagination": {"page": 0, "page_size": 100}
        })

        # Should return 400 for unsupported database
        assert response.status_code == 400
        assert "unsupported" in response.json()["detail"].lower()

    def test_v3_execute_starburst_structure(self):
        """Test v3 endpoint accepts starburst database type"""
        response = client.post("/api/v1/execute/v3", json={
            "query": "SELECT * FROM users LIMIT 10",
            "execution_id": "test-789",
            "database_type": "starburst",
            "pagination": {"page": 0, "page_size": 10}
        })

        # Should not return 400 for database type (might be 503 if not configured)
        assert response.status_code != 400 or "unsupported database type" not in response.json()["detail"].lower()

    def test_v3_execute_with_connection_params(self):
        """Test v3 endpoint with custom connection parameters"""
        response = client.post("/api/v1/execute/v3", json={
            "query": "select from table",
            "execution_id": "test-connection-params",
            "database_type": "kdb",
            "connection_params": {
                "host": "custom-host",
                "port": 5002
            },
            "pagination": {"page": 0, "page_size": 100}
        })

        # Endpoint should accept connection_params without validation error
        # Might fail on connection (503) but not on schema validation (422)
        assert response.status_code != 422

    def test_v3_execute_query_complexity_kdb(self):
        """Test v3 endpoint with query_complexity for KDB"""
        response = client.post("/api/v1/execute/v3", json={
            "query": "select from table",
            "execution_id": "test-complexity",
            "database_type": "kdb",
            "query_complexity": "SINGLE_LINE",
            "pagination": {"page": 0, "page_size": 50}
        })

        # Should not fail on validation
        assert response.status_code != 422

    def test_v3_execute_pagination_validation(self):
        """Test v3 endpoint validates pagination parameters"""
        # Test invalid page number
        response = client.post("/api/v1/execute/v3", json={
            "query": "SELECT * FROM users",
            "execution_id": "test-invalid-page",
            "database_type": "starburst",
            "pagination": {"page": -1, "page_size": 100}
        })
        assert response.status_code == 400

        # Test invalid page size
        response = client.post("/api/v1/execute/v3", json={
            "query": "SELECT * FROM users",
            "execution_id": "test-invalid-size",
            "database_type": "starburst",
            "pagination": {"page": 0, "page_size": 20000}
        })
        assert response.status_code == 400


def test_connector_factory_integration():
    """Test that connector factory is properly integrated"""
    from app.services.connectors import get_connector

    # Test KDB connector creation
    kdb_connector = get_connector("kdb")
    assert kdb_connector is not None
    assert kdb_connector.__class__.__name__ == "KDBConnector"

    # Test Starburst connector creation
    starburst_connector = get_connector("starburst")
    assert starburst_connector is not None
    assert starburst_connector.__class__.__name__ == "StarburstConnector"

    # Test Trino alias
    trino_connector = get_connector("trino")
    assert trino_connector is not None
    assert trino_connector.__class__.__name__ == "StarburstConnector"


def test_execution_request_schema():
    """Test that ExecutionRequest schema includes all required fields"""
    from app.schemas.query import ExecutionRequest

    # Create a valid request
    request = ExecutionRequest(
        query="SELECT * FROM table",
        execution_id="test-schema",
        database_type="starburst",
        connection_params={"host": "localhost"},
        query_complexity="SINGLE_LINE"
    )

    assert request.query == "SELECT * FROM table"
    assert request.database_type == "starburst"
    assert request.connection_params == {"host": "localhost"}
    assert request.query_complexity == "SINGLE_LINE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
