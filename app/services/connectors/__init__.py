"""
Database connectors package.

Provides a unified interface for executing queries across different database types:
- KDB+/q
- Starburst/Trino
- PostgreSQL
- MySQL
- SQLite

Usage:
    from app.services.connectors import get_connector

    connector = get_connector("starburst")
    results, metadata = await connector.execute("SELECT * FROM users")
"""

from app.services.connectors.connector_factory import get_connector

__all__ = ["get_connector"]
