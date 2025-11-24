"""
Connector factory for database selection.

Provides a unified interface for creating database connectors based on database type.

Supported databases:
- kdb: KDB+/q
- starburst: Starburst/Trino
- trino: Trino (alias for starburst)
- postgres: PostgreSQL (future)
- mysql: MySQL (future)
"""

from typing import Optional, Dict, Any

from app.services.connectors.base_connector import BaseConnector
from app.services.connectors.kdb_connector import KDBConnector
from app.services.connectors.starburst_connector import StarburstConnector
from app.core.logging import logger


def get_connector(
    database_type: str,
    **connection_params
) -> BaseConnector:
    """
    Factory function to get appropriate database connector.

    Args:
        database_type: Type of database ("kdb", "starburst", "trino", "postgres", "mysql")
        **connection_params: Optional connection parameters specific to each database

    Returns:
        Database connector instance (BaseConnector subclass)

    Raises:
        ValueError: If database type is not supported

    Examples:
        # KDB connector
        connector = get_connector("kdb", host="localhost", port=5001)

        # Starburst connector with explicit parameters
        connector = get_connector(
            "starburst",
            host="starburst.example.com",
            port=8080,
            catalog="hive",
            schema="default",
            user="admin",
            password="secret"
        )

        # Trino connector (alias for Starburst)
        connector = get_connector("trino", host="trino.example.com")
    """
    database_type = database_type.lower().strip()

    # KDB+/q
    if database_type == "kdb":
        logger.info("Creating KDB connector")
        return KDBConnector(**connection_params)

    # Starburst/Trino
    elif database_type in ["starburst", "trino"]:
        logger.info(f"Creating Starburst/Trino connector (type: {database_type})")
        return StarburstConnector(**connection_params)

    # PostgreSQL (future implementation)
    elif database_type in ["postgres", "postgresql"]:
        raise NotImplementedError(
            "PostgreSQL connector not yet implemented. "
            "Coming soon! For now, use Starburst with PostgreSQL catalog."
        )

    # MySQL (future implementation)
    elif database_type == "mysql":
        raise NotImplementedError(
            "MySQL connector not yet implemented. "
            "Coming soon! For now, use Starburst with MySQL catalog."
        )

    # SQLite (future implementation)
    elif database_type == "sqlite":
        raise NotImplementedError(
            "SQLite connector not yet implemented. "
            "Coming soon!"
        )

    # Unsupported database
    else:
        supported_types = ["kdb", "starburst", "trino", "postgres (future)", "mysql (future)"]
        raise ValueError(
            f"Unsupported database type: '{database_type}'. "
            f"Supported types: {', '.join(supported_types)}"
        )


def get_available_connectors() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available connectors.

    Returns:
        Dictionary with connector information

    Example:
        {
            "kdb": {
                "class": "KDBConnector",
                "status": "available",
                "description": "KDB+/q database"
            },
            "starburst": {
                "class": "StarburstConnector",
                "status": "available",
                "description": "Starburst/Trino distributed SQL"
            },
            ...
        }
    """
    connectors = {
        "kdb": {
            "class": "KDBConnector",
            "status": "available",
            "description": "KDB+/q time-series database",
            "features": ["pagination", "multi-statement", "count_caching"]
        },
        "starburst": {
            "class": "StarburstConnector",
            "status": "available",
            "description": "Starburst/Trino distributed SQL query engine",
            "aliases": ["trino"],
            "features": ["pagination", "sql_standard", "distributed"]
        },
        "postgres": {
            "class": "PostgresConnector",
            "status": "planned",
            "description": "PostgreSQL relational database",
            "note": "Use Starburst with PostgreSQL catalog for now"
        },
        "mysql": {
            "class": "MySQLConnector",
            "status": "planned",
            "description": "MySQL relational database",
            "note": "Use Starburst with MySQL catalog for now"
        }
    }

    return connectors
