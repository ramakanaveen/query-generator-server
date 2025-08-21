# app/services/enhanced_schema_service.py - COMPLETE FILE

from typing import Dict, List, Any, Optional, Tuple
import time
from dataclasses import dataclass
from app.core.db import db_pool
from app.core.logging import logger
from app.services.embedding_provider import EmbeddingProvider
from app.services.feedback_manager import FeedbackManager


@dataclass
class SchemaRetrievalConfig:
    """Configuration for schema retrieval operations."""
    vector_similarity_threshold: float = 0.4
    max_tables: int = 5
    max_schema_examples: int = 5
    max_verified_examples: int = 3
    max_total_examples: int = 8
    include_user_examples: bool = True
    include_schema_examples: bool = True
    similarity_boost_user_examples: float = 0.1


@dataclass
class SchemaDescriptionConfig:
    """Configuration for schema description retrieval."""
    detail_level: str = "standard"
    include_examples: bool = False
    include_relationships: bool = True
    max_tables: int = 20
    max_columns_per_table: int = 50


@dataclass
class SchemaExample:
    """Unified example structure."""
    id: str
    natural_language: str
    generated_query: str
    description: str
    source_type: str
    relevance_score: float
    table_names: List[str]
    is_user_specific: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class SchemaRetrievalResult:
    """Complete schema retrieval result."""
    schema_structure: Dict[str, Any]
    examples: List[SchemaExample]
    retrieval_metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]


@dataclass
class SchemaDescriptionResult:
    """Result for schema description retrieval."""
    schema_structure: Dict[str, Any]
    metadata: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class EnhancedSchemaService:
    """
    High-performance schema retrieval service that consolidates:
    1. Vector-based table discovery
    2. Schema structure retrieval
    3. Schema examples from relevant tables
    4. User verified queries
    5. Intelligent relevance ranking
    6. Schema description generation

    Designed for both workflow nodes and REST API endpoints.
    Performance optimized with embedding reuse and smart caching.
    """

    def __init__(self):
        self.embedding_provider = EmbeddingProvider()
        self.feedback_manager = FeedbackManager()

        # Enhanced caching system
        self._result_cache = {}
        self._embedding_cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cleanup = 0

    # ========== MAIN ENTRY POINTS ==========

    async def retrieve_schema_with_examples(
            self,
            query_text: str,
            directives: List[str] = None,
            entities: List[str] = None,
            user_id: Optional[str] = None,
            config: SchemaRetrievalConfig = None
    ) -> SchemaRetrievalResult:
        """
        Main entry point: Retrieve schema and examples for query generation.

        Strategy Implementation:
        1. Vector search to find relevant tables based on NL query
        2. Get schema examples from those specific tables
        3. Get user verified queries with similarity matching
        4. Rank and combine all examples by relevance
        """
        start_time = time.time()
        config = config or SchemaRetrievalConfig()

        # Check cache first
        cache_key = self._build_cache_key(query_text, directives, entities, user_id, config)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            logger.info("📦 Returning cached schema result")
            return cached_result

        try:
            # Step 1: Get or generate query embedding (with caching)
            query_embedding = await self._get_or_generate_query_embedding(query_text, directives, entities)
            if not query_embedding:
                return self._create_empty_result("Failed to generate query embedding", start_time)

            # Step 2: Vector search for relevant tables using existing embeddings
            relevant_tables = await self._find_relevant_tables_optimized(
                query_embedding=query_embedding,
                query_text=query_text,
                directives=directives,
                config=config
            )

            if not relevant_tables:
                result = self._create_empty_result("No relevant tables found", start_time)
                self._cache_result(cache_key, result)
                return result

            # Step 3: Build schema structure from relevant tables
            schema_structure = await self._build_schema_structure(relevant_tables)

            # Step 4: Get examples from multiple sources in parallel
            examples_tasks = []

            if config.include_schema_examples:
                schema_examples_task = self._get_schema_examples_from_tables(
                    relevant_tables, config
                )
                examples_tasks.append(("schema", schema_examples_task))

            if config.include_user_examples and user_id:
                verified_examples_task = self._get_verified_user_examples_optimized(
                    query_embedding, query_text, user_id, config
                )
                examples_tasks.append(("verified", verified_examples_task))

            # Execute example retrieval in parallel
            all_examples = []
            for source_type, task in examples_tasks:
                try:
                    examples = await task
                    all_examples.extend(examples)
                except Exception as e:
                    logger.warning(f"Failed to get {source_type} examples: {e}")

            # Step 5: Rank examples using embedding similarity (reusing query embedding)
            ranked_examples = await self._rank_examples_with_embeddings(
                all_examples, query_embedding, query_text, config
            )

            # Step 6: Build final result
            end_time = time.time()

            result = SchemaRetrievalResult(
                schema_structure=schema_structure,
                examples=ranked_examples,
                retrieval_metadata={
                    "tables_found": len(relevant_tables),
                    "total_examples_found": len(all_examples),
                    "examples_returned": len(ranked_examples),
                    "vector_threshold_used": config.vector_similarity_threshold,
                    "query_processed": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                    "embedding_reused": True,
                    "cache_hit": False
                },
                performance_metrics={
                    "total_time_ms": round((end_time - start_time) * 1000, 2),
                    "tables_retrieved": len(relevant_tables),
                    "examples_processed": len(all_examples),
                    "embedding_generation_time": 0
                }
            )

            # Cache the result
            self._cache_result(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error in enhanced schema retrieval: {str(e)}", exc_info=True)
            return self._create_error_result(str(e), start_time)

    async def retrieve_schema_for_description(
            self,
            schema_targets: Dict[str, Any],
            user_id: Optional[str] = None,
            config: SchemaDescriptionConfig = None
    ) -> SchemaDescriptionResult:
        """
        Retrieve schema information specifically for description generation.

        This is different from query generation because:
        1. We know exactly what tables/columns user wants
        2. We don't need examples (usually)
        3. We need more comprehensive data (higher limits)
        4. We use direct table/schema lookups instead of vector search
        """
        start_time = time.time()
        config = config or SchemaDescriptionConfig()

        try:
            # Extract targets
            target_tables = schema_targets.get("tables", [])
            target_columns = schema_targets.get("columns", [])
            detail_level = schema_targets.get("detail_level", "standard")

            # Override config detail level if specified in targets
            if detail_level != "standard":
                config.detail_level = detail_level

            # Check cache first
            cache_key = self._build_description_cache_key(schema_targets, config)
            cached_result = self._get_cached_description_result(cache_key)
            if cached_result:
                logger.info("📦 Returning cached schema description result")
                return cached_result

            # Determine retrieval strategy
            if "*ALL*" in target_tables or not target_tables:
                # Get all available schemas and tables
                schema_structure = await self._retrieve_all_schemas_for_description(config)
            else:
                # Get specific tables/schemas
                schema_structure = await self._retrieve_specific_schemas_for_description(
                    target_tables, target_columns, config
                )

            # Build result
            end_time = time.time()
            result = SchemaDescriptionResult(
                schema_structure=schema_structure,
                metadata={
                    "targets": schema_targets,
                    "detail_level": config.detail_level,
                    "retrieval_method": "direct_lookup",
                    "tables_found": len(schema_structure.get("schemas", {})),
                    "cache_hit": False
                },
                performance_metrics={
                    "total_time_ms": round((end_time - start_time) * 1000, 2),
                    "retrieval_method": "schema_description_optimized"
                }
            )

            # Cache the result
            self._cache_description_result(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error in schema description retrieval: {str(e)}")
            return self._create_empty_description_result(str(e), start_time)

    # ========== PERFORMANCE OPTIMIZED METHODS ==========

    async def _get_or_generate_query_embedding(
            self,
            query_text: str,
            directives: List[str] = None,
            entities: List[str] = None
    ) -> Optional[List[float]]:
        """
        Get or generate query embedding with smart caching.
        Reuses cached embeddings for identical queries.
        """
        # Build search text
        search_parts = [query_text]
        if directives:
            search_parts.extend(directives)
        if entities:
            search_parts.extend(entities)
        search_text = " ".join(search_parts)

        # Check embedding cache
        embedding_key = self._get_embedding_cache_key(search_text)

        # Cleanup old embeddings periodically
        self._cleanup_embedding_cache()

        if embedding_key in self._embedding_cache:
            logger.debug("🔄 Reusing cached query embedding")
            return self._embedding_cache[embedding_key]["embedding"]

        # Generate new embedding
        start_time = time.time()
        embedding = await self.embedding_provider.get_embedding(search_text)

        if embedding:
            # Cache the embedding
            self._embedding_cache[embedding_key] = {
                "embedding": embedding,
                "timestamp": time.time(),
                "search_text": search_text[:100] + "..." if len(search_text) > 100 else search_text
            }

            generation_time = time.time() - start_time
            logger.info(f"⚡ Generated new embedding in {generation_time:.3f}s")

        return embedding

    async def _find_relevant_tables_optimized(
            self,
            query_embedding: List[float],
            query_text: str,
            directives: List[str],
            config: SchemaRetrievalConfig
    ) -> List[Dict[str, Any]]:
        """
        Optimized table search using pre-computed query embedding.
        Directly uses cosine similarity with existing table embeddings.
        """

        # Format embedding for PostgreSQL vector operations
        embedding_str = '[' + ','.join(str(float(x)) for x in query_embedding) + ']'

        # Multi-threshold search for better recall
        thresholds = [config.vector_similarity_threshold, config.vector_similarity_threshold - 0.1, 0.3]

        conn = await db_pool.get_connection()
        try:
            for threshold in thresholds:
                try:
                    # Optimized query using existing table embeddings
                    query = """
                            SELECT
                                td.id,
                                td.name AS table_name,
                                td.description,
                                td.content,
                                sd.id AS schema_id,
                                sd.name AS schema_name,
                                sg.id AS group_id,
                                sg.name AS group_name,
                                sv.id AS schema_version_id,
                                sv.version AS schema_version,
                                1 - (td.embedding <=> $1::vector) AS similarity
                            FROM
                                table_definitions td
                                    JOIN
                                schema_versions sv ON td.schema_version_id = sv.id
                                    JOIN
                                schema_definitions sd ON sv.schema_id = sd.id
                                    JOIN
                                schema_groups sg ON sd.group_id = sg.id
                                    JOIN
                                active_schemas a ON sv.id = a.current_version_id
                            WHERE
                                1 - (td.embedding <=> $1::vector) > $2
                            ORDER BY similarity DESC
                                LIMIT $3 \
                            """

                    results = await conn.fetch(query, embedding_str, threshold, config.max_tables)

                    if results:
                        logger.info(f"🎯 Found {len(results)} tables at threshold {threshold}")

                        # Convert to expected format
                        tables = []
                        for row in results:
                            content = row["content"]
                            if isinstance(content, str):
                                import json
                                content = json.loads(content)

                            tables.append({
                                "id": row["id"],
                                "table_name": row["table_name"],
                                "schema_id": row["schema_id"],
                                "schema_name": row["schema_name"],
                                "group_id": row["group_id"],
                                "group_name": row["group_name"],
                                "schema_version_id": row["schema_version_id"],
                                "schema_version": row["schema_version"],
                                "description": row["description"],
                                "content": content,
                                "similarity": row["similarity"]
                            })

                        return tables

                except Exception as e:
                    logger.warning(f"Vector search failed at threshold {threshold}: {e}")
                    continue

            return []

        finally:
            await db_pool.release_connection(conn)

    async def _get_verified_user_examples_optimized(
            self,
            query_embedding: List[float],
            query_text: str,
            user_id: str,
            config: SchemaRetrievalConfig
    ) -> List[SchemaExample]:
        """
        Optimized verified examples retrieval using pre-computed embeddings.
        Reuses both query embedding and stored verified query embeddings.
        """

        embedding_str = '[' + ','.join(str(float(x)) for x in query_embedding) + ']'

        try:
            conn = await db_pool.get_connection()
            try:
                # Direct similarity calculation using existing embeddings
                query = """
                        SELECT
                            id,
                            original_query,
                            generated_query,
                            1 - (embedding <=> $1::vector) AS similarity,
                            CASE WHEN user_id = $2 THEN true ELSE false END AS is_user_specific
                        FROM verified_queries
                        WHERE
                            (user_id = $2 OR is_public = true) AND
                            embedding IS NOT NULL AND
                            1 - (embedding <=> $1::vector) > 0.4
                        ORDER BY
                            CASE WHEN user_id = $2 THEN 1 ELSE 2 END,  -- Prioritize user's own queries
                            similarity DESC
                            LIMIT $3 \
                        """

                results = await conn.fetch(query, embedding_str, user_id, config.max_verified_examples)

                # Convert to unified format
                verified_examples = []
                for row in results:
                    verified_examples.append(SchemaExample(
                        id=f"verified_{row['id']}",
                        natural_language=row["original_query"],
                        generated_query=row["generated_query"],
                        description="User verified query",
                        source_type="verified_query",
                        relevance_score=row["similarity"],
                        table_names=self._extract_table_names_from_query(row["generated_query"]),
                        is_user_specific=row["is_user_specific"],
                        metadata={"similarity": row["similarity"], "embedding_reused": True}
                    ))

                return verified_examples

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error getting optimized verified examples: {str(e)}")
            return []

    async def _rank_examples_with_embeddings(
            self,
            all_examples: List[SchemaExample],
            query_embedding: List[float],
            query_text: str,
            config: SchemaRetrievalConfig
    ) -> List[SchemaExample]:
        """
        Rank examples using embedding similarity instead of word overlap.
        Reuses the query embedding for consistent similarity calculation.
        """

        if not all_examples:
            return []

        # For schema examples that don't have similarity scores yet, calculate them
        for example in all_examples:
            if example.source_type == "schema_example" and example.relevance_score == 0.0:
                # Generate embedding for schema example if needed
                try:
                    example_embedding = await self.embedding_provider.get_embedding(example.natural_language)
                    if example_embedding:
                        # Calculate cosine similarity
                        example.relevance_score = self._calculate_cosine_similarity(
                            query_embedding, example_embedding
                        )
                    else:
                        # Fallback to text similarity
                        example.relevance_score = self._calculate_text_similarity(
                            query_text, example.natural_language
                        )
                except Exception as e:
                    logger.warning(f"Failed to calculate embedding similarity for example: {e}")
                    example.relevance_score = self._calculate_text_similarity(
                        query_text, example.natural_language
                    )

            # Boost user-specific examples
            if example.is_user_specific:
                example.relevance_score += config.similarity_boost_user_examples

        # Sort by relevance score (descending)
        ranked_examples = sorted(
            all_examples,
            key=lambda x: x.relevance_score,
            reverse=True
        )

        # Filter to max total examples
        return ranked_examples[:config.max_total_examples]

    def _calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        Fast implementation for ranking.
        """
        try:
            import numpy as np

            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {e}")
            return 0.0

    # ========== SCHEMA DESCRIPTION IMPLEMENTATION ==========

    async def _retrieve_all_schemas_for_description(
            self,
            config: SchemaDescriptionConfig
    ) -> Dict[str, Any]:
        """Retrieve all available schemas for description."""
        try:
            conn = await db_pool.get_connection()
            try:
                # Get all active schemas with their tables
                query = """
                        SELECT
                            sg.name as group_name,
                            sg.description as group_description,
                            sd.name as schema_name,
                            sd.description as schema_description,
                            td.name as table_name,
                            td.description as table_description,
                            td.content as table_content
                        FROM
                            schema_groups sg
                                JOIN
                            schema_definitions sd ON sg.id = sd.group_id
                                JOIN
                            active_schemas a ON sd.id = a.schema_id
                                JOIN
                            schema_versions sv ON a.current_version_id = sv.id
                                JOIN
                            table_definitions td ON sv.id = td.schema_version_id
                        ORDER BY
                            sg.name, sd.name, td.name \
                        """

                results = await conn.fetch(query)

                # Organize results hierarchically
                schema_structure = {"schemas": {}}

                for row in results:
                    group_name = row["group_name"]
                    schema_name = row["schema_name"]
                    table_name = row["table_name"]

                    # Initialize group if not exists
                    if group_name not in schema_structure["schemas"]:
                        schema_structure["schemas"][group_name] = {
                            "description": row["group_description"],
                            "schemas": {}
                        }

                    # Initialize schema if not exists
                    if schema_name not in schema_structure["schemas"][group_name]["schemas"]:
                        schema_structure["schemas"][group_name]["schemas"][schema_name] = {
                            "description": row["schema_description"],
                            "tables": {}
                        }

                    # Add table content
                    table_content = row["table_content"]
                    if isinstance(table_content, str):
                        import json
                        table_content = json.loads(table_content)

                    # Apply detail level filtering
                    filtered_content = self._apply_detail_level_filtering(
                        table_content, config.detail_level
                    )

                    schema_structure["schemas"][group_name]["schemas"][schema_name]["tables"][table_name] = {
                        "description": row["table_description"],
                        **filtered_content
                    }

                return schema_structure

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error retrieving all schemas: {str(e)}")
            return {"schemas": {}}

    async def _retrieve_specific_schemas_for_description(
            self,
            target_tables: List[str],
            target_columns: List[str],
            config: SchemaDescriptionConfig
    ) -> Dict[str, Any]:
        """Retrieve specific schemas/tables for description."""
        try:
            conn = await db_pool.get_connection()
            try:
                schema_structure = {"schemas": {}}

                for target in target_tables:
                    # Try as schema name first, then as table name
                    tables_found = await self._find_tables_for_target(conn, target, target_columns, config)

                    # Merge results into schema structure
                    for group_name, group_data in tables_found.get("schemas", {}).items():
                        if group_name not in schema_structure["schemas"]:
                            schema_structure["schemas"][group_name] = group_data
                        else:
                            # Merge schemas within group
                            for schema_name, schema_data in group_data.get("schemas", {}).items():
                                if schema_name not in schema_structure["schemas"][group_name]["schemas"]:
                                    schema_structure["schemas"][group_name]["schemas"][schema_name] = schema_data
                                else:
                                    # Merge tables within schema
                                    schema_structure["schemas"][group_name]["schemas"][schema_name]["tables"].update(
                                        schema_data.get("tables", {})
                                    )

                return schema_structure

            finally:
                await db_pool.release_connection(conn)

        except Exception as e:
            logger.error(f"Error retrieving specific schemas: {str(e)}")
            return {"schemas": {}}

    async def _find_tables_for_target(
            self,
            conn,
            target: str,
            target_columns: List[str],
            config: SchemaDescriptionConfig
    ) -> Dict[str, Any]:
        """Find tables for a specific target (schema or table name)."""
        # Try as schema name first
        schema_query = """
                       SELECT
                           sg.name as group_name,
                           sg.description as group_description,
                           sd.name as schema_name,
                           sd.description as schema_description,
                           td.name as table_name,
                           td.description as table_description,
                           td.content as table_content
                       FROM
                           schema_groups sg
                               JOIN
                           schema_definitions sd ON sg.id = sd.group_id
                               JOIN
                           active_schemas a ON sd.id = a.schema_id
                               JOIN
                           schema_versions sv ON a.current_version_id = sv.id
                               JOIN
                           table_definitions td ON sv.id = td.schema_version_id
                       WHERE
                           LOWER(sd.name) = LOWER($1)
                       ORDER BY td.name \
                       """

        results = await conn.fetch(schema_query, target)

        # If no results as schema, try as table name
        if not results:
            table_query = """
                          SELECT
                              sg.name as group_name,
                              sg.description as group_description,
                              sd.name as schema_name,
                              sd.description as schema_description,
                              td.name as table_name,
                              td.description as table_description,
                              td.content as table_content
                          FROM
                              table_definitions td
                                  JOIN
                              schema_versions sv ON td.schema_version_id = sv.id
                                  JOIN
                              schema_definitions sd ON sv.schema_id = sd.id
                                  JOIN
                              schema_groups sg ON sd.group_id = sg.id
                                  JOIN
                              active_schemas a ON sd.id = a.schema_id
                          WHERE
                              LOWER(td.name) = LOWER($1) \
                          """

            results = await conn.fetch(table_query, target)

        # Process results into schema structure
        schema_structure = {"schemas": {}}

        for row in results:
            group_name = row["group_name"]
            schema_name = row["schema_name"]
            table_name = row["table_name"]

            # Initialize structure
            if group_name not in schema_structure["schemas"]:
                schema_structure["schemas"][group_name] = {
                    "description": row["group_description"],
                    "schemas": {}
                }

            if schema_name not in schema_structure["schemas"][group_name]["schemas"]:
                schema_structure["schemas"][group_name]["schemas"][schema_name] = {
                    "description": row["schema_description"],
                    "tables": {}
                }

            # Process table content
            table_content = row["table_content"]
            if isinstance(table_content, str):
                import json
                table_content = json.loads(table_content)

            # Filter by target columns if specified
            if target_columns:
                table_content = self._filter_by_target_columns(table_content, target_columns)

            # Apply detail level filtering
            filtered_content = self._apply_detail_level_filtering(table_content, config.detail_level)

            schema_structure["schemas"][group_name]["schemas"][schema_name]["tables"][table_name] = {
                "description": row["table_description"],
                **filtered_content
            }

        return schema_structure

    def _apply_detail_level_filtering(self, table_content: Dict[str, Any], detail_level: str) -> Dict[str, Any]:
        """Apply detail level filtering to table content."""
        if detail_level == "summary":
            # For summary, only include basic info
            return {
                "columns": [
                    {"name": col.get("name", ""), "type": col.get("type", col.get("kdb_type", ""))}
                    for col in table_content.get("columns", [])[:10]  # Limit to 10 columns
                ]
            }
        elif detail_level == "detailed":
            # For detailed, include everything
            return table_content
        else:  # standard
            # For standard, include most info but limit examples
            result = table_content.copy()
            if "examples" in result:
                result["examples"] = result["examples"][:3]  # Limit examples
            return result

    def _filter_by_target_columns(self, table_content: Dict[str, Any], target_columns: List[str]) -> Dict[str, Any]:
        """Filter table content to only include target columns."""
        if not target_columns or "columns" not in table_content:
            return table_content

        filtered_content = table_content.copy()
        filtered_columns = []

        for col in table_content["columns"]:
            col_name = col.get("name", "")
            if col_name.lower() in [tc.lower() for tc in target_columns]:
                filtered_columns.append(col)

        filtered_content["columns"] = filtered_columns
        return filtered_content

    # ========== HELPER METHODS ==========

    async def _build_schema_structure(self, relevant_tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build schema structure from relevant tables."""

        schema_structure = {
            "description": "Enhanced schema with relevant tables",
            "tables": {},
            "metadata": {
                "retrieval_method": "vector_search",
                "table_count": len(relevant_tables)
            }
        }

        for table in relevant_tables:
            table_name = table["table_name"]
            schema_structure["tables"][table_name] = table["content"]

            # Add retrieval metadata to each table
            if "metadata" not in schema_structure["tables"][table_name]:
                schema_structure["tables"][table_name]["metadata"] = {}

            schema_structure["tables"][table_name]["metadata"].update({
                "similarity_score": table.get("similarity", 0.0),
                "schema_name": table.get("schema_name", "unknown"),
                "group_name": table.get("group_name", "unknown")
            })

        return schema_structure

    async def _get_schema_examples_from_tables(
            self,
            relevant_tables: List[Dict[str, Any]],
            config: SchemaRetrievalConfig
    ) -> List[SchemaExample]:
        """Get schema examples from the relevant tables."""

        table_ids = [table["id"] for table in relevant_tables]

        if not table_ids:
            return []

        try:
            from app.services.schema_management import SchemaManager
            schema_manager = SchemaManager()

            raw_examples = await schema_manager.get_examples_for_tables(
                table_ids=table_ids,
                limit=config.max_schema_examples
            )

            # Convert to unified format
            schema_examples = []
            for example in raw_examples:
                schema_examples.append(SchemaExample(
                    id=f"schema_{example['id']}",
                    natural_language=example["natural_language"],
                    generated_query=example["query"],
                    description=example.get("description", ""),
                    source_type="schema_example",
                    relevance_score=0.0,  # Will be calculated later
                    table_names=self._extract_table_names_from_query(example["query"]),
                    is_user_specific=False,
                    metadata={"table_ids": table_ids}
                ))

            return schema_examples

        except Exception as e:
            logger.error(f"Error getting schema examples: {str(e)}")
            return []

    def _extract_table_names_from_query(self, query: str) -> List[str]:
        """Extract table names from generated query."""
        import re

        # Simple regex to find table names after 'from' keyword
        matches = re.findall(r'from\s+(\w+)', query, re.IGNORECASE)
        return list(set(matches))  # Remove duplicates

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        # Simple word overlap similarity (can be enhanced with embeddings)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    # ========== CACHING METHODS ==========

    def _build_cache_key(self, query_text: str, directives: List[str], entities: List[str],
                         user_id: Optional[str], config: SchemaRetrievalConfig) -> str:
        """Build cache key for full results."""
        key_parts = [
            query_text,
            ",".join(sorted(directives or [])),
            ",".join(sorted(entities or [])),
            user_id or "anonymous",
            str(config.max_tables),
            str(config.vector_similarity_threshold),
            str(config.include_user_examples),
            str(config.include_schema_examples)
        ]

        # Create hash for compact key
        import hashlib
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_embedding_cache_key(self, search_text: str) -> str:
        """Get cache key for embeddings."""
        import hashlib
        return hashlib.md5(search_text.encode()).hexdigest()

    def _build_description_cache_key(self, schema_targets: Dict[str, Any], config: SchemaDescriptionConfig) -> str:
        """Build cache key for description results."""
        import hashlib
        key_parts = [
            str(schema_targets.get("tables", [])),
            str(schema_targets.get("columns", [])),
            str(schema_targets.get("detail_level", "standard")),
            config.detail_level,
            str(config.include_examples),
            str(config.include_relationships)
        ]
        key_str = "|".join(key_parts)
        return f"desc_{hashlib.md5(key_str.encode()).hexdigest()}"

    def _get_cached_result(self, cache_key: str) -> Optional[SchemaRetrievalResult]:
        """Get cached result if valid."""
        if cache_key in self._result_cache:
            cached_item = self._result_cache[cache_key]
            if time.time() - cached_item["timestamp"] <= self._cache_ttl:
                cached_item["result"].retrieval_metadata["cache_hit"] = True
                return cached_item["result"]
            else:
                # Remove expired entry
                del self._result_cache[cache_key]
        return None

    def _get_cached_description_result(self, cache_key: str) -> Optional[SchemaDescriptionResult]:
        """Get cached description result if valid."""
        if cache_key in self._result_cache:
            cached_item = self._result_cache[cache_key]
            if time.time() - cached_item["timestamp"] <= self._cache_ttl:
                cached_item["result"].metadata["cache_hit"] = True
                return cached_item["result"]
            else:
                # Remove expired entry
                del self._result_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: SchemaRetrievalResult):
        """Cache a result."""
        self._result_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        # Cleanup old entries if cache is getting large
        if len(self._result_cache) > 100:
            self._cleanup_result_cache()

    def _cache_description_result(self, cache_key: str, result: SchemaDescriptionResult):
        """Cache description result."""
        self._result_cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        # Cleanup old entries if cache is getting large
        if len(self._result_cache) > 100:
            self._cleanup_result_cache()

    def _cleanup_embedding_cache(self):
        """Clean up old embedding cache entries."""
        current_time = time.time()
        if current_time - self._last_cleanup < 60:  # Cleanup every minute
            return

        expired_keys = []
        for key, cached_item in self._embedding_cache.items():
            if current_time - cached_item["timestamp"] > self._cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._embedding_cache[key]

        self._last_cleanup = current_time

        if expired_keys:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired embedding cache entries")

    def _cleanup_result_cache(self):
        """Clean up old result cache entries."""
        current_time = time.time()
        expired_keys = []

        for key, cached_item in self._result_cache.items():
            if current_time - cached_item["timestamp"] > self._cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._result_cache[key]

        if expired_keys:
            logger.info(f"🧹 Cleaned up {len(expired_keys)} expired result cache entries")

    # ========== UTILITY METHODS ==========

    def _create_empty_result(self, reason: str, start_time: float) -> SchemaRetrievalResult:
        """Create empty result for no data found scenarios."""
        return SchemaRetrievalResult(
            schema_structure={"tables": {}, "metadata": {"reason": reason}},
            examples=[],
            retrieval_metadata={"reason": reason},
            performance_metrics={
                "total_time_ms": round((time.time() - start_time) * 1000, 2),
                "tables_retrieved": 0,
                "examples_processed": 0
            }
        )

    def _create_error_result(self, error_msg: str, start_time: float) -> SchemaRetrievalResult:
        """Create error result."""
        return SchemaRetrievalResult(
            schema_structure={"tables": {}, "metadata": {"error": error_msg}},
            examples=[],
            retrieval_metadata={"error": error_msg},
            performance_metrics={
                "total_time_ms": round((time.time() - start_time) * 1000, 2),
                "tables_retrieved": 0,
                "examples_processed": 0,
                "error": error_msg
            }
        )

    def _create_empty_description_result(self, reason: str, start_time: float) -> SchemaDescriptionResult:
        """Create empty description result for error cases."""
        return SchemaDescriptionResult(
            schema_structure={"schemas": {}},
            metadata={"error": reason},
            performance_metrics={
                "total_time_ms": round((time.time() - start_time) * 1000, 2),
                "error": reason
            }
        )

    # ========== PERFORMANCE MONITORING ==========

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        current_time = time.time()

        # Result cache stats
        result_cache_size = len(self._result_cache)
        result_cache_expired = sum(
            1 for item in self._result_cache.values()
            if current_time - item["timestamp"] > self._cache_ttl
        )

        # Embedding cache stats
        embedding_cache_size = len(self._embedding_cache)
        embedding_cache_expired = sum(
            1 for item in self._embedding_cache.values()
            if current_time - item["timestamp"] > self._cache_ttl
        )

        return {
            "result_cache": {
                "size": result_cache_size,
                "expired": result_cache_expired,
                "hit_ratio": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
            },
            "embedding_cache": {
                "size": embedding_cache_size,
                "expired": embedding_cache_expired
            },
            "last_cleanup": self._last_cleanup,
            "cache_ttl": self._cache_ttl
        }

    def clear_cache(self):
        """Clear all caches - useful for testing or memory management."""
        self._result_cache.clear()
        self._embedding_cache.clear()
        logger.info("🧹 Cleared all caches")

    # ========== API HELPER METHODS ==========

    async def get_schema_for_api(
            self,
            query: str,
            directives: List[str] = None,
            user_id: str = None,
            include_examples: bool = True,
            max_tables: int = 5
    ) -> Dict[str, Any]:
        """
        API-friendly version that returns JSON-serializable data.
        Perfect for REST endpoints and FastMCP integration.
        """
        config = SchemaRetrievalConfig(
            max_tables=max_tables,
            include_schema_examples=include_examples,
            include_user_examples=include_examples and bool(user_id)
        )

        result = await self.retrieve_schema_with_examples(
            query_text=query,
            directives=directives or [],
            user_id=user_id,
            config=config
        )

        return {
            "schema": result.schema_structure,
            "examples": [
                {
                    "id": ex.id,
                    "natural_language": ex.natural_language,
                    "generated_query": ex.generated_query,
                    "description": ex.description,
                    "source_type": ex.source_type,
                    "relevance_score": round(ex.relevance_score, 4),
                    "table_names": ex.table_names,
                    "is_user_specific": ex.is_user_specific,
                    "metadata": ex.metadata or {}
                }
                for ex in result.examples
            ],
            "metadata": result.retrieval_metadata,
            "performance": result.performance_metrics,
            "cache_stats": self.get_cache_stats()
        }

    async def get_schema_description_data(
            self,
            schema_targets: Dict[str, Any],
            user_id: str = None,
            detail_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        API-friendly version for schema descriptions.
        Perfect for REST endpoints and schema description node.
        """
        config = SchemaDescriptionConfig(
            detail_level=detail_level,
            include_examples=detail_level == "detailed",
            include_relationships=True
        )

        result = await self.retrieve_schema_for_description(
            schema_targets=schema_targets,
            user_id=user_id,
            config=config
        )

        return {
            "schema_data": result.schema_structure,
            "metadata": result.metadata,
            "performance": result.performance_metrics,
            "cache_stats": self.get_cache_stats()
        }

    # ========== BENCHMARKING METHODS ==========

    async def benchmark_performance(
            self,
            test_queries: List[str],
            user_id: str = None,
            iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark the service performance with multiple queries.
        Useful for performance testing and optimization.
        """
        results = {
            "test_queries": len(test_queries),
            "iterations": iterations,
            "total_time": 0,
            "avg_time_per_query": 0,
            "cache_performance": {},
            "detailed_results": []
        }

        total_start = time.time()

        for iteration in range(iterations):
            iteration_results = []

            for i, query in enumerate(test_queries):
                start_time = time.time()

                try:
                    result = await self.retrieve_schema_with_examples(
                        query_text=query,
                        user_id=user_id
                    )

                    execution_time = time.time() - start_time

                    iteration_results.append({
                        "query_index": i,
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "execution_time_ms": round(execution_time * 1000, 2),
                        "tables_found": len(result.schema_structure.get("tables", {})),
                        "examples_found": len(result.examples),
                        "cache_hit": result.retrieval_metadata.get("cache_hit", False),
                        "embedding_reused": result.retrieval_metadata.get("embedding_reused", False)
                    })

                except Exception as e:
                    iteration_results.append({
                        "query_index": i,
                        "query": query[:50] + "..." if len(query) > 50 else query,
                        "error": str(e),
                        "execution_time_ms": round((time.time() - start_time) * 1000, 2)
                    })

            results["detailed_results"].append({
                "iteration": iteration + 1,
                "results": iteration_results
            })

        results["total_time"] = time.time() - total_start
        results["avg_time_per_query"] = results["total_time"] / (len(test_queries) * iterations)
        results["cache_performance"] = self.get_cache_stats()

        return results