# app/routes/debug_schema.py
from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional
import time
import json
import re

from app.services.schema_management import SchemaManager
from app.core.logging import logger

router = APIRouter()

# Multi-threshold search function (copied from your multi-threshold implementation)
async def multi_threshold_vector_search(schema_manager, search_text, max_results=5, thresholds=None, min_results=1):
    """
    Multi-threshold vector search that tries progressively lower thresholds
    until it finds results or exhausts all thresholds.
    """
    if thresholds is None:
        thresholds = [0.65, 0.59, 0.55, 0.50, 0.45, 0.40]

    logger.info(f"Starting multi-threshold search with thresholds: {thresholds}")

    for threshold in thresholds:
        try:
            logger.debug(f"Trying vector search with threshold {threshold}")

            results = await schema_manager.find_tables_by_vector_search(
                search_text,
                similarity_threshold=threshold,
                max_results=max_results
            )

            if results and len(results) >= min_results:
                # Found sufficient results
                logger.info(f"Multi-threshold search found {len(results)} results at threshold {threshold}")

                # Add threshold info to results
                for result in results:
                    result['search_threshold_used'] = threshold

                return results

        except Exception as e:
            logger.warning(f"Vector search failed at threshold {threshold}: {str(e)}")
            continue

    # No results found at any threshold
    logger.info("Multi-threshold vector search found no results at any threshold")
    return []

@router.post("/debug/schema-search")
async def debug_schema_search(
        data: Dict[str, Any] = Body(...)
):
    """
    Debug endpoint to test schema retrieval with detailed breakdown.

    Request body:
    {
        "query": "your natural language query",
        "methods": ["vector", "directive", "multi_threshold", "bm25"],  // optional, defaults to ["vector", "directive"]
        "vector_threshold": 0.35,  // optional, defaults to 0.65 (ignored for multi_threshold)
        "multi_threshold_config": {
            "thresholds": [0.65, 0.59, 0.55, 0.50, 0.45, 0.40],  // optional
            "min_results": 1  // optional
        },
        "max_results": 10  // optional, defaults to 5
    }
    """
    try:
        # Extract parameters
        query = data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        methods = data.get("methods", ["vector", "directive"])
        vector_threshold = data.get("vector_threshold", 0.65)
        multi_threshold_config = data.get("multi_threshold_config", {})
        max_results = data.get("max_results", 5)

        logger.info(f"Debug schema search for query: {query}")

        # Initialize components
        schema_manager = SchemaManager()
        debug_results = {
            "query": query,
            "timestamp": time.time(),
            "methods_tested": methods,
            "results": {}
        }

        # Check if schemas exist
        schemas_exist = await schema_manager.check_schemas_available()
        debug_results["schemas_available"] = schemas_exist

        if not schemas_exist:
            debug_results["error"] = "No schemas found in database"
            return debug_results

        # 1. Test Vector Search
        if "vector" in methods:
            vector_start = time.time()
            try:
                vector_results = await schema_manager.find_tables_by_vector_search(
                    query,
                    similarity_threshold=vector_threshold,
                    max_results=max_results
                )

                vector_time = time.time() - vector_start

                # Format vector results
                formatted_vector = []
                for result in vector_results:
                    formatted_vector.append({
                        "table_id": result["id"],
                        "table_name": result["table_name"],
                        "schema_name": result["schema_name"],
                        "group_name": result["group_name"],
                        "similarity_score": result["similarity"],
                        "description": result["description"],
                        "content_preview": _get_content_preview(result["content"])
                    })

                debug_results["results"]["vector"] = {
                    "success": True,
                    "execution_time": round(vector_time, 4),
                    "threshold_used": vector_threshold,
                    "results_count": len(vector_results),
                    "results": formatted_vector
                }

            except Exception as e:
                debug_results["results"]["vector"] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - vector_start
                }

        # 2. Test Directive Search
        if "directive" in methods:
            directive_start = time.time()
            try:
                # Extract directives from query
                directives = _extract_directives(query)

                directive_results = []
                if directives:
                    directive_results = await _directive_search(directives, schema_manager)

                directive_time = time.time() - directive_start

                debug_results["results"]["directive"] = {
                    "success": True,
                    "execution_time": round(directive_time, 4),
                    "directives_found": directives,
                    "results_count": len(directive_results),
                    "results": directive_results
                }

            except Exception as e:
                debug_results["results"]["directive"] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - directive_start
                }

        # 3. Test Multi-Threshold Search
        if "multi_threshold" in methods:
            multi_threshold_start = time.time()
            try:
                # Use multi-threshold search
                thresholds = multi_threshold_config.get("thresholds", [0.65, 0.59, 0.55, 0.50, 0.45, 0.40])
                min_results = multi_threshold_config.get("min_results", 1)

                multi_threshold_results = await multi_threshold_vector_search(
                    schema_manager,
                    query,
                    max_results,
                    thresholds,
                    min_results
                )

                multi_threshold_time = time.time() - multi_threshold_start

                # Format results
                formatted_multi_threshold = []
                threshold_used = None

                for result in multi_threshold_results:
                    threshold_used = result.get("search_threshold_used", "unknown")
                    formatted_multi_threshold.append({
                        "table_id": result["id"],
                        "table_name": result["table_name"],
                        "schema_name": result["schema_name"],
                        "group_name": result["group_name"],
                        "similarity_score": result["similarity"],
                        "threshold_used": threshold_used,
                        "description": result["description"],
                        "content_preview": _get_content_preview(result["content"])
                    })

                debug_results["results"]["multi_threshold"] = {
                    "success": True,
                    "execution_time": round(multi_threshold_time, 4),
                    "final_threshold_used": threshold_used,
                    "thresholds_tried": thresholds,
                    "results_count": len(multi_threshold_results),
                    "results": formatted_multi_threshold
                }

            except Exception as e:
                debug_results["results"]["multi_threshold"] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - multi_threshold_start
                }

        # 4. Test BM25 Search (if available)
        # 4. Test BM25 Search (if available)
        if "bm25" in methods:
            bm25_start = time.time()
            try:
                # Try to use BM25 if implemented
                bm25_results = await _test_bm25_search(query, schema_manager)
                bm25_time = time.time() - bm25_start

                debug_results["results"]["bm25"] = {
                    "success": True,
                    "execution_time": round(bm25_time, 4),
                    "results_count": len(bm25_results),
                    "results": bm25_results
                }

            except Exception as e:
                debug_results["results"]["bm25"] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - bm25_start,
                    "note": "BM25 search not yet implemented"
                }

        # 5. Analysis and Recommendations
        debug_results["analysis"] = _analyze_results(debug_results, query)

        return debug_results

    except Exception as e:
        logger.error(f"Error in debug schema search: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Debug search failed: {str(e)}")

@router.post("/debug/threshold-comparison")
async def debug_threshold_comparison(
        data: Dict[str, Any] = Body(...)
):
    """
    Compare results across different similarity thresholds.

    Request body:
    {
        "query": "your query",
        "thresholds": [0.65, 0.60, 0.55, 0.50, 0.45, 0.40],  // optional
        "max_results": 10  // optional
    }
    """
    try:
        query = data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        thresholds = data.get("thresholds", [0.65, 0.60, 0.55, 0.50, 0.45, 0.40])
        max_results = data.get("max_results", 10)

        schema_manager = SchemaManager()

        comparison_results = {
            "query": query,
            "thresholds_tested": thresholds,
            "max_results": max_results,
            "results_by_threshold": {},
            "summary": {
                "first_success_threshold": None,
                "total_unique_tables": 0,
                "threshold_performance": []
            }
        }

        all_found_tables = set()
        first_success = None

        for threshold in thresholds:
            threshold_start = time.time()
            try:
                results = await schema_manager.find_tables_by_vector_search(
                    query,
                    similarity_threshold=threshold,
                    max_results=max_results
                )

                execution_time = time.time() - threshold_start

                # Track unique tables found
                table_names = []
                for result in results:
                    table_name = f"{result['schema_name']}.{result['table_name']}"
                    table_names.append(table_name)
                    all_found_tables.add(table_name)

                # Record first successful threshold
                if results and first_success is None:
                    first_success = threshold

                comparison_results["results_by_threshold"][str(threshold)] = {
                    "success": True,
                    "execution_time": round(execution_time, 4),
                    "results_count": len(results),
                    "table_names": table_names,
                    "results": [
                        {
                            "table_name": r["table_name"],
                            "schema_name": r["schema_name"],
                            "similarity_score": r["similarity"],
                            "description": r["description"][:100] + "..." if len(r["description"]) > 100 else r["description"]
                        }
                        for r in results
                    ]
                }

                # Track performance
                comparison_results["summary"]["threshold_performance"].append({
                    "threshold": threshold,
                    "results_count": len(results),
                    "execution_time": round(execution_time, 4),
                    "unique_tables": len(set(table_names))
                })

            except Exception as e:
                comparison_results["results_by_threshold"][str(threshold)] = {
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - threshold_start
                }

                comparison_results["summary"]["threshold_performance"].append({
                    "threshold": threshold,
                    "results_count": 0,
                    "execution_time": time.time() - threshold_start,
                    "error": str(e)
                })

        # Complete summary
        comparison_results["summary"]["first_success_threshold"] = first_success
        comparison_results["summary"]["total_unique_tables"] = len(all_found_tables)
        comparison_results["summary"]["unique_table_names"] = list(all_found_tables)

        # Recommendations
        recommendations = []
        if first_success:
            recommendations.append(f"First successful threshold: {first_success}")
            if first_success < 0.65:
                recommendations.append(f"Consider lowering default threshold from 0.65 to {first_success}")
        else:
            recommendations.append("No results found at any threshold - check query/schema mismatch")

        comparison_results["recommendations"] = recommendations

        return comparison_results

    except Exception as e:
        logger.error(f"Error in threshold comparison: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Threshold comparison failed: {str(e)}")

@router.post("/debug/embedding-comparison")
async def debug_embedding_comparison(
        data: Dict[str, Any] = Body(...)
):
    """
    Compare embedding similarity between query and specific tables.

    Request body:
    {
        "query": "your query",
        "table_names": ["market_price", "trades"],  // optional, compares with all if not specified
        "show_embedding_text": true  // optional, shows what text is actually embedded
    }
    """
    try:
        query = data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        table_names = data.get("table_names", [])
        show_embedding_text = data.get("show_embedding_text", False)

        schema_manager = SchemaManager()

        # Get all tables or specific tables
        conn = await schema_manager._get_db_connection()
        try:
            if table_names:
                # Get specific tables
                placeholders = ",".join([f"${i+1}" for i in range(len(table_names))])
                query_sql = f"""
                SELECT 
                    td.id, td.name AS table_name, td.description, td.content,
                    sd.name AS schema_name, sg.name AS group_name
                FROM table_definitions td
                JOIN schema_versions sv ON td.schema_version_id = sv.id
                JOIN schema_definitions sd ON sv.schema_id = sd.id
                JOIN schema_groups sg ON sd.group_id = sg.id
                JOIN active_schemas a ON sv.id = a.current_version_id
                WHERE td.name IN ({placeholders})
                """
                results = await conn.fetch(query_sql, *table_names)
            else:
                # Get all tables
                query_sql = """
                            SELECT
                                td.id, td.name AS table_name, td.description, td.content,
                                sd.name AS schema_name, sg.name AS group_name
                            FROM table_definitions td
                                     JOIN schema_versions sv ON td.schema_version_id = sv.id
                                     JOIN schema_definitions sd ON sv.schema_id = sd.id
                                     JOIN schema_groups sg ON sd.group_id = sg.id
                                     JOIN active_schemas a ON sv.id = a.current_version_id
                            ORDER BY sd.name, td.name
                                LIMIT 10 \
                            """
                results = await conn.fetch(query_sql)

            # Analyze each table
            comparison_results = []

            for row in results:
                table_info = {
                    "table_name": row["table_name"],
                    "schema_name": row["schema_name"],
                    "group_name": row["group_name"],
                    "description": row["description"]
                }

                # Parse content
                content = row["content"]
                if isinstance(content, str):
                    content = json.loads(content)

                # Create embedding text (simulate what gets embedded)
                embedding_text = _create_embedding_text_preview(
                    row["table_name"],
                    row["description"],
                    content
                )

                table_info["embedding_text"] = embedding_text if show_embedding_text else "[hidden]"
                table_info["content_summary"] = _get_content_preview(content)

                # Try to get similarity if possible
                try:
                    # This would require calling the actual embedding function
                    # For now, we'll do text-based analysis
                    similarity_analysis = _analyze_text_similarity(query, embedding_text)
                    table_info["similarity_analysis"] = similarity_analysis
                except Exception as e:
                    table_info["similarity_analysis"] = {"error": str(e)}

                comparison_results.append(table_info)

        finally:
            await conn.close()

        return {
            "query": query,
            "tables_analyzed": len(comparison_results),
            "show_embedding_text": show_embedding_text,
            "results": comparison_results,
            "query_analysis": _analyze_query_terms(query)
        }

    except Exception as e:
        logger.error(f"Error in embedding comparison: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding comparison failed: {str(e)}")

@router.get("/debug/corpus-info")
async def debug_corpus_info():
    """Get information about available tables and schemas for debugging."""
    try:
        schema_manager = SchemaManager()

        # Get schema and table counts
        conn = await schema_manager._get_db_connection()
        try:
            # Get schema groups
            groups_query = """
                           SELECT sg.name as group_name, COUNT(sd.id) as schema_count
                           FROM schema_groups sg
                                    LEFT JOIN schema_definitions sd ON sg.id = sd.group_id
                           GROUP BY sg.id, sg.name
                           ORDER BY sg.name \
                           """
            groups = await conn.fetch(groups_query)

            # Get active tables
            tables_query = """
                           SELECT
                               sg.name as group_name,
                               sd.name as schema_name,
                               td.name as table_name,
                               td.description
                           FROM table_definitions td
                                    JOIN schema_versions sv ON td.schema_version_id = sv.id
                                    JOIN schema_definitions sd ON sv.schema_id = sd.id
                                    JOIN schema_groups sg ON sd.group_id = sg.id
                                    JOIN active_schemas a ON sv.id = a.current_version_id
                           ORDER BY sg.name, sd.name, td.name \
                           """
            tables = await conn.fetch(tables_query)

            # Organize results
            corpus_info = {
                "total_groups": len(groups),
                "total_tables": len(tables),
                "groups": [dict(row) for row in groups],
                "tables_by_group": {}
            }

            # Group tables by schema group
            for table in tables:
                group_name = table["group_name"]
                if group_name not in corpus_info["tables_by_group"]:
                    corpus_info["tables_by_group"][group_name] = []

                corpus_info["tables_by_group"][group_name].append({
                    "schema": table["schema_name"],
                    "table": table["table_name"],
                    "description": table["description"]
                })

            return corpus_info

        finally:
            await conn.close()

    except Exception as e:
        logger.error(f"Error getting corpus info: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get corpus info: {str(e)}")

# Helper functions
def _extract_directives(query: str) -> List[str]:
    """Extract @DIRECTIVE patterns from query"""
    directives = re.findall(r'@([A-Z][A-Z0-9_]*)', query.upper())
    return list(set(directives))

async def _directive_search(directives: List[str], schema_manager: SchemaManager) -> List[Dict[str, Any]]:
    """Perform directive-based search"""
    results = []

    conn = await schema_manager._get_db_connection()
    try:
        for directive in directives:
            query = """
                    SELECT
                        td.id, td.name AS table_name, td.description, td.content,
                        sd.name AS schema_name, sg.name AS group_name
                    FROM table_definitions td
                             JOIN schema_versions sv ON td.schema_version_id = sv.id
                             JOIN schema_definitions sd ON sv.schema_id = sd.id
                             JOIN schema_groups sg ON sd.group_id = sg.id
                             JOIN active_schemas a ON sv.id = a.current_version_id
                    WHERE UPPER(sd.name) = UPPER($1) OR UPPER(sg.name) = UPPER($1) \
                    """

            directive_results = await conn.fetch(query, directive)

            for row in directive_results:
                results.append({
                    "directive": directive,
                    "table_name": row["table_name"],
                    "schema_name": row["schema_name"],
                    "group_name": row["group_name"],
                    "description": row["description"],
                    "content_preview": _get_content_preview(
                        json.loads(row["content"]) if isinstance(row["content"], str) else row["content"]
                    )
                })

    finally:
        await conn.close()

    return results

async def _test_bm25_search(query: str, schema_manager: SchemaManager) -> List[Dict[str, Any]]:
    """Test BM25 search if available"""
    # This would use the BM25 implementation when available
    # For now, return placeholder
    return []
    """Test BM25 search if available"""
    # This would use the BM25 implementation when available
    # For now, return placeholder
    return []

def _get_content_preview(content: Dict[str, Any]) -> Dict[str, Any]:
    """Get a preview of table content"""
    if not isinstance(content, dict):
        return {"error": "Invalid content format"}

    preview = {}

    # Column info
    columns = content.get("columns", [])
    if columns:
        preview["column_count"] = len(columns)
        preview["column_names"] = [col.get("name", "unknown") for col in columns[:10]]
        preview["sample_columns"] = [
            {
                "name": col.get("name", "unknown"),
                "type": col.get("type", col.get("kdb_type", "unknown")),
                "description": col.get("column_desc", col.get("description", ""))
            }
            for col in columns[:5]
        ]

    # Examples info
    examples = content.get("examples", [])
    if examples:
        preview["example_count"] = len(examples)
        preview["sample_examples"] = [
            {
                "natural_language": ex.get("natural_language", ""),
                "query": ex.get("query", "")[:100] + "..." if len(ex.get("query", "")) > 100 else ex.get("query", "")
            }
            for ex in examples[:3]
        ]

    return preview

def _create_embedding_text_preview(table_name: str, description: str, content: Dict[str, Any]) -> str:
    """Simulate what text would be created for embedding"""
    text_parts = []

    # Basic info
    text_parts.append(table_name or "")
    text_parts.append(description or "")

    # Column info
    if isinstance(content, dict):
        for column in content.get("columns", []):
            if isinstance(column, dict):
                text_parts.append(column.get("name", ""))
                text_parts.append(column.get("column_desc", ""))
                text_parts.append(column.get("description", ""))
                text_parts.append(column.get("type", ""))

    # Clean and join
    clean_parts = [part.strip() for part in text_parts if part and part.strip()]
    return " ".join(clean_parts)

def _analyze_text_similarity(query: str, embedding_text: str) -> Dict[str, Any]:
    """Basic text similarity analysis"""
    query_words = set(query.lower().split())
    embedding_words = set(embedding_text.lower().split())

    # Remove common stop words
    stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
    query_words = query_words - stop_words
    embedding_words = embedding_words - stop_words

    # Find matches
    common_words = query_words & embedding_words

    return {
        "query_word_count": len(query_words),
        "embedding_word_count": len(embedding_words),
        "common_words": list(common_words),
        "common_word_count": len(common_words),
        "jaccard_similarity": len(common_words) / len(query_words | embedding_words) if (query_words | embedding_words) else 0,
        "query_words": list(query_words),
        "embedding_words": list(embedding_words)[:20]  # Limit for readability
    }

def _analyze_query_terms(query: str) -> Dict[str, Any]:
    """Analyze query terms"""
    # Extract different types of terms
    directives = _extract_directives(query)

    # Find potential currency pairs
    currency_pairs = re.findall(r'\b[A-Z]{6}\b', query.upper())

    # Find potential symbols
    symbols = re.findall(r'\b[A-Z]{3,6}\b', query.upper())
    symbols = [s for s in symbols if s not in directives]

    # Basic word analysis
    words = re.findall(r'\b\w+\b', query.lower())
    stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "a", "an"}
    content_words = [w for w in words if w not in stop_words and len(w) > 2]

    return {
        "directives": directives,
        "currency_pairs": currency_pairs,
        "symbols": symbols,
        "content_words": content_words,
        "total_words": len(words),
        "content_word_count": len(content_words)
    }

def _analyze_results(debug_results: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Analyze results and provide recommendations"""
    analysis = {
        "summary": {},
        "issues": [],
        "recommendations": []
    }

    # Summarize results
    total_results = 0
    successful_methods = []

    for method, result in debug_results.get("results", {}).items():
        if result.get("success", False):
            successful_methods.append(method)
            total_results += result.get("results_count", 0)

    analysis["summary"] = {
        "total_results_found": total_results,
        "successful_methods": successful_methods,
        "failed_methods": [m for m in debug_results.get("methods_tested", []) if m not in successful_methods]
    }

    # Identify issues
    if total_results == 0:
        analysis["issues"].append("No results found by any search method")
        analysis["recommendations"].append("Try lowering the vector similarity threshold")
        analysis["recommendations"].append("Check if query terms match table content")

    # Vector-specific analysis
    vector_results = debug_results.get("results", {}).get("vector", {})
    if vector_results.get("success", False):
        if vector_results.get("results_count", 0) == 0:
            analysis["issues"].append("Vector search returned no results - similarity threshold may be too high")
            analysis["recommendations"].append(f"Try lowering vector threshold from {vector_results.get('threshold_used', 'unknown')} to 0.3-0.4")

    # Directive analysis
    directive_results = debug_results.get("results", {}).get("directive", {})
    if directive_results.get("success", False):
        directives_found = directive_results.get("directives_found", [])
        if not directives_found:
            analysis["issues"].append("No @DIRECTIVE patterns found in query")
            analysis["recommendations"].append("Add @SCHEMANAME directive to query for better results")

    return analysis