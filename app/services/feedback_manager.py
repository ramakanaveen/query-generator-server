# app/services/feedback_manager.py
import uuid
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import asyncpg

from app.core.config import settings
from app.core.logging import logger

class FeedbackManager:
    """
    Service for managing user feedback on generated queries.
    Stores feedback and manages verified/failed queries for future learning.
    """
    
    def __init__(self):
        # In-memory store for development/fallback
        self.feedback_store = {}
        self.verified_queries = {}
        self.failed_queries = {}
        self.db_url = settings.DATABASE_URL
    
    async def _get_db_connection(self):
        """Get a database connection."""
        return await asyncpg.connect(self.db_url)
    
    async def save_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save basic feedback for a query.
        
        Args:
            feedback_data: Dictionary containing feedback information
            
        Returns:
            The saved feedback entry
        """
        try:
            # Generate ID if not provided
            feedback_id = feedback_data.get("id", str(uuid.uuid4()))
            
            # Add timestamps and IDs
            feedback_entry = {
                "id": feedback_id,
                "timestamp": datetime.now().isoformat(),
                **feedback_data
            }
            
            # First try to store in database
            try:
                conn = await self._get_db_connection()
                try:
                    # Convert to JSON for storage
                    feedback_json = json.dumps(feedback_entry)
                    
                    # Store in generic feedback table (useful for analysis)
                    await conn.execute(
                        """
                        INSERT INTO query_feedback (id, feedback_type, query_id, user_id, feedback_data)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        feedback_id,
                        feedback_data.get("feedback_type", "unknown"),
                        feedback_data.get("query_id", ""),
                        feedback_data.get("user_id", "anonymous"),
                        feedback_json
                    )
                    
                    logger.info(f"Saved feedback to database with ID {feedback_id}")
                except Exception as db_error:
                    # Log the error but continue to save in memory
                    logger.error(f"Error saving feedback to database: {str(db_error)}")
                    # Fall back to in-memory storage
                    self.feedback_store[feedback_id] = feedback_entry
                finally:
                    await conn.close()
            except Exception as conn_error:
                # Log connection error and use in-memory store
                logger.error(f"Database connection error: {str(conn_error)}")
                self.feedback_store[feedback_id] = feedback_entry
            
            return feedback_entry
        
        except Exception as e:
            logger.error(f"Error in save_feedback: {str(e)}", exc_info=True)
            # Return minimal response on error
            return {"id": str(uuid.uuid4()), "error": str(e)}
    
    async def save_positive_feedback(self, 
                                    query_id: str,
                                    user_id: str,
                                    original_query: str,
                                    generated_query: str,
                                    conversation_id: str = None,
                                    is_public: bool = False,
                                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Save positive feedback (thumbs up) for a query.
        Also stores the query as a verified query for future few-shot learning.
        
        Args:
            query_id: ID of the query being rated
            user_id: User ID providing the feedback
            original_query: The original natural language query
            generated_query: The generated database query
            conversation_id: Optional conversation ID
            is_public: Whether this query can be used for other users
            metadata: Additional metadata
            
        Returns:
            The saved feedback entry
        """
        try:
            # First record the basic feedback
            feedback_entry = await self.save_feedback({
                "query_id": query_id,
                "feedback_type": "positive",
                "user_id": user_id,
                "original_query": original_query,
                "generated_query": generated_query,
                "conversation_id": conversation_id,
                "is_public": is_public,
                "metadata": metadata or {}
            })
            
            # Then store it as a verified query
            # First generate embedding
            embedding = None
            tables_used = []
            
            try:
                # Get embedding for the query
                from app.services.embedding_provider import EmbeddingProvider
                embedding_provider = EmbeddingProvider()
                embedding = await embedding_provider.get_embedding(original_query)
                
                # Parse metadata for tables if available
                if metadata and isinstance(metadata, dict) and 'tables_used' in metadata:
                    tables_used = metadata['tables_used']
            except Exception as embed_error:
                logger.error(f"Error generating embedding: {str(embed_error)}")
                # Continue without embedding
            
            # Try to store in database
            try:
                conn = await self._get_db_connection()
                try:
                    # Convert tables to JSON if needed
                    tables_json = json.dumps(tables_used) if tables_used else '{}'
                    
                    # Format embedding for database if available
                    embedding_formatted = None
                    if embedding:
                        if isinstance(embedding, list):
                            embedding_formatted = '[' + ','.join(str(float(x)) for x in embedding) + ']'
                    
                    # Store in verified queries table
                    query = """
                    INSERT INTO verified_queries 
                    (user_id, original_query, generated_query, conversation_id, 
                     embedding, tables_used, is_public, created_at, metadata)
                    VALUES ($1, $2, $3, $4, $5::vector, $6, $7, $8, $9)
                    RETURNING id
                    """
                    
                    verified_id = await conn.fetchval(
                        query,
                        user_id, 
                        original_query,
                        generated_query,
                        conversation_id,
                        embedding_formatted,
                        tables_json,
                        is_public,
                        datetime.now(),
                        json.dumps(metadata or {})
                    )
                    
                    logger.info(f"Saved verified query to database with ID {verified_id}")
                    feedback_entry["verified_id"] = verified_id
                except Exception as db_error:
                    # Log error but continue with in-memory fallback
                    logger.error(f"Error saving verified query to database: {str(db_error)}")
                    # Fallback to in-memory storage
                    verified_id = str(uuid.uuid4())
                    self.verified_queries[verified_id] = {
                        "id": verified_id,
                        "user_id": user_id,
                        "original_query": original_query,
                        "generated_query": generated_query,
                        "conversation_id": conversation_id,
                        "embedding": embedding,
                        "tables_used": tables_used,
                        "is_public": is_public,
                        "created_at": datetime.now().isoformat(),
                        "metadata": metadata or {}
                    }
                    feedback_entry["verified_id"] = verified_id
                finally:
                    await conn.close()
            except Exception as conn_error:
                # Log connection error and use in-memory store
                logger.error(f"Database connection error: {str(conn_error)}")
                verified_id = str(uuid.uuid4())
                self.verified_queries[verified_id] = {
                    "id": verified_id,
                    "user_id": user_id,
                    "original_query": original_query,
                    "generated_query": generated_query,
                    "conversation_id": conversation_id,
                    "embedding": embedding,
                    "tables_used": tables_used,
                    "is_public": is_public,
                    "created_at": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                feedback_entry["verified_id"] = verified_id
            
            return feedback_entry
        
        except Exception as e:
            logger.error(f"Error in save_positive_feedback: {str(e)}", exc_info=True)
            # Return minimal response on error
            return {"id": str(uuid.uuid4()), "error": str(e)}
    
    async def save_negative_feedback(self, 
                                   query_id: str,
                                   user_id: str,
                                   original_query: str,
                                   generated_query: str,
                                   feedback_text: str,
                                   conversation_id: str = None,
                                   metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Save negative feedback (thumbs down) for a query.
        Stores the feedback text and query for analysis.
        
        Args:
            query_id: ID of the query being rated
            user_id: User ID providing the feedback
            original_query: The original natural language query
            generated_query: The generated database query
            feedback_text: User explanation of what's wrong
            conversation_id: Optional conversation ID
            metadata: Additional metadata
            
        Returns:
            The saved feedback entry
        """
        try:
            # First record the basic feedback
            feedback_entry = await self.save_feedback({
                "query_id": query_id,
                "feedback_type": "negative",
                "user_id": user_id,
                "original_query": original_query,
                "generated_query": generated_query,
                "feedback_text": feedback_text,
                "conversation_id": conversation_id,
                "metadata": metadata or {}
            })
            
            # Then store as a failed query
            # First generate embedding
            embedding = None
            tables_used = []
            
            try:
                # Get embedding for the query
                from app.services.embedding_provider import EmbeddingProvider
                embedding_provider = EmbeddingProvider()
                embedding = await embedding_provider.get_embedding(original_query)
                
                # Parse metadata for tables if available
                if metadata and isinstance(metadata, dict) and 'tables_used' in metadata:
                    tables_used = metadata['tables_used']
            except Exception as embed_error:
                logger.error(f"Error generating embedding: {str(embed_error)}")
                # Continue without embedding
            
            # Try to store in database
            try:
                conn = await self._get_db_connection()
                try:
                    # Convert tables to JSON if needed
                    tables_json = json.dumps(tables_used) if tables_used else '{}'
                    
                    # Format embedding for database if available
                    embedding_formatted = None
                    if embedding:
                        if isinstance(embedding, list):
                            embedding_formatted = '[' + ','.join(str(float(x)) for x in embedding) + ']'
                    
                    # Store in failed queries table
                    query = """
                    INSERT INTO failed_queries 
                    (user_id, original_query, generated_query, feedback_text, conversation_id, 
                     embedding, created_at, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8)
                    RETURNING id
                    """
                    
                    failed_id = await conn.fetchval(
                        query,
                        user_id, 
                        original_query,
                        generated_query,
                        feedback_text,
                        conversation_id,
                        embedding_formatted,
                        datetime.now(),
                        json.dumps(metadata or {})
                    )
                    
                    logger.info(f"Saved failed query to database with ID {failed_id}")
                    feedback_entry["failed_id"] = failed_id
                except Exception as db_error:
                    # Log error but continue with in-memory fallback
                    logger.error(f"Error saving failed query to database: {str(db_error)}")
                    # Fallback to in-memory storage
                    failed_id = str(uuid.uuid4())
                    self.failed_queries[failed_id] = {
                        "id": failed_id,
                        "user_id": user_id,
                        "original_query": original_query,
                        "generated_query": generated_query,
                        "feedback_text": feedback_text,
                        "conversation_id": conversation_id,
                        "embedding": embedding,
                        "created_at": datetime.now().isoformat(),
                        "metadata": metadata or {}
                    }
                    feedback_entry["failed_id"] = failed_id
                finally:
                    await conn.close()
            except Exception as conn_error:
                # Log connection error and use in-memory store
                logger.error(f"Database connection error: {str(conn_error)}")
                failed_id = str(uuid.uuid4())
                self.failed_queries[failed_id] = {
                    "id": failed_id,
                    "user_id": user_id,
                    "original_query": original_query,
                    "generated_query": generated_query,
                    "feedback_text": feedback_text,
                    "conversation_id": conversation_id,
                    "embedding": embedding,
                    "created_at": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                feedback_entry["failed_id"] = failed_id
            
            return feedback_entry
        
        except Exception as e:
            logger.error(f"Error in save_negative_feedback: {str(e)}", exc_info=True)
            # Return minimal response on error
            return {"id": str(uuid.uuid4()), "error": str(e)}
    
    async def find_similar_verified_queries(self, 
                                          query_text: str, 
                                          user_id: str = None, 
                                          similarity_threshold: float = 0.6,
                                          limit: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar verified queries using vector similarity.
        Prioritizes the user's own verified queries before others.
        
        Args:
            query_text: The query text to find similar queries for
            user_id: Optional user ID to prioritize their examples
            similarity_threshold: Minimum similarity threshold
            limit: Maximum number of results to return
            
        Returns:
            List of similar verified queries with similarity scores
        """
        try:
            # Get embedding for the query
            from app.services.embedding_provider import EmbeddingProvider
            embedding_provider = EmbeddingProvider()
            query_embedding = await embedding_provider.get_embedding(query_text)
            
            if not query_embedding:
                logger.error("Failed to generate embedding for query text")
                return []
            
            # Format embedding for database query
            if isinstance(query_embedding, list):
                query_embedding_str = '[' + ','.join(str(float(x)) for x in query_embedding) + ']'
            else:
                logger.error(f"Invalid embedding format: {type(query_embedding)}")
                return []
            
            # Try to search in the database
            try:
                conn = await self._get_db_connection()
                try:
                    results = []
                    
                    # First try user's own queries if user_id is provided
                    if user_id:
                        user_query = """
                        SELECT 
                            id, 
                            original_query, 
                            generated_query, 
                            1 - (embedding <=> $1::vector) AS similarity 
                        FROM 
                            verified_queries 
                        WHERE 
                            user_id = $2 AND
                            1 - (embedding <=> $1::vector) > $3
                        ORDER BY 
                            similarity DESC 
                        LIMIT $4
                        """
                        
                        user_results = await conn.fetch(
                            user_query, 
                            query_embedding_str, 
                            user_id, 
                            similarity_threshold, 
                            limit
                        )
                        
                        for row in user_results:
                            results.append({
                                "id": row["id"],
                                "original_query": row["original_query"],
                                "generated_query": row["generated_query"],
                                "similarity": row["similarity"],
                                "is_user_specific": True
                            })
                    
                    # If we don't have enough results, try public queries from other users
                    remaining = limit - len(results)
                    if remaining > 0:
                        others_query = """
                        SELECT 
                            id, 
                            original_query, 
                            generated_query, 
                            1 - (embedding <=> $1::vector) AS similarity 
                        FROM 
                            verified_queries 
                        WHERE 
                            is_public = true AND
                            user_id != $2 AND
                            1 - (embedding <=> $1::vector) > $3
                        ORDER BY 
                            similarity DESC 
                        LIMIT $4
                        """
                        
                        # Use empty string if user_id is None to prevent SQL errors
                        safe_user_id = user_id if user_id else ""
                        
                        other_results = await conn.fetch(
                            others_query, 
                            query_embedding_str, 
                            safe_user_id, 
                            similarity_threshold, 
                            remaining
                        )
                        
                        for row in other_results:
                            results.append({
                                "id": row["id"],
                                "original_query": row["original_query"],
                                "generated_query": row["generated_query"],
                                "similarity": row["similarity"],
                                "is_user_specific": False
                            })
                    
                    return results
                finally:
                    await conn.close()
            except Exception as db_error:
                logger.error(f"Database error in find_similar_verified_queries: {str(db_error)}")
                # Fall back to in-memory search if database fails
            
            # In-memory fallback implementation
            if not hasattr(self, 'verified_queries') or not self.verified_queries:
                return []
                
            import numpy as np
            from scipy.spatial.distance import cosine
            
            similarities = []
            
            # First check the user's own queries
            if user_id:
                for vid, vq in self.verified_queries.items():
                    if vq.get('user_id') != user_id:
                        continue
                        
                    if vq.get('embedding') is None:
                        continue
                        
                    # Calculate similarity
                    similarity = 1 - cosine(query_embedding, vq['embedding'])
                    
                    if similarity >= similarity_threshold:
                        similarities.append((similarity, vq, True))
            
            # If we don't have enough user-specific results, check public queries
            if len(similarities) < limit:
                for vid, vq in self.verified_queries.items():
                    # Skip the user's own queries (already processed)
                    if user_id and vq.get('user_id') == user_id:
                        continue
                        
                    # Skip non-public queries from other users
                    if vq.get('user_id') != user_id and not vq.get('is_public', False):
                        continue
                        
                    if vq.get('embedding') is None:
                        continue
                        
                    # Calculate similarity
                    similarity = 1 - cosine(query_embedding, vq['embedding'])
                    
                    if similarity >= similarity_threshold:
                        similarities.append((similarity, vq, False))
            
            # Sort by similarity (descending) and take top 'limit'
            similarities.sort(reverse=True, key=lambda x: x[0])
            top_matches = similarities[:limit]
            
            # Format results
            results = []
            for similarity, vq, is_user_specific in top_matches:
                results.append({
                    "id": vq["id"],
                    "original_query": vq["original_query"],
                    "generated_query": vq["generated_query"],
                    "similarity": similarity,
                    "is_user_specific": is_user_specific
                })
                
            return results
                
        except Exception as e:
            logger.error(f"Error in find_similar_verified_queries: {str(e)}", exc_info=True)
            return []
    
    async def get_feedback_stats(self, user_id: str = None) -> Dict[str, Any]:
        """
        Get feedback statistics, optionally filtered by user.
        
        Args:
            user_id: Optional user ID to filter statistics
            
        Returns:
            Dictionary with statistics
        """
        try:
            # Try to get stats from database
            try:
                conn = await self._get_db_connection()
                try:
                    stats = {}
                    
                    # Get overall feedback counts
                    if user_id:
                        query = """
                        SELECT 
                            feedback_type, 
                            COUNT(*) as count 
                        FROM 
                            query_feedback 
                        WHERE 
                            user_id = $1 
                        GROUP BY 
                            feedback_type
                        """
                        rows = await conn.fetch(query, user_id)
                    else:
                        query = """
                        SELECT 
                            feedback_type, 
                            COUNT(*) as count 
                        FROM 
                            query_feedback 
                        GROUP BY 
                            feedback_type
                        """
                        rows = await conn.fetch(query)
                    
                    # Process results
                    total_count = 0
                    positive_count = 0
                    negative_count = 0
                    
                    for row in rows:
                        if row["feedback_type"] == "positive":
                            positive_count = row["count"]
                        elif row["feedback_type"] == "negative":
                            negative_count = row["count"]
                        
                        total_count += row["count"]
                    
                    # Calculate percentages
                    positive_pct = round(positive_count / total_count * 100, 2) if total_count > 0 else 0
                    negative_pct = round(negative_count / total_count * 100, 2) if total_count > 0 else 0
                    
                    stats = {
                        "total_count": total_count,
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                        "positive_percentage": positive_pct,
                        "negative_percentage": negative_pct
                    }
                    
                    return stats
                finally:
                    await conn.close()
            except Exception as db_error:
                logger.error(f"Database error in get_feedback_stats: {str(db_error)}")
                # Fall back to in-memory calculation
            
            # In-memory fallback
            total_count = len(self.feedback_store)
            positive_count = 0
            negative_count = 0
            
            for feedback in self.feedback_store.values():
                if user_id and feedback.get("user_id") != user_id:
                    continue
                    
                if feedback.get("feedback_type") == "positive":
                    positive_count += 1
                elif feedback.get("feedback_type") == "negative":
                    negative_count += 1
            
            # If filtering by user, adjust the total
            if user_id:
                total_count = positive_count + negative_count
            
            return {
                "total_count": total_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "positive_percentage": round(positive_count / total_count * 100, 2) if total_count > 0 else 0,
                "negative_percentage": round(negative_count / total_count * 100, 2) if total_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error in get_feedback_stats: {str(e)}", exc_info=True)
            return {
                "total_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_percentage": 0,
                "negative_percentage": 0,
                "error": str(e)
            }