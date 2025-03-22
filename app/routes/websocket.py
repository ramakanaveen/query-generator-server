import socketio
import uuid
import json
from typing import Dict, Any

from app.services.llm_provider import LLMProvider
from app.services.conversation_manager import ConversationManager
from app.services.query_generator import QueryGenerator
from app.services.database_connector import DatabaseConnector

# Store active connections
connected_clients = {}

def register_handlers(sio: socketio.AsyncServer):
    @sio.event
    async def connect(sid, environ, auth):
        """Handle client connection."""
        print(f"Client connected: {sid}")
        connected_clients[sid] = {
            "conversation_id": auth.get("conversation_id") if auth else None
        }
        await sio.emit("status", {"status": "connected"}, to=sid)

    @sio.event
    async def disconnect(sid):
        """Handle client disconnection."""
        print(f"Client disconnected: {sid}")
        if sid in connected_clients:
            del connected_clients[sid]

    @sio.event
    async def query(sid, data):
        """Handle query requests."""
        try:
            # Parse data
            if isinstance(data, str):
                data = json.loads(data)
            
            user_query = data.get("content", "")
            model_name = data.get("model", "gemini")
            conversation_id = connected_clients[sid].get("conversation_id")
            
            # Initialize services
            llm_provider = LLMProvider()
            llm = llm_provider.get_model(model_name)
            query_generator = QueryGenerator(llm)
            
            # Send thinking status
            await sio.emit("status", {"type": "thinking"}, to=sid)
            
            # Generate query
            database_type = "kdb"  # Default, could be extracted from query
            generated_query, thinking = await query_generator.generate(
                user_query,
                database_type,
                conversation_id
            )
            
            # Send generated query
            execution_id = str(uuid.uuid4())
            await sio.emit("query_generated", {
                "content": generated_query,
                "execution_id": execution_id,
                "thinking": thinking
            }, to=sid)
            
            # Store in conversation if needed
            if conversation_id:
                conversation_manager = ConversationManager()
                await conversation_manager.add_message(
                    conversation_id,
                    {
                        "id": str(uuid.uuid4()),
                        "role": "user",
                        "content": user_query
                    }
                )
                await conversation_manager.add_message(
                    conversation_id,
                    {
                        "id": str(uuid.uuid4()),
                        "role": "assistant",
                        "content": generated_query,
                        "metadata": {"execution_id": execution_id}
                    }
                )
        
        except Exception as e:
            await sio.emit("error", {"message": str(e)}, to=sid)

    @sio.event
    async def execute(sid, data):
        """Handle query execution requests."""
        try:
            # Parse data
            if isinstance(data, str):
                data = json.loads(data)
            
            query = data.get("query", "")
            params = data.get("params", {})
            
            # Send executing status
            await sio.emit("status", {"type": "executing"}, to=sid)
            
            # Execute query
            db_connector = DatabaseConnector()
            results, metadata = await db_connector.execute(query, params)
            
            # Send results
            await sio.emit("results", {
                "results": results,
                "metadata": metadata
            }, to=sid)
        
        except Exception as e:
            await sio.emit("error", {"message": str(e)}, to=sid)