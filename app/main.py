# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
from contextlib import asynccontextmanager

from app.routes import query, conversation, websocket, directives, feedback, schema_manager
from app.core.config import settings
from app.routes import schema_management
from app.core.db import db_pool
from app.routes import debug_schema

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize resources
    await db_pool.initialize()
    yield
    # Shutdown: Cleanup resources
    await db_pool.close()

app = FastAPI(
    title="Query Generator API",
    description="API for generating database queries from natural language",
    version="0.1.0",
    lifespan=lifespan,
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router, prefix="/api/v1", tags=["query"])
app.include_router(conversation.router, prefix="/api/v1", tags=["conversation"])
app.include_router(directives.router, prefix="/api/v1", tags=["directives"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
app.include_router(schema_management.router, prefix="/api/v1", tags=["schema"])
app.include_router(schema_manager.router, prefix="/api/v1", tags=["schema-manager"])
app.include_router(debug_schema.router, prefix="/api/v1", tags=["debug"])

# Socket.IO setup
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins=settings.CORS_ORIGINS
)
socket_app = socketio.ASGIApp(sio)
app.mount("/ws", socket_app)

# Register Socket.IO events
websocket.register_handlers(sio)

@app.get("/")
async def root():
    return {"message": "Welcome to Query Generator API"}