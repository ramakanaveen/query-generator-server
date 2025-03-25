from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

from app.routes import query, conversation, websocket, directives
from app.core.config import settings

app = FastAPI(
    title="Query Generator API",
    description="API for generating database queries from natural language",
    version="0.1.0",
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