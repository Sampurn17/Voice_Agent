from contextlib import asynccontextmanager
from typing import AsyncGenerator
from pathlib import Path

import uvicorn
import logging

from fastapi import FastAPI
from fastapi.responses import FileResponse
from src.api.routes import todo, reminder, calendar_event, call, vapi_webhook
from src.models.database import create_tables

logging.basicConfig(
    level=getattr(logging, "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting application, initializing database...")
    create_tables()
    logger.info("Application startup complete!")
    yield
    logger.info("Application shutdown")

app = FastAPI(title="Personal Assistant Voice Agent", lifespan=lifespan)

# Include routers
app.include_router(todo.router)
app.include_router(reminder.router)
app.include_router(calendar_event.router)
app.include_router(call.router)
app.include_router(vapi_webhook.router)

# Serve the voice assistant frontend at root
INDEX_HTML = Path(__file__).resolve().parent.parent.parent / "index.html"

@app.get("/")
async def serve_frontend():
    return FileResponse(INDEX_HTML, media_type="text/html")

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)