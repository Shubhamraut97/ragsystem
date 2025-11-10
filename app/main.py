"""
Main FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.metadata_db import close_db, init_db
from app.db.redis_client import redis_client
from app.db.vector_db import vector_store
from app.models.schemas import ErrorResponse, HealthCheckResponse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting up RAG Booking System...")
    try:
        init_db()
        logger.info(" Database initialized")
        logger.info(" Redis connected")
        logger.info(" Qdrant connected")
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise

    yield


    logger.info("Shutting down...")
    close_db()
    redis_client.close()
    vector_store.close()
    logger.info("Application shutdown complete")



app = FastAPI(
    title=settings.app_name,
    description="RAG System with Interview Booking ",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            details={"error": str(exc)} if settings.debug else None,
        ).model_dump(),
    )


# Health check endpoint
@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint with health check."""
    try:
        # Check services
        services = {
            "database": "healthy",
            "redis": "healthy" if redis_client.client.ping() else "unhealthy",
            "vector_db": "healthy",
        }

        return HealthCheckResponse(
            status="healthy", timestamp=datetime.utcnow(), services=services
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content=HealthCheckResponse(
                status="unhealthy",
                timestamp=datetime.utcnow(),
                services={"error": str(e)},
            ).model_dump(),
        )


@app.get("/health")
async def health_check():
    """Dedicated health check endpoint."""
    return await root()


# Import and include routers
from app.api import chat, ingest

app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Document Ingestion"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Conversational RAG"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, reload=settings.debug
    )
