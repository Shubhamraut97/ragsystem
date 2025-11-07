"""PostgreSQL database connection and session management."""

import logging
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.models.database import Base

logger = logging.getLogger(__name__)


# Create engine
engine = create_engine(
    settings.database_url,
    poolclass=NullPool,
    echo=settings.debug,
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.exception("Error creating database tables: %s", e)
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.

    Yields:
        Session: SQLAlchemy database session

    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def close_db() -> None:
    """Close database connections."""
    engine.dispose()
    logger.info("Database connections closed")
