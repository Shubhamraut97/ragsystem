import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Settings
    app_name: str = "RAG Booking System"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000

    google_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    # Database
    database_url: str

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    # Qdrant Vector Database
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection_name: str = "document_chunks"

    # Document Processing
    max_file_size: int = 10485760  # 10MB
    allowed_extensions: list[str] = [".pdf", ".txt"]

    # Chunking Settings
    chunk_size: int = 512
    chunk_overlap: int = 50

    top_k_results: int = 5
    similarity_threshold: float = 0.7
    temperature: float = 0.7
    max_tokens: int = 4096  # Increased for longer responses
    max_context_chars: int = 4000  # Limit context size to prevent token overflow

    conversation_memory_limit: int = 10
    session_ttl: int = 86400

    # Model config for Pydantic V2
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def qdrant_connection_url(self) -> str:
        """Get Qdrant connection URL."""
        if self.qdrant_url:
            return self.qdrant_url
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


settings = Settings()
