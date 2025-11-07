"""Qdrant vector database client for storing and querying embeddings."""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Qdrant vector store wrapper."""

    def __init__(self):
        """Initialize Qdrant client."""
        self.client: QdrantClient | None = None
        self.collection_name = settings.qdrant_collection_name
        self.embedding_dimension = 384  # For sentence-transformers/all-MiniLM-L6-v2
        self._connect()

    def _connect(self) -> None:
        """Establish connection to Qdrant."""
        try:
            if settings.qdrant_url:
                self.client = QdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key,
                )
            else:
                self.client = QdrantClient(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                )
            logger.info("Qdrant connection established")
            self._ensure_collection_exists()
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection already exists: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            raise

    def insert_vectors(
        self,
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> list[str]:
        """
        Insert vectors with metadata into Qdrant.

        Args:
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries

        Returns:
            List[str]: List of inserted vector IDs

        """
        try:
            points = []
            vector_ids = []

            for vector, payload in zip(vectors, payloads, strict=False):
                vector_id = str(uuid.uuid4())
                vector_ids.append(vector_id)

                points.append(PointStruct(id=vector_id, vector=vector, payload=payload))

            self.client.upsert(collection_name=self.collection_name, points=points)

            logger.info(f"Inserted {len(points)} vectors into Qdrant")
            return vector_ids

        except Exception as e:
            logger.error(f"Error inserting vectors: {e}")
            raise

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filter_conditions: Optional metadata filters

        Returns:
            List[Dict]: List of search results with scores and metadata

        """
        try:
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_vector,
                "limit": top_k,
            }

            if score_threshold:
                search_params["score_threshold"] = score_threshold

            if filter_conditions:
                search_params["query_filter"] = self._build_filter(filter_conditions)

            results = self.client.search(**search_params)

            return [
                {"id": result.id, "score": result.score, "payload": result.payload}
                for result in results
            ]

        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise

    def _build_filter(self, conditions: dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from conditions.

        Args:
            conditions: Dictionary of field: value pairs

        Returns:
            Filter: Qdrant filter object

        """
        must_conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in conditions.items()
        ]

        return Filter(must=must_conditions)

    def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all vectors for a document.

        Args:
            document_id: Document UUID

        Returns:
            bool: True if successful

        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        ),
                    ],
                ),
            )
            logger.info(f"Deleted vectors for document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False

    def close(self) -> None:
        """Close Qdrant connection."""
        if self.client:
            self.client.close()
            logger.info("Qdrant connection closed")


# Global vector store instance
vector_store = VectorStore()


def get_vector_store() -> VectorStore:
    """
    Dependency function to get vector store.

    Returns:
        VectorStore: Vector store instance

    """
    return vector_store
