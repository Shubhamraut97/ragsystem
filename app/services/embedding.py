"""Embedding generation service using Google Gemini or Sentence Transformers."""

import logging

import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


genai.configure(api_key=settings.google_api_key)


class EmbeddingService:
    """Service for generating embeddings."""

    def __init__(self, use_gemini: bool = False):
        """
        Initialize embedding service.

        Args:
            use_gemini: If True, use Google Gemini embeddings (paid).
                       If False, use sentence-transformers (free).

        """
        self.use_gemini = use_gemini
        self.model = None

        if not use_gemini:
            # Use free sentence-transformers
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.dimension = 384  # Dimension for all-MiniLM-L6-v2
            logger.info("Using sentence-transformers for embeddings (free)")
        else:
            self.dimension = 768  # Dimension for Gemini embeddings
            logger.info("Using Google Gemini for embeddings")

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            List[float]: Embedding vector

        """
        try:
            if self.use_gemini:
                return self._generate_gemini_embedding(text)
            return self._generate_sentence_transformer_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batch).

        Args:
            texts: List of input texts

        Returns:
            List[List[float]]: List of embedding vectors

        """
        try:
            if self.use_gemini:
                return [self._generate_gemini_embedding(text) for text in texts]
            else:
                return self._generate_sentence_transformer_embeddings(texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def _generate_gemini_embedding(self, text: str) -> list[float]:
        """Generate embedding using Google Gemini."""
        result = genai.embed_content(
            model=settings.gemini_embedding_model,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    def _generate_sentence_transformer_embedding(self, text: str) -> list[float]:
        """Generate embedding using sentence-transformers."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def _generate_sentence_transformer_embeddings(
        self, texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings using sentence-transformers (batch)."""
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False,
        )
        return embeddings.tolist()

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            int: Embedding vector dimension

        """
        return self.dimension


embedding_service = EmbeddingService(use_gemini=False)


def get_embedding_service() -> EmbeddingService:
    """
    Dependency function to get embedding service.

    Returns:
        EmbeddingService: Embedding service instance

    """
    return embedding_service
