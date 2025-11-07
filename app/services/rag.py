"""Custom RAG (Retrieval-Augmented Generation)."""

import logging
from typing import Any

import google.generativeai as genai
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.vector_db import VectorStore
from app.models.database import Chunk, Document
from app.services.embedding import EmbeddingService

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=settings.google_api_key)


class RAGService:
    """Custom RAG service without using RetrievalQAChain."""

    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        """
        Initialize RAG service.

        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector database client
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm = genai.GenerativeModel(settings.gemini_model)

    def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score

        Returns:
            List[Dict]: Retrieved chunks with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)

            # Search in vector database
            results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                score_threshold=score_threshold,
            )

            if not results and score_threshold is not None:
                logger.info(
                    "No chunks met similarity threshold %.2f; retrying without threshold",
                    score_threshold,
                )
                results = self.vector_store.search(
                    query_vector=query_embedding,
                    top_k=top_k,
                    score_threshold=None,
                )

            logger.info(f"Retrieved {len(results)} chunks for query")
            return results

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

    def generate_response(
        self,
        query: str,
        context_chunks: list[dict[str, Any]],
        conversation_history: str = "",
        temperature: float = 0.7,
    ) -> str:
        """
        Generate response using LLM with retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved document chunks
            conversation_history: Previous conversation
            temperature: LLM temperature

        Returns:
            str: Generated response
        """
        try:
            # Build context from chunks
            context = self._build_context(context_chunks)

            # Build prompt
            prompt = self._build_prompt(query, context, conversation_history)

            # Generate response
            response = self.llm.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=settings.max_tokens,
                ),
            )

            response_text = self._extract_response_text(response)
            if response_text:
                # Check if response was truncated (finish_reason 2 = MAX_TOKENS)
                finish_reasons = []
                for candidate in getattr(response, "candidates", []):
                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason is not None:
                        finish_reasons.append(str(finish_reason))

                # If truncated, append a note to the response
                if "2" in finish_reasons:
                    response_text += "\n\n[Note: Response was truncated due to token limit. Consider rephrasing your query for a complete answer.]"

                return response_text

            # No text extracted - check finish reasons to provide helpful error message
            finish_reasons = []
            for candidate in getattr(response, "candidates", []):
                finish_reason = getattr(candidate, "finish_reason", None)
                if finish_reason is not None:
                    finish_reasons.append(str(finish_reason))

            # Finish reason 2 = MAX_TOKENS (response was truncated)
            if "2" in finish_reasons:
                error_msg = (
                    "The response exceeded the maximum token limit and no text could be extracted. "
                    "Please try rephrasing your query or breaking it into smaller parts."
                )
            else:
                error_msg = (
                    f"Gemini API returned no text. Finish reasons: {', '.join(finish_reasons) if finish_reasons else 'unknown'}"
                )

            raise ValueError(error_msg)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            str: Formatted context
        """
        if not chunks:
            return "No relevant context found."

        max_context_chars = settings.max_context_chars
        context_parts = []
        current_length = 0

        for i, chunk in enumerate(chunks, 1):
            payload = chunk.get("payload", {})
            content = payload.get("content", "")
            doc_name = payload.get("document_name", "Unknown")

            # Format the chunk with source info
            chunk_header = f"[Source {i} - {doc_name}]\n"
            chunk_footer = "\n"
            header_footer_length = len(chunk_header) + len(chunk_footer)

            # Calculate available space for this chunk
            remaining_space = max_context_chars - current_length - header_footer_length

            if remaining_space <= 0:
                # No more space, stop adding chunks
                logger.warning(
                    f"Context limit reached after {i-1} chunks. Truncating context."
                )
                break

            # Truncate content if needed
            if len(content) > remaining_space:
                content = content[:remaining_space - 3] + "..."
                logger.debug(f"Truncated chunk {i} content to fit context limit")

            chunk_text = chunk_header + content + chunk_footer
            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n".join(context_parts)

    def _build_prompt(
        self, query: str, context: str, conversation_history: str = ""
    ) -> str:
        """
        Build prompt for LLM.

        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous conversation

        Returns:
            str: Complete prompt
        """
        prompt_parts = [
            "You are a helpful AI assistant that answers questions based on provided documents.",
            "Use the context below to answer the user's question accurately.",
            "If the context doesn't contain relevant information, say so clearly.",
            "Always cite which source you're using when providing information.",
            "",
        ]

        if conversation_history:
            prompt_parts.extend(["Previous Conversation:", conversation_history, ""])

        prompt_parts.extend(
            [
                "Context from Documents:",
                context,
                "",
                f"User Question: {query}",
                "",
                "Answer (cite sources using [Source N] format):",
            ]
        )

        return "\n".join(prompt_parts)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text from Gemini response handling missing quick accessor."""

        text: str | None = None

        # Try the quick accessor first
        try:
            text = response.text
            if text:
                return text
        except Exception:
            # Quick accessor failed, fall through to manual extraction
            pass

        # Manual extraction from candidates
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue

            parts = getattr(content, "parts", None) or []
            collected_parts: list[str] = []

            for part in parts:
                # Try multiple ways to get text from part
                part_text = None

                # Method 1: Direct text attribute
                if hasattr(part, "text"):
                    part_text = getattr(part, "text", None)

                # Method 2: Check if it's a dict-like object
                if not part_text and isinstance(part, dict):
                    part_text = part.get("text")

                # Method 3: String representation if it's a simple type
                if not part_text and isinstance(part, str):
                    part_text = part

                if part_text:
                    collected_parts.append(str(part_text))

            if collected_parts:
                return "".join(collected_parts)

        return ""

    def query(
        self,
        db: Session,
        query: str,
        conversation_history: str = "",
        top_k: int = 5,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """
        Complete RAG query pipeline.

        Args:
            db: Database session
            query: User query
            conversation_history: Previous conversation
            top_k: Number of chunks to retrieve
            temperature: LLM temperature

        Returns:
            Dict: Response with answer and sources
        """
        try:
            # Step 1: Retrieve relevant chunks
            retrieved_chunks = self.retrieve_relevant_chunks(
                query=query, top_k=top_k, score_threshold=settings.similarity_threshold
            )

            # Step 2: Enrich chunks with full metadata from database
            enriched_chunks = self._enrich_chunks_with_metadata(db, retrieved_chunks)

            # Step 3: Generate response
            response_text = self.generate_response(
                query=query,
                context_chunks=enriched_chunks,
                conversation_history=conversation_history,
                temperature=temperature,
            )

            # Step 4: Format sources
            sources = self._format_sources(enriched_chunks)

            return {
                "response": response_text,
                "sources": sources,
                "num_sources": len(sources),
            }

        except Exception as e:
            logger.error(f"Error in RAG query: {e}")
            raise

    def _enrich_chunks_with_metadata(
        self, db: Session, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Enrich retrieved chunks with metadata from database.

        Args:
            db: Database session
            chunks: Retrieved chunks from vector DB

        Returns:
            List[Dict]: Enriched chunks
        """
        enriched = []

        for chunk in chunks:
            payload = chunk.get("payload", {})
            chunk_id = payload.get("chunk_id")

            if chunk_id:
                # Get full chunk data from database
                db_chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()

                if db_chunk:
                    # Get document info
                    document = (
                        db.query(Document)
                        .filter(Document.id == db_chunk.document_id)
                        .first()
                    )

                    enriched.append(
                        {
                            "id": chunk.get("id"),
                            "score": chunk.get("score"),
                            "payload": {
                                "content": db_chunk.content,
                                "chunk_id": str(db_chunk.id),
                                "document_id": str(db_chunk.document_id),
                                "document_name": document.filename
                                if document
                                else "Unknown",
                                "chunk_index": db_chunk.chunk_index,
                            },
                        }
                    )
                else:
                    enriched.append(chunk)
            else:
                enriched.append(chunk)

        return enriched

    def _format_sources(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format source information for API response.

        Args:
            chunks: Enriched chunks

        Returns:
            List[Dict]: Formatted source information
        """
        sources = []

        max_preview_chars = 200

        for chunk in chunks:
            payload = chunk.get("payload", {})
            content = payload.get("content", "")

            # Create preview capped at 200 characters (including ellipsis)
            if len(content) > max_preview_chars:
                preview = content[: max_preview_chars - 3] + "..."
            else:
                preview = content

            sources.append(
                {
                    "document_id": payload.get("document_id"),
                    "document_name": payload.get("document_name"),
                    "chunk_id": payload.get("chunk_id"),
                    "relevance_score": round(chunk.get("score", 0), 3),
                    "content_preview": preview,
                }
            )

        return sources


def get_rag_service(
    embedding_service: EmbeddingService, vector_store: VectorStore
) -> RAGService:
    """
    Dependency function to get RAG service.

    Args:
        embedding_service: Embedding service
        vector_store: Vector store

    Returns:
        RAGService: RAG service instance
    """
    return RAGService(embedding_service, vector_store)
