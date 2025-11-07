"""Text chunking strategies for document processing."""

import logging
import re

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for chunking text using different strategies."""

    @staticmethod
    def fixed_size_chunking(
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        """
        Split text into fixed-size chunks with overlap.

        Args:
            text: Input text
            chunk_size: Target chunk size in tokens (approximate)
            overlap: Number of overlapping tokens between chunks

        Returns:
            List[str]: List of text chunks
        """
        # Simple word-based approximation (1 token ≈ 0.75 words)
        words_per_chunk = int(chunk_size * 0.75)
        words_overlap = int(overlap * 0.75)

        # Split text into words
        words = text.split()

        if len(words) <= words_per_chunk:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = start + words_per_chunk
            chunk_words = words[start:end]
            chunk = " ".join(chunk_words)
            chunks.append(chunk)

            # Move start position with overlap
            start = end - words_overlap

            # Prevent infinite loop if overlap is too large
            if start <= 0:
                start = end

        logger.info(f"Created {len(chunks)} chunks using fixed-size strategy")
        return chunks

    @staticmethod
    def semantic_chunking(
        text: str,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000,
    ) -> list[str]:
        """

        Split text into semantic chunks based on paragraphs and sentences.

        Preserves natural boundaries while respecting size constraints.

        Args:
            text: Input text
            min_chunk_size: Minimum chunk size in tokens (approximate)
            max_chunk_size: Maximum chunk size in tokens (approximate)

        Returns:
            List[str]: List of text chunks

        """
        # Convert token counts to approximate word counts
        min_words = int(min_chunk_size * 0.75)
        max_words = int(max_chunk_size * 0.75)

        # Split into paragraphs (double newline or more)
        paragraphs = re.split(r"\n\s*\n+", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = []
        current_word_count = 0

        for paragraph in paragraphs:
            para_words = len(paragraph.split())

            # If single paragraph exceeds max, split it by sentences
            if para_words > max_words:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0

                # Split large paragraph into sentences
                sentences = ChunkingService._split_into_sentences(paragraph)
                sentence_chunk = []
                sentence_word_count = 0

                for sentence in sentences:
                    sent_words = len(sentence.split())

                    if sentence_word_count + sent_words > max_words and sentence_chunk:
                        chunks.append(" ".join(sentence_chunk))
                        sentence_chunk = [sentence]
                        sentence_word_count = sent_words
                    else:
                        sentence_chunk.append(sentence)
                        sentence_word_count += sent_words

                if sentence_chunk:
                    chunks.append(" ".join(sentence_chunk))

                continue

            # Check if adding this paragraph exceeds max chunk size
            if current_word_count + para_words > max_words and current_chunk:
                # Save current chunk if it meets minimum size
                if current_word_count >= min_words:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [paragraph]
                    current_word_count = para_words
                else:
                    # Current chunk too small, add paragraph anyway
                    current_chunk.append(paragraph)
                    current_word_count += para_words
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_word_count += para_words

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Filter out very small chunks and merge them
        chunks = ChunkingService._merge_small_chunks(chunks, min_words)

        logger.info(f"Created {len(chunks)} chunks using semantic strategy")
        return chunks

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            list[str]: List of sentences

        """
        # Simple sentence splitting (can be improved with NLP libraries)
        sentence_endings = r"[.!?]+[\s]+"
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    @staticmethod
    def _merge_small_chunks(chunks: list[str], min_words: int) -> list[str]:
        """
        Merge chunks that are too small.

        Args:
            chunks: List of text chunks
            min_words: Minimum word count

        Returns:
            list[str]: Merged chunks
        """
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for i in range(1, len(chunks)):
            current_words = len(current.split())
            next_chunk = chunks[i]

            if current_words < min_words:
                # Merge with next chunk
                current = current + " " + next_chunk
            else:
                # Save current and start new
                merged.append(current)
                current = next_chunk

        # Add last chunk
        merged.append(current)

        return merged

    @staticmethod
    def chunk_text(text: str, strategy: str = "fixed", **kwargs) -> list[str]:
        """
        Chunk text using specified strategy.

        Args:
            text: Input text
            strategy: "fixed" or "semantic"
            **kwargs: Strategy-specific parameters

        Returns:
            List[str]: List of text chunks
        """
        if strategy == "fixed":
            return ChunkingService.fixed_size_chunking(
                text,
                chunk_size=kwargs.get("chunk_size", 512),
                overlap=kwargs.get("overlap", 50),
            )
        if strategy == "semantic":
            return ChunkingService.semantic_chunking(
                text,
                min_chunk_size=kwargs.get("min_chunk_size", 200),
                max_chunk_size=kwargs.get("max_chunk_size", 1000),
            )
        raise ValueError(f"Unknown chunking strategy: {strategy}")


# Global chunking service instance
chunking_service = ChunkingService()


def get_chunking_service() -> ChunkingService:
    """
    Dependency function to get chunking service.

    Returns:
        ChunkingService: Chunking service instance

    """
    return chunking_service
