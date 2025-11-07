"""Document ingestion API routes."""

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.metadata_db import get_db
from app.db.vector_db import VectorStore, get_vector_store
from app.models.database import Chunk, Document
from app.models.schemas import (
    ChunkingStrategy,
    DocumentListItem,
    DocumentListResponse,
    DocumentUploadResponse,
    SuccessResponse,
)
from app.services.chunking import ChunkingService, get_chunking_service
from app.services.document_processor import DocumentProcessor, get_document_processor
from app.services.embedding import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

router = APIRouter()

# === Module-level singletons to satisfy B008 ===
FileUpload = File(...)
FormChunkingStrategy = Form(ChunkingStrategy.FIXED)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = Annotated[UploadFile, FileUpload],
    chunking_strategy: ChunkingStrategy = Annotated[
        ChunkingStrategy, FormChunkingStrategy
    ],
    db: Session = Depends(get_db),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    chunking_service: ChunkingService = Depends(get_chunking_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentUploadResponse:
    """
    Upload and process a document (PDF or TXT).
    """
    temp_file_path: str | None = None

    try:
        # Validate file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {settings.allowed_extensions}",
            )

        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        # Validate file size
        if file_size > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {settings.max_file_size} bytes",
            )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=file_extension
        ) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        logger.info("Processing file: %s (%s bytes)", file.filename, file_size)

        # Extract text
        text = doc_processor.extract_text(temp_file_path, file_extension)

        if not text or len(text.strip()) == 0:
            raise HTTPException(
                status_code=400, detail="No text could be extracted from the document"
            )

        logger.info("Extracted %s characters from document", len(text))

        # Chunk text
        if chunking_strategy == ChunkingStrategy.FIXED:
            chunks = chunking_service.fixed_size_chunking(
                text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
        else:  # SEMANTIC
            chunks = chunking_service.semantic_chunking(text)

        logger.info("Created %s chunks", len(chunks))

        # Generate embeddings
        embeddings = embedding_service.generate_embeddings(chunks)

        logger.info("Generated %s embeddings", len(embeddings))

        # Create document record
        document = Document(
            filename=file.filename,
            file_size=file_size,
            chunking_strategy=chunking_strategy.value,
            total_chunks=len(chunks),
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # Prepare vector payloads
        payloads: list[dict] = []
        chunk_records: list[Chunk] = []

        for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = uuid.uuid4()

            payloads.append(
                {
                    "chunk_id": str(chunk_id),
                    "document_id": str(document.id),
                    "document_name": file.filename,
                    "chunk_index": i,
                    "content": chunk_text[:500],
                }
            )

            chunk_records.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk_text,
                    token_count=len(chunk_text.split()),
                    vector_id="",
                )
            )

        # Store vectors
        vector_ids = vector_store.insert_vectors(embeddings, payloads)

        # Update vector IDs
        for chunk_record, vector_id in zip(chunk_records, vector_ids):
            chunk_record.vector_id = vector_id

        # Save chunks
        db.bulk_save_objects(chunk_records)
        db.commit()

        logger.info("Successfully processed document: %s", document.id)

        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            total_chunks=document.total_chunks,
            chunking_strategy=ChunkingStrategy(document.chunking_strategy),
            status="success",
            created_at=document.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("Error processing document: %s", e, exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)  # noqa: PTH108


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
    db: Session = Depends(get_db),
):
    """
    List all uploaded documents with pagination.
    """
    try:
        # Get total count
        total = db.query(Document).count()

        # Get paginated documents
        documents = db.query(Document).offset(skip).limit(limit).all()

        document_items = [
            DocumentListItem(
                id=doc.id,
                filename=doc.filename,
                file_size=doc.file_size,
                chunking_strategy=ChunkingStrategy(doc.chunking_strategy),
                total_chunks=doc.total_chunks,
                created_at=doc.created_at,
            )
            for doc in documents
        ]

        return DocumentListResponse(
            documents=document_items,
            total=total,
            page=skip // limit + 1 if limit > 0 else 1,
            page_size=limit,
        )

    except Exception as e:
        logger.error(f"Error listing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
