"""Conversational RAG and booking API routes."""

import logging
import uuid
from datetime import datetime


from fastapi import APIRouter, Depends, HTTPException, Path, Query
from google.api_core.exceptions import ResourceExhausted
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.metadata_db import get_db
from app.db.redis_client import RedisClient, get_redis
from app.db.vector_db import VectorStore, get_vector_store
from app.models.schemas import (
    BookingListResponse,
    BookingResponse,
    BookingUpdateRequest,
    ChatQueryRequest,
    ChatQueryResponse,
    ChatSessionResponse,
    ConversationHistoryResponse,
    ConversationMessage,
    SourceChunk,
    SuccessResponse,
)
from app.models.database import Booking
from app.services.booking import BookingService, get_booking_service
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.memory import MemoryService, get_memory_service
from app.services.rag import RAGService, get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/session", response_model=ChatSessionResponse)
async def create_session(redis_client: RedisClient = Depends(get_redis)):
    """
    Create a new chat session.

    Returns session ID and expiration time.
    """
    try:
        memory_service = get_memory_service(redis_client)
        session_id, expires_at = memory_service.create_session()

        return ChatSessionResponse(
            session_id=session_id, created_at=datetime.utcnow(), expires_at=expires_at
        )

    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=ChatQueryResponse)
async def chat_query(
    request: ChatQueryRequest,
    db: Session = Depends(get_db),
    redis_client: RedisClient = Depends(get_redis),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStore = Depends(get_vector_store),
    booking_service: BookingService = Depends(get_booking_service),
):
    """
    Process a chat query with RAG and booking detection.

    - Retrieves relevant document chunks
    - Generates response using conversation context
    - Detects and extracts booking information
    - Maintains conversation history
    """
    try:
        # Initialize services
        memory_service = get_memory_service(redis_client)
        rag_service = get_rag_service(embedding_service, vector_store)

        # Verify session exists
        if not memory_service.session_exists(request.session_id):
            raise HTTPException(status_code=404, detail="Session not found or expired")

        # Get conversation history
        conversation_history = memory_service.get_context_window(
            request.session_id, num_turns=5
        )

        # Add user message to memory
        memory_service.add_message(request.session_id, "user", request.query)

        # Check for booking intent first (before RAG to avoid unnecessary processing)
        booking_detected = False
        booking_info = None
        booking_saved = False
        booking_id = None

        if booking_service.detect_booking_intent(request.query):
            logger.info("Booking intent detected, extracting information...")

            # Get full conversation for context
            full_history = memory_service.get_history(request.session_id)
            history_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in full_history]
            )
            logger.debug(f"Conversation history for booking extraction: {history_text[:200]}...")

            # Extract booking information
            extracted_booking = booking_service.extract_booking_info(
                conversation_history=history_text, current_message=request.query
            )

            if extracted_booking:
                logger.info(f"Booking extracted successfully: {extracted_booking}")
                booking_detected = True
                booking_info = extracted_booking

                # Save booking to database
                try:
                    saved_booking = booking_service.save_booking(
                        db=db,
                        session_id=request.session_id,
                        booking_info=extracted_booking,
                    )
                    booking_saved = True
                    booking_id = saved_booking.id
                    logger.info(
                        f"Booking saved successfully. ID: {booking_id}, "
                        f"Session: {request.session_id}"
                    )
                except Exception as e:
                    logger.error(f"Error saving booking: {e}", exc_info=True)
                    # Continue even if booking save fails
            else:
                logger.warning(
                    "Booking intent detected but extraction failed or returned no data"
                )

        # Perform RAG query (only if booking wasn't successfully saved)
        if not booking_saved:
            rag_result = rag_service.query(
                db=db,
                query=request.query,
                conversation_history=conversation_history,
                top_k=request.max_results,
                temperature=request.temperature,
            )

            response_text = rag_result["response"]
            sources = rag_result["sources"]
        else:
            # Generate booking confirmation response
            response_text = (
                f"Great! I've successfully booked your interview.\n\n"
                f"**Booking Details:**\n"
                f"- Booking ID: {booking_id}\n"
                f"- Name: {booking_info.name}\n"
                f"- Email: {booking_info.email}\n"
                f"- Date: {booking_info.date.strftime('%B %d, %Y')}\n"
                f"- Time: {booking_info.time.strftime('%I:%M %p')}\n\n"
                f"Your booking is currently pending confirmation. "
                f"You'll receive a confirmation email at {booking_info.email} shortly.\n\n"
                f"You can check your booking status using the Booking ID: {booking_id}"
            )
            sources = []

        # Add assistant response to memory
        memory_service.add_message(request.session_id, "assistant", response_text)

        # Format sources for response
        source_chunks = [
            SourceChunk(
                document_id=uuid.UUID(src["document_id"]),
                document_name=src["document_name"],
                chunk_id=src["chunk_id"],
                relevance_score=src["relevance_score"],
                content_preview=src["content_preview"],
            )
            for src in sources
        ]

        return ChatQueryResponse(
            response=response_text,
            sources=source_chunks,
            booking_detected=booking_detected,
            booking_info=booking_info,
            booking_id=booking_id,
            session_id=request.session_id,
            timestamp=datetime.utcnow(),
        )

    except ResourceExhausted as e:
        logger.warning("Gemini quota exceeded: %s", e)
        raise HTTPException(
            status_code=429,
            detail=(
                "Gemini API quota exceeded. Please wait a moment and try again or upgrade your plan."
            ),
        ) from e

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/sessions/{session_id}/history", response_model=ConversationHistoryResponse
)
async def get_conversation_history(
    session_id: uuid.UUID, redis_client: RedisClient = Depends(get_redis)
):
    """
    Get conversation history for a session.
    """
    try:
        memory_service = get_memory_service(redis_client)

        if not memory_service.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found or expired")

        messages = memory_service.get_history(session_id)

        conversation_messages = [
            ConversationMessage(
                role=msg["role"],
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg["timestamp"]),
            )
            for msg in messages
        ]

        return ConversationHistoryResponse(
            session_id=session_id,
            messages=conversation_messages,
            total_messages=len(conversation_messages),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SuccessResponse)
async def delete_session(
    session_id: uuid.UUID, redis_client: RedisClient = Depends(get_redis)
):
    """
    Delete a chat session.
    """
    try:
        memory_service = get_memory_service(redis_client)

        if not memory_service.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Session not found")

        success = memory_service.delete_session(session_id)

        if success:
            return SuccessResponse(
                success=True, message=f"Session {session_id} deleted successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to delete session")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bookings", response_model=BookingListResponse)
async def list_bookings(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    session_id: uuid.UUID | None = Query(None, description="Filter by session ID"),
    email: str | None = Query(None, description="Filter by email address"),
    db: Session = Depends(get_db),
    booking_service: BookingService = Depends(get_booking_service),
):
    """
    List bookings with pagination and optional filters.

    You can filter by:
    - session_id: Get all bookings for a specific session
    - email: Get all bookings for a specific email address
    """
    try:
        # Filter by session_id if provided
        if session_id:
            bookings = booking_service.get_bookings_by_session(db, session_id)
            total = len(bookings)
        # Filter by email if provided
        elif email:
            bookings = db.query(Booking).filter(Booking.email == email).offset(skip).limit(limit).all()
            total = db.query(Booking).filter(Booking.email == email).count()
        # Get all bookings
        else:
            bookings = booking_service.get_all_bookings(db, skip=skip, limit=limit)
            total = db.query(Booking).count()

        booking_responses = [
            BookingResponse(
                id=booking.id,
                session_id=booking.session_id,
                name=booking.name,
                email=booking.email,
                booking_date=booking.booking_date,
                booking_time=booking.booking_time,
                status=booking.status,
                created_at=booking.created_at,
            )
            for booking in bookings
        ]

        return BookingListResponse(bookings=booking_responses, total=total)

    except Exception as e:
        logger.error(f"Error listing bookings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bookings/{booking_id}", response_model=BookingResponse)
async def get_booking(
    booking_id: uuid.UUID = Path(..., description="Booking ID (UUID)"),
    db: Session = Depends(get_db),
    booking_service: BookingService = Depends(get_booking_service),
):
    """
    Get details of a specific booking.
    """
    try:
        booking = booking_service.get_booking_by_id(db, booking_id)

        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")

        return BookingResponse(
            id=booking.id,
            session_id=booking.session_id,
            name=booking.name,
            email=booking.email,
            booking_date=booking.booking_date,
            booking_time=booking.booking_time,
            status=booking.status,
            created_at=booking.created_at,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid booking ID format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid booking ID format: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting booking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/bookings/{booking_id}", response_model=BookingResponse)
async def update_booking(
    booking_id: uuid.UUID = Path(..., description="Booking ID (UUID)"),
    request: BookingUpdateRequest = ...,
    db: Session = Depends(get_db),
    booking_service: BookingService = Depends(get_booking_service),
):
    """
    Update booking status (pending, confirmed, cancelled).
    """
    try:
        # Convert enum to string value
        status_value = request.status.value if hasattr(request.status, 'value') else str(request.status)

        booking = booking_service.update_booking_status(
            db=db, booking_id=booking_id, status=status_value
        )

        if not booking:
            raise HTTPException(status_code=404, detail="Booking not found")

        return BookingResponse(
            id=booking.id,
            session_id=booking.session_id,
            name=booking.name,
            email=booking.email,
            booking_date=booking.booking_date,
            booking_time=booking.booking_time,
            status=booking.status,
            created_at=booking.created_at,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid booking ID or status format: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating booking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
