import uuid
from datetime import date, datetime, time
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

# ============== Enums ==============


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"
    SEMANTIC = "semantic"


class BookingStatus(str, Enum):
    """Booking status options."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


# ============== Document Ingestion ==============


class DocumentUploadResponse(BaseModel):
    """Response after successful document upload."""

    document_id: uuid.UUID
    filename: str
    total_chunks: int
    chunking_strategy: ChunkingStrategy
    status: str = "success"
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentListItem(BaseModel):
    """Individual document in list response."""

    id: uuid.UUID
    filename: str
    file_size: int
    chunking_strategy: ChunkingStrategy
    total_chunks: int
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """List of documents response."""

    documents: list[DocumentListItem]
    total: int
    page: int
    page_size: int


# ============== Chat/RAG ==============


class ChatSessionResponse(BaseModel):
    """Response when creating a new chat session."""

    session_id: uuid.UUID
    created_at: datetime
    expires_at: datetime


class SourceChunk(BaseModel):
    """Source information for RAG response."""

    document_id: uuid.UUID
    document_name: str
    chunk_id: str
    relevance_score: float
    content_preview: str = Field(..., max_length=200)


class BookingInfo(BaseModel):
    """
    Extracted booking information.

    We use native date/time types and V2 field_validators for conversion/validation.
    """

    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr

    date: date
    time: time

    @field_validator("time", mode="before")
    @classmethod
    def validate_time_format(cls, v: str) -> time:
        """Validate and convert time string (HH:MM) to datetime.time object."""
        try:
            return datetime.strptime(v, "%H:%M").time()
        except ValueError as e:
            # Re-raise Pydantic's expected error type with a clear message
            raise ValueError("Time must be in HH:MM format (e.g., 14:30)") from e

    @field_validator("date", mode="before")
    @classmethod
    def validate_date_format(cls, v: str) -> date:
        """Validate and convert date string (YYYY-MM-DD) to datetime.date object."""
        try:
            return datetime.strptime(v, "%Y-%m-%d").date()
        except ValueError as e:
            raise ValueError("Date must be in YYYY-MM-DD format") from e


class ChatQueryRequest(BaseModel):
    """Request for chat query."""

    session_id: uuid.UUID
    query: str = Field(..., min_length=1, max_length=2000)
    max_results: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class ChatQueryResponse(BaseModel):
    """Response from chat query."""

    response: str
    sources: list[SourceChunk]
    booking_detected: bool = False
    booking_info: BookingInfo | None = None
    booking_id: uuid.UUID | None = None  # Booking ID when booking is successfully created
    session_id: uuid.UUID
    timestamp: datetime


class ConversationMessage(BaseModel):
    """Single message in conversation history."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


class ConversationHistoryResponse(BaseModel):
    """Full conversation history."""

    session_id: uuid.UUID
    messages: list[ConversationMessage]
    total_messages: int


# ============== Bookings ==============


class BookingCreateRequest(BaseModel):
    """Request to create a booking (internal use)."""

    session_id: uuid.UUID
    name: str
    email: EmailStr
    booking_date: date
    booking_time: time


class BookingResponse(BaseModel):
    """Booking information response."""

    id: uuid.UUID
    session_id: uuid.UUID
    name: str
    email: str
    booking_date: date
    booking_time: time
    status: BookingStatus
    created_at: datetime

    class Config:
        from_attributes = True


class BookingListResponse(BaseModel):
    """List of bookings."""

    bookings: list[BookingResponse]
    total: int


class BookingUpdateRequest(BaseModel):
    """Update booking status."""

    status: BookingStatus


# ============== Common Responses ==============


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = True
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = False
    error_code: str
    message: str
    details: dict | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: datetime
    services: dict[str, str]
