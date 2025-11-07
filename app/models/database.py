import uuid as uuid_pkg

from sqlalchemy import Column, Date, DateTime, ForeignKey, Integer, String, Text, Time
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class Document(Base):
    """Document metadata table."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4)
    filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    chunking_strategy = Column(String(50), nullable=False)
    total_chunks = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename})>"


class Chunk(Base):
    """Document chunk metadata table."""

    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    vector_id = Column(String(255), nullable=False)  # ID in vector database
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<Chunk(id={self.id}, document_id={self.document_id}, index={self.chunk_index})>"


class Booking(Base):
    """Interview booking table."""

    __tablename__ = "bookings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid_pkg.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    name = Column(String(100), nullable=False)
    email = Column(String(255), nullable=False)
    booking_date = Column(Date, nullable=False)
    booking_time = Column(Time, nullable=False)
    status = Column(String(50), default="pending", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<Booking(id={self.id}, name={self.name}, date={self.booking_date})>"
