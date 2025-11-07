"""
Booking service for extracting and managing interview bookings.
"""

import json
import logging
import uuid
from datetime import date, datetime, time
from typing import Any, Optional

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.database import Booking
from app.models.schemas import BookingInfo

logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=settings.google_api_key)


class BookingService:
    """Service for extracting booking information using LLM."""

    def __init__(self):
        """Initialize booking service."""
        self.model = genai.GenerativeModel(settings.gemini_model)

    def detect_booking_intent(self, text: str) -> bool:
        """
        Detect if user wants to book an interview.

        Args:
            text: User message

        Returns:
            bool: True if booking intent detected
        """
        booking_keywords = [
            "book",
            "schedule",
            "appointment",
            "interview",
            "meeting",
            "reserve",
            "set up",
            "arrange",
            "slot",
            "time",
            "date",
            "calendar",
        ]

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in booking_keywords)

    def extract_booking_info(
        self, conversation_history: str, current_message: str
    ) -> BookingInfo | None:
        """
        Extract booking information from conversation using Gemini.

        Args:
            conversation_history: Previous conversation context
            current_message: Current user message

        Returns:
            Optional[BookingInfo]: Extracted booking info or None
        """
        try:
            prompt = self._build_extraction_prompt(
                conversation_history, current_message
            )

            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low temperature for structured extraction
                ),
            )

            # Extract text from response (handle potential errors)
            response_text = self._extract_response_text(response)
            if not response_text:
                logger.warning("No text extracted from booking extraction response")
                # Log the raw response for debugging
                logger.debug(f"Raw response object: {response}")
                return None

            logger.debug(f"Raw booking extraction response text: {response_text}")

            # Parse JSON response
            result = self._parse_llm_response(response_text)
            logger.info(f"Parsed booking extraction result: {result}")

            if result and self._validate_booking_data(result):
                logger.info("Booking data validated successfully")
                return BookingInfo(**result)
            else:
                logger.warning(
                    f"Booking data validation failed. Result: {result}"
                )

            return None

        except ResourceExhausted as e:
            logger.warning(
                f"Gemini API quota exceeded during booking extraction: {e}. "
                "Booking extraction skipped due to rate limit."
            )
            return None
        except Exception as e:
            logger.error(f"Error extracting booking info: {e}", exc_info=True)
            return None

    def _build_extraction_prompt(
        self, conversation_history: str, current_message: str
    ) -> str:
        """Build prompt for booking extraction."""
        return f"""You are an AI assistant that extracts interview booking information from conversations.

Conversation History:
{conversation_history}

Current Message:
{current_message}

Extract the following information if available:
- name: Full name of the person
- email: Email address
- date: Date in YYYY-MM-DD format
- time: Time in HH:MM format (24-hour)

Rules:
1. Extract ALL information that is explicitly mentioned, even if the date might be invalid (e.g., weekend)
2. For dates like "tomorrow", "next Monday", calculate based on today's date: {datetime.now().strftime("%Y-%m-%d")}
3. Convert time to 24-hour format (e.g., "2 PM" → "14:00", "10:00 AM" → "10:00")
4. Extract the date exactly as mentioned (e.g., if user says "November 15, 2025", extract "2025-11-15")
5. If any required field is missing, return null for that field
6. Extract email addresses exactly as written

IMPORTANT: Extract the information even if the date is a weekend or outside business hours. Just extract what the user provided.

Return ONLY a JSON object with this structure (no markdown, no explanation, no additional text):
{{
    "name": "string or null",
    "email": "string or null",
    "date": "YYYY-MM-DD or null",
    "time": "HH:MM or null"
}}"""

    def _extract_response_text(self, response: Any) -> str:
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

    def _parse_llm_response(self, response_text: str) -> dict[str, Any] | None:
        """Parse LLM response and extract JSON."""
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()

            # Handle markdown code blocks
            if "```" in cleaned:
                # Find JSON block
                parts = cleaned.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        cleaned = part[4:].strip()
                        break
                    elif part.startswith("{"):
                        cleaned = part
                        break
            else:
                # Try to find JSON object in the text
                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned = cleaned[start_idx : end_idx + 1]

            cleaned = cleaned.strip()

            # Parse JSON
            data = json.loads(cleaned)

            # Convert to dict and filter out None values (but keep empty strings)
            result = {}
            for k, v in data.items():
                if v is not None:
                    result[k] = v

            logger.debug(f"Successfully parsed JSON: {result}")
            return result if result else None

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response text was: {response_text}")
            # Try to extract JSON manually as a last resort
            try:
                # Find the first { and last } to extract JSON object
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx : end_idx + 1]
                    data = json.loads(json_str)
                    logger.info(f"Successfully extracted JSON manually: {data}")
                    return {k: v for k, v in data.items() if v is not None}
            except Exception as e2:
                logger.debug(f"Manual JSON extraction also failed: {e2}")
                pass
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}", exc_info=True)
            return None

    def _validate_booking_data(self, data: dict[str, Any]) -> bool:
        """
        Validate extracted booking data.

        Args:
            data: Extracted booking data

        Returns:
            bool: True if all required fields are present
        """
        required_fields = ["name", "email", "date", "time"]

        # Check all fields exist
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.warning(
                f"Missing required fields in booking data: {missing_fields}. "
                f"Available fields: {list(data.keys())}"
            )
            return False

        # Check no fields are None or empty
        empty_fields = [
            field for field in required_fields if not data.get(field)
        ]
        if empty_fields:
            logger.warning(
                f"Empty required fields in booking data: {empty_fields}"
            )
            return False

        return True

    def save_booking(
        self, db: Session, session_id: uuid.UUID, booking_info: BookingInfo
    ) -> Booking:
        """
        Save booking to database.

        Args:
            db: Database session
            session_id: Chat session UUID
            booking_info: Booking information

        Returns:
            Booking: Created booking record
        """
        try:
            # BookingInfo already has date and time as proper objects (via Pydantic validators)
            # Create booking
            booking = Booking(
                session_id=session_id,
                name=booking_info.name,
                email=booking_info.email,
                booking_date=booking_info.date,  # Already a date object
                booking_time=booking_info.time,  # Already a time object
                status="pending",
            )

            db.add(booking)
            db.commit()
            db.refresh(booking)

            logger.info(f"Booking saved: {booking.id}")
            return booking

        except Exception as e:
            logger.error(f"Error saving booking: {e}")
            db.rollback()
            raise

    def get_booking_by_id(self, db: Session, booking_id: uuid.UUID) -> Booking | None:
        """Get booking by ID."""
        return db.query(Booking).filter(Booking.id == booking_id).first()

    def get_bookings_by_session(
        self, db: Session, session_id: uuid.UUID
    ) -> list[Booking]:
        """Get all bookings for a session."""
        return db.query(Booking).filter(Booking.session_id == session_id).all()

    def get_all_bookings(
        self, db: Session, skip: int = 0, limit: int = 100
    ) -> list[Booking]:
        """Get all bookings with pagination."""
        return db.query(Booking).offset(skip).limit(limit).all()

    def update_booking_status(
        self, db: Session, booking_id: uuid.UUID, status: str
    ) -> Booking | None:
        """Update booking status."""
        try:
            booking = self.get_booking_by_id(db, booking_id)
            if booking:
                booking.status = status
                db.commit()
                db.refresh(booking)
                logger.info(f"Updated booking {booking_id} status to {status}")
                return booking
            return None
        except Exception as e:
            logger.error(f"Error updating booking status: {e}")
            db.rollback()
            raise


# Global booking service instance
booking_service = BookingService()


def get_booking_service() -> BookingService:
    """
    Dependency function to get booking service.

    Returns:
        BookingService: Booking service instance

    """
    return booking_service
