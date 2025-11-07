"""Conversation memory service using Redis."""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.config import settings
from app.db.redis_client import RedisClient

logger = logging.getLogger(__name__)


class MemoryService:
    """Service for managing conversation memory in Redis."""

    def __init__(self, redis_client: RedisClient):
        """
        Initialize memory service.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.memory_limit = settings.conversation_memory_limit
        self.session_ttl = settings.session_ttl

    def create_session(self) -> tuple[uuid.UUID, datetime]:
        """
        Create a new conversation session.

        Returns:
            tuple: (session_id, expires_at)
        """
        session_id = uuid.uuid4()
        key = self._get_session_key(session_id)

        session_data = {
            "session_id": str(session_id),
            "messages": [],
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "last_accessed": datetime.now(tz=timezone.utc).isoformat(),
        }

        success = self.redis.set_json(key, session_data, ttl=self.session_ttl)

        if success:
            expires_at = datetime.now(tz=timezone.utc) + timedelta(
                seconds=self.session_ttl
            )
            logger.info(f"Created session: {session_id}")
            return session_id, expires_at
        else:
            raise Exception("Failed to create session in Redis")

    def add_message(self, session_id: uuid.UUID, role: str, content: str) -> bool:
        """
        Add a message to conversation history.

        Args:
            session_id: Session UUID
            role: Message role ("user" or "assistant")
            content: Message content

        Returns:
            bool: True if successful
        """
        key = self._get_session_key(session_id)
        session_data = self.redis.get_json(key)

        if not session_data:
            logger.warning(f"Session not found: {session_id}")
            return False

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }

        session_data["messages"].append(message)
        session_data["last_accessed"] = datetime.now(tz=timezone.utc).isoformat()

        # Keep only last N messages (sliding window)
        if len(session_data["messages"]) > self.memory_limit:
            session_data["messages"] = session_data["messages"][-self.memory_limit :]

        # Update session with refreshed TTL
        success = self.redis.set_json(key, session_data, ttl=self.session_ttl)

        if success:
            logger.debug(f"Added {role} message to session {session_id}")
            return True
        else:
            logger.error(f"Failed to update session {session_id}")
            return False

    def get_history(
        self, session_id: uuid.UUID, limit: Optional[int] = None
    ) -> list[dict[str, str]]:
        """
        Get conversation history for a session.

        Args:
            session_id: Session UUID
            limit: Maximum number of messages to return (None = all)

        Returns:
            List[Dict]: List of messages
        """
        key = self._get_session_key(session_id)
        session_data = self.redis.get_json(key)

        if not session_data:
            logger.warning(f"Session not found: {session_id}")
            return []

        messages = session_data.get("messages", [])

        if limit and limit > 0:
            messages = messages[-limit:]

        # Update last accessed time
        session_data["last_accessed"] = datetime.now(tz=timezone.utc).isoformat()
        self.redis.set_json(key, session_data, ttl=self.session_ttl)

        return messages

    def get_context_window(self, session_id: uuid.UUID, num_turns: int = 5) -> str:
        """
        Get formatted conversation context for LLM.

        Args:
            session_id: Session UUID
            num_turns: Number of conversation turns to include

        Returns:
            str: Formatted conversation history

        """
        messages = self.get_history(session_id, limit=num_turns * 2)

        if not messages:
            return ""

        context_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")

        return "\n".join(context_parts)

    def session_exists(self, session_id: uuid.UUID) -> bool:
        """
        Check if session exists.

        Args:
            session_id: Session UUID

        Returns:
            bool: True if session exists

        """
        key = self._get_session_key(session_id)
        return self.redis.exists(key)

    def delete_session(self, session_id: uuid.UUID) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session UUID

        Returns:
            bool: True if deleted

        """
        key = self._get_session_key(session_id)
        success = self.redis.delete(key)

        if success:
            logger.info(f"Deleted session: {session_id}")

        return success

    def get_session_info(self, session_id: uuid.UUID) -> dict | None:
        """
        Get session metadata.

        Args:
            session_id: Session UUID

        Returns:
            Optional[Dict]: Session info or None
        """
        key = self._get_session_key(session_id)
        session_data = self.redis.get_json(key)

        if not session_data:
            return None

        return {
            "session_id": session_data.get("session_id"),
            "created_at": session_data.get("created_at"),
            "last_accessed": session_data.get("last_accessed"),
            "message_count": len(session_data.get("messages", [])),
        }

    def _get_session_key(self, session_id: uuid.UUID) -> str:
        """
        Get Redis key for session.

        Args:
            session_id: Session UUID

        Returns:
            str: Redis key
        """
        return f"session:{str(session_id)}"


def get_memory_service(redis_client: RedisClient) -> MemoryService:
    """
    Dependency function to get memory service.

    Args:
        redis_client: Redis client instance

    Returns:
        MemoryService: Memory service instance

    """
    return MemoryService(redis_client)
