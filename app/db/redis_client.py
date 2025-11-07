"""Redis client for conversation memory storage."""

import json
import logging
from typing import Optional

import redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Redis client wrapper for memory management."""

    def __init__(self):
        """Initialize Redis connection."""
        self.client: redis.Redis | None = None
        self._connect()

    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # Test connection
            self.client.ping()
            logger.info("Redis connection established")
        except redis.ConnectionError:
            logger.exception("Failed to connect to Redis")
            raise

    def set(self, key: str, value: str, ttl: int | None = None) -> bool:
        """
        Set a key-value pair in Redis.

        Args:
            key: Redis key
            value: Value to store (will be JSON serialized if dict)
            ttl: Time to live in seconds

        Returns:
            bool: True if successful

        """
        try:
            if ttl:
                return self.client.setex(key, ttl, value)
            return self.client.set(key, value)
        except Exception as e:
            logger.exception("Error setting Redis key %s", key)
            return False

    def get(self, key: str) -> str | None:
        """
        Get value from Redis.

        Args:
            key: Redis key

        Returns:
            Optional[str]: Value if exists, None otherwise

        """
        try:
            return self.client.get(key)
        except Exception:
            logger.exception("Error getting Redis key %s", key)
            return None

    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key: Redis key

        Returns:
            bool: True if deleted

        """
        try:
            return bool(self.client.delete(key))
        except Exception:
            logger.exception("Error deleting Redis key %s", key)
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.

        Args:
            key: Redis key

        Returns:
            bool: True if exists

        """
        try:
            return bool(self.client.exists(key))
        except Exception:
            logger.exception("Error checking Redis key %s", key)
            return False

    def set_json(self, key: str, value: dict, ttl: int | None = None) -> bool:
        """
        Set a JSON object in Redis.

        Args:
            key: Redis key
            value: Dictionary to store
            ttl: Time to live in seconds

        Returns:
            bool: True if successful

        """
        try:
            json_str = json.dumps(value)
            return self.set(key, json_str, ttl)
        except Exception as e:
            logger.exception("Error setting JSON in Redis: %s", e)
            return False

    def get_json(self, key: str) -> dict | None:
        """
        Get a JSON object from Redis.

        Args:
            key: Redis key

        Returns:
            Optional[dict]: Parsed JSON if exists

        """
        try:
            value = self.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.exception("Error getting JSON from Redis: %s", e)
            return None

    def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")


# Global Redis client instance
redis_client = RedisClient()


def get_redis() -> RedisClient:
    """
    Dependency function to get Redis client.

    Returns:
        RedisClient: Redis client instance

    """
    return redis_client
