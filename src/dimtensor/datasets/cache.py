"""Centralized cache management for dataset loaders.

Provides cache directory management, metadata tracking, TTL enforcement,
and utilities for cache inspection and cleanup.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class CacheEntry:
    """Metadata for a cached file.

    Attributes:
        cache_key: Unique identifier for the cached file.
        url: Source URL of the downloaded data.
        filepath: Path to the cached data file.
        size: File size in bytes.
        timestamp: Unix timestamp when cached.
        ttl: Time-to-live in seconds (None = never expires).
        content_type: MIME type of the cached content.
        source: Loader name or source identifier.
    """
    cache_key: str
    url: str
    filepath: str
    size: int
    timestamp: float
    ttl: float | None = None
    content_type: str = ""
    source: str = ""

    def is_expired(self) -> bool:
        """Check if this cache entry has expired.

        Returns:
            True if the entry has passed its TTL, False otherwise.
        """
        if self.ttl is None:
            return False

        age = time.time() - self.timestamp
        return age > self.ttl

    def age_seconds(self) -> float:
        """Get the age of this cache entry in seconds.

        Returns:
            Number of seconds since the entry was created.
        """
        return time.time() - self.timestamp

    def age_human(self) -> str:
        """Get human-readable age string.

        Returns:
            String like "2 hours ago" or "3 days ago".
        """
        age = self.age_seconds()

        if age < 60:
            return f"{int(age)}s ago"
        elif age < 3600:
            return f"{int(age / 60)}m ago"
        elif age < 86400:
            return f"{int(age / 3600)}h ago"
        else:
            return f"{int(age / 86400)}d ago"

    def size_human(self) -> str:
        """Get human-readable size string.

        Returns:
            String like "1.5 MB" or "342 KB".
        """
        if self.size < 1024:
            return f"{self.size} B"
        elif self.size < 1024 * 1024:
            return f"{self.size / 1024:.1f} KB"
        elif self.size < 1024 * 1024 * 1024:
            return f"{self.size / (1024 * 1024):.1f} MB"
        else:
            return f"{self.size / (1024 * 1024 * 1024):.1f} GB"


class CacheManager:
    """Manages cache directory and metadata for dataset loaders.

    Provides centralized cache operations including:
    - Creating and tracking cache entries
    - Expiring old entries based on TTL
    - Enforcing disk space limits
    - Generating cache statistics
    - Cleaning up by source or age

    Attributes:
        cache_dir: Path to cache directory.
        max_size: Maximum cache size in bytes (None = unlimited).
        default_ttl: Default TTL for entries in seconds (None = never expires).
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_size: int | None = None,
        default_ttl: float | None = None,
    ):
        """Initialize cache manager.

        Args:
            cache_dir: Cache directory path (default: ~/.dimtensor/cache/).
            max_size: Maximum cache size in bytes (default: None/unlimited).
            default_ttl: Default time-to-live in seconds (default: None/never expires).
        """
        if cache_dir is None:
            cache_dir = self._get_default_cache_dir()

        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._metadata_file = self.cache_dir / "metadata.json"

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory.

        Returns:
            Path to default cache directory (~/.dimtensor/cache/).
            Can be overridden with DIMTENSOR_CACHE_DIR environment variable.
        """
        cache_dir = os.environ.get("DIMTENSOR_CACHE_DIR")
        if cache_dir:
            return Path(cache_dir)

        # Default to ~/.dimtensor/cache/
        home = Path.home()
        return home / ".dimtensor" / "cache"

    def _load_metadata(self) -> dict[str, CacheEntry]:
        """Load cache metadata from disk.

        Returns:
            Dictionary mapping cache keys to CacheEntry objects.
        """
        if not self._metadata_file.exists():
            return {}

        try:
            data = json.loads(self._metadata_file.read_text())
            return {
                key: CacheEntry(**entry_data)
                for key, entry_data in data.items()
            }
        except (json.JSONDecodeError, TypeError, KeyError):
            # If metadata is corrupted, return empty dict
            return {}

    def _save_metadata(self, metadata: dict[str, CacheEntry]) -> None:
        """Save cache metadata to disk.

        Args:
            metadata: Dictionary of cache entries to save.
        """
        data = {
            key: asdict(entry)
            for key, entry in metadata.items()
        }
        self._metadata_file.write_text(json.dumps(data, indent=2))

    def add_entry(
        self,
        cache_key: str,
        url: str,
        filepath: Path,
        ttl: float | None = None,
        content_type: str = "",
        source: str = "",
    ) -> CacheEntry:
        """Add a new cache entry.

        Args:
            cache_key: Unique identifier for this cache entry.
            url: Source URL of the data.
            filepath: Path to the cached file.
            ttl: Time-to-live in seconds (None = use default or never expires).
            content_type: MIME type of the content.
            source: Loader name or source identifier.

        Returns:
            The created CacheEntry.
        """
        # Use default TTL if not specified
        if ttl is None:
            ttl = self.default_ttl

        # Get file size
        size = filepath.stat().st_size if filepath.exists() else 0

        # Create entry
        entry = CacheEntry(
            cache_key=cache_key,
            url=url,
            filepath=str(filepath),
            size=size,
            timestamp=time.time(),
            ttl=ttl,
            content_type=content_type,
            source=source,
        )

        # Load existing metadata, add new entry, save
        metadata = self._load_metadata()
        metadata[cache_key] = entry
        self._save_metadata(metadata)

        # Enforce size limit if set
        if self.max_size is not None:
            self._enforce_size_limit()

        return entry

    def get_entry(self, cache_key: str) -> CacheEntry | None:
        """Get a cache entry by key.

        Args:
            cache_key: Cache key to look up.

        Returns:
            CacheEntry if found, None otherwise.
        """
        metadata = self._load_metadata()
        return metadata.get(cache_key)

    def remove_entry(self, cache_key: str) -> bool:
        """Remove a cache entry and its associated file.

        Args:
            cache_key: Cache key to remove.

        Returns:
            True if entry was removed, False if not found.
        """
        metadata = self._load_metadata()
        entry = metadata.get(cache_key)

        if entry is None:
            return False

        # Delete the cached file
        filepath = Path(entry.filepath)
        if filepath.exists():
            filepath.unlink()

        # Remove metadata entry
        del metadata[cache_key]
        self._save_metadata(metadata)

        return True

    def list_entries(self, source: str | None = None) -> list[CacheEntry]:
        """List all cache entries.

        Args:
            source: Filter by source name (None = all sources).

        Returns:
            List of cache entries.
        """
        metadata = self._load_metadata()
        entries = list(metadata.values())

        if source is not None:
            entries = [e for e in entries if e.source == source]

        return entries

    def clean_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed.
        """
        metadata = self._load_metadata()
        expired_keys = [
            key for key, entry in metadata.items()
            if entry.is_expired()
        ]

        for key in expired_keys:
            self.remove_entry(key)

        return len(expired_keys)

    def clean_by_age(self, max_age_seconds: float) -> int:
        """Remove cache entries older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds.

        Returns:
            Number of entries removed.
        """
        metadata = self._load_metadata()
        old_keys = [
            key for key, entry in metadata.items()
            if entry.age_seconds() > max_age_seconds
        ]

        for key in old_keys:
            self.remove_entry(key)

        return len(old_keys)

    def clean_by_source(self, source: str) -> int:
        """Remove all cache entries from a specific source.

        Args:
            source: Source name to remove.

        Returns:
            Number of entries removed.
        """
        metadata = self._load_metadata()
        source_keys = [
            key for key, entry in metadata.items()
            if entry.source == source
        ]

        for key in source_keys:
            self.remove_entry(key)

        return len(source_keys)

    def clear_all(self) -> int:
        """Remove all cache entries.

        Returns:
            Number of entries removed.
        """
        metadata = self._load_metadata()
        count = len(metadata)

        # Delete all cached files
        for entry in metadata.values():
            filepath = Path(entry.filepath)
            if filepath.exists():
                filepath.unlink()

        # Clear metadata
        self._save_metadata({})

        return count

    def _enforce_size_limit(self) -> None:
        """Enforce maximum cache size by removing oldest entries."""
        if self.max_size is None:
            return

        metadata = self._load_metadata()
        total_size = sum(entry.size for entry in metadata.values())

        if total_size <= self.max_size:
            return

        # Sort entries by age (oldest first)
        entries_by_age = sorted(
            metadata.items(),
            key=lambda x: x[1].timestamp,
        )

        # Remove oldest entries until under limit
        for key, entry in entries_by_age:
            if total_size <= self.max_size:
                break

            self.remove_entry(key)
            total_size -= entry.size

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics including:
            - total_entries: Number of cache entries
            - total_size: Total size in bytes
            - total_size_human: Human-readable size
            - expired_entries: Number of expired entries
            - sources: List of unique sources
            - oldest_entry: Age of oldest entry in seconds
            - newest_entry: Age of newest entry in seconds
        """
        metadata = self._load_metadata()
        entries = list(metadata.values())

        if not entries:
            return {
                "total_entries": 0,
                "total_size": 0,
                "total_size_human": "0 B",
                "expired_entries": 0,
                "sources": [],
                "oldest_entry": None,
                "newest_entry": None,
            }

        total_size = sum(e.size for e in entries)
        expired_count = sum(1 for e in entries if e.is_expired())
        sources = sorted(set(e.source for e in entries if e.source))

        ages = [e.age_seconds() for e in entries]
        oldest = max(ages)
        newest = min(ages)

        # Human-readable total size
        if total_size < 1024:
            size_human = f"{total_size} B"
        elif total_size < 1024 * 1024:
            size_human = f"{total_size / 1024:.1f} KB"
        elif total_size < 1024 * 1024 * 1024:
            size_human = f"{total_size / (1024 * 1024):.1f} MB"
        else:
            size_human = f"{total_size / (1024 * 1024 * 1024):.1f} GB"

        return {
            "total_entries": len(entries),
            "total_size": total_size,
            "total_size_human": size_human,
            "expired_entries": expired_count,
            "sources": sources,
            "oldest_entry": oldest,
            "newest_entry": newest,
        }

    def verify_integrity(self) -> list[str]:
        """Verify cache integrity and return list of issues.

        Returns:
            List of issue descriptions (empty if no issues).
        """
        metadata = self._load_metadata()
        issues = []

        for key, entry in metadata.items():
            filepath = Path(entry.filepath)

            # Check if file exists
            if not filepath.exists():
                issues.append(f"Missing file for cache key '{key}': {entry.filepath}")
                continue

            # Check if size matches
            actual_size = filepath.stat().st_size
            if actual_size != entry.size:
                issues.append(
                    f"Size mismatch for '{key}': "
                    f"expected {entry.size}, got {actual_size}"
                )

        return issues


# Global cache manager instance
_global_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance.

    Returns:
        The global CacheManager instance.
    """
    global _global_cache_manager

    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()

    return _global_cache_manager


def set_cache_manager(manager: CacheManager) -> None:
    """Set the global cache manager instance.

    Args:
        manager: CacheManager to use as global instance.
    """
    global _global_cache_manager
    _global_cache_manager = manager
