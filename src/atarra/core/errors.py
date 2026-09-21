"""Exception hierarchy.

A dedicated root exception lets the API layer distinguish "we could not reach or
interpret the imagery archive" (upstream, often retryable) from genuine
programming errors, and map the former to a clean HTTP status instead of a 500.
"""

from __future__ import annotations


class AtarraError(Exception):
    """Root of every error ATARRA raises deliberately."""


class ConfigError(AtarraError):
    """A config file is missing, malformed, or internally inconsistent."""


class ImageryError(AtarraError):
    """Imagery could not be discovered, fetched, or interpreted."""


class CacheLimitError(AtarraError):
    """The on-disk cache could not be kept within its configured ceiling."""
