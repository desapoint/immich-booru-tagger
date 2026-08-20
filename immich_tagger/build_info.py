"""Runtime access to immutable container build identity."""

import os


def get_build_info() -> dict:
    """Return safe build metadata with useful local-development fallbacks."""
    version = os.getenv("APP_VERSION", "development").strip() or "development"
    revision = os.getenv("GIT_REVISION", "unknown").strip() or "unknown"
    return {
        "version": version,
        "revision": revision,
    }
