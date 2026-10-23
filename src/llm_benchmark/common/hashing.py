"""Stable hash profiles; preserve each experiment's established IDs."""

import hashlib
import json
from pathlib import Path


def fingerprint(value: object) -> str:
    """Hash JSON content deterministically, preserving existing IDs and cache keys."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
