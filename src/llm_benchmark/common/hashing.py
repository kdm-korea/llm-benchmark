"""기존 질의 ID와 임베딩 캐시 키를 유지하기 위한 해시 계산 규칙."""

import hashlib
import json
from pathlib import Path


def fingerprint(value: object) -> str:
    """JSON 키 순서를 고정해 같은 내용에서 항상 같은 해시를 얻는다.

    직렬화 형식이 달라지면 기존 질의 ID와 캐시 키도 바뀌므로 현재 형식을 유지한다."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compact_digest(value: object) -> str:
    """KoBLEX v1 profile: insertion order and compact JSON separators."""
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
