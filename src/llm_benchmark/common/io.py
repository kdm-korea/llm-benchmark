"""특정 실험이나 프로세스 전역 상태에 의존하지 않는 JSON 파일 입출력."""

from contextlib import contextmanager
import json
from pathlib import Path
import tempfile


@contextmanager
def _atomic_text_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w',
            encoding='utf-8',
            dir=path.parent,
            prefix=path.name + '.',
            suffix='.tmp',
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yield stream
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    """기존 UTF-8 JSON 형식을 유지하고, 저장이 끝난 임시 파일로 대상 파일을 교체한다."""
    with _atomic_text_file(path) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    """모든 행의 직렬화가 성공한 경우에만 JSONL 파일을 교체해 불완전한 저장을 막는다."""
    with _atomic_text_file(path) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f'Invalid JSON: {path}:{number}') from exc
