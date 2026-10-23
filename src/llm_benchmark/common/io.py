"""JSON storage without experiment dependencies or process-global state."""

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
    """Atomically replace JSON, preserving the existing UTF-8 pretty format."""
    with _atomic_text_file(path) as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    """Commit only a completely serialized JSONL stream."""
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
