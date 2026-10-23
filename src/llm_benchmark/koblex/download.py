"""manifest에 고정된 원본만 다운로드하고 크기·해시를 검증한다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from urllib.request import urlopen

from llm_benchmark.common.hashing import file_sha256
from llm_benchmark.common.io import write_json

from .config import DEFAULT_CONFIG, Settings, load_settings
from .models import PathInput


def download(settings: Settings, raw: PathInput | None = None) -> None:
    raw = Path(raw) if raw is not None else settings.raw
    for source in settings.manifest['sources']:
        for entry in source['files']:
            path = settings.source_path(raw, entry)
            if path.exists() and path.stat().st_size == entry['size_bytes'] and file_sha256(path) == entry['sha256']:
                continue
            with urlopen(entry['url'], timeout=settings.timeout) as response:
                data = response.read()
            if len(data) != entry['size_bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError(f'Source checksum mismatch: {path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + '.part')
            temporary.write_bytes(data)
            temporary.replace(path)
            print(path)
    receipt = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'manifest': settings.manifest}
    write_json(raw / 'download-receipt.json', receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--raw', type=Path)
    args = parser.parse_args()
    download(load_settings(args.config), args.raw)


if __name__ == '__main__':
    main()
