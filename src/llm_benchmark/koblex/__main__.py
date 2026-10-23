"""고정 원본으로 KoBLEX 평가 데이터를 구축한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, load_settings
from .prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--raw', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit-dir', type=Path)
    args = parser.parse_args()
    prepare(args.raw, args.output, args.audit_dir, settings=load_settings(args.config))


if __name__ == '__main__':
    main()
