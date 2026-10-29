"""실행 설정과 원본 manifest를 읽는다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import cast

from .models import (
    ExperimentConfig, ManifestSource, PathInput, SourceFile, SourceManifest, SourceSelection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / 'configs/koblex.toml'


@dataclass(frozen=True)
class Source:
    revision: str
    file: Path
    duplicate_file: Path


@dataclass(frozen=True)
class SplitSettings:
    seed: int
    dev_ratio: float
    similarity_threshold: float
    shingle_size: int


@dataclass(frozen=True)
class Settings:
    version: str
    manifest_path: Path
    manifest: SourceManifest
    raw_prefix: Path
    raw: Path
    output: Path
    audit: Path
    qa: Source
    corpus: Source
    source_split: str
    split: SplitSettings
    timeout: float

    def source_path(self, raw: PathInput, entry: SourceFile) -> Path:
        """manifest의 원본 위치를 실제 실행의 raw 경로로 옮긴다."""
        return Path(raw) / Path(entry['path']).relative_to(self.raw_prefix)


def load_settings(path: PathInput = DEFAULT_CONFIG) -> Settings:
    """상대 경로는 프로젝트 루트 기준이며 리비전은 manifest만 사용한다."""
    config = cast(ExperimentConfig, tomllib.loads(Path(path).read_text()))
    manifest_path = PROJECT_ROOT / config['source_manifest']
    manifest = cast(SourceManifest, json.loads(manifest_path.read_text()))
    raw_prefix = Path(manifest['raw_root'])
    sources_by_repo = {source['repo']: source for source in manifest['sources']}

    split = SplitSettings(**config['split'])
    if type(split.seed) is not int or type(split.shingle_size) is not int or split.shingle_size <= 0:
        raise ValueError('seed는 정수, shingle_size는 양의 정수여야 합니다')
    if not 0 < split.dev_ratio < 1 or not 0 <= split.similarity_threshold <= 1:
        raise ValueError('dev_ratio는 0~1 사이, similarity_threshold는 0~1 범위여야 합니다')
    timeout = config['download']['timeout_seconds']
    if timeout <= 0:
        raise ValueError('timeout_seconds는 양수여야 합니다')
    return Settings(
        version=config['normalization_version'],
        manifest_path=manifest_path,
        manifest=manifest,
        raw_prefix=raw_prefix,
        raw=PROJECT_ROOT / config['paths']['raw'],
        output=PROJECT_ROOT / config['paths']['output'],
        audit=PROJECT_ROOT / config['paths']['audit'],
        qa=_load_source(config['qa'], sources_by_repo, raw_prefix),
        corpus=_load_source(config['corpus'], sources_by_repo, raw_prefix),
        source_split=config['qa']['split'],
        split=split,
        timeout=timeout,
    )


def _load_source(
    spec: SourceSelection, sources_by_repo: Mapping[str, ManifestSource], raw_prefix: Path,
) -> Source:
    record = sources_by_repo[spec['repo']]
    directory = Path(spec['repo'].rsplit('/', 1)[1])
    selected_file = directory / spec['file']
    duplicate_file = directory / spec['duplicate_file']
    recorded_files = {
        Path(entry['path']).relative_to(raw_prefix) for entry in record['files']
    }
    if selected_file not in recorded_files or duplicate_file not in recorded_files:
        raise ValueError(f"{spec['repo']}: 설정한 파일이 source manifest에 없습니다")
    return Source(
        revision=record['revision'],
        file=selected_file,
        duplicate_file=duplicate_file,
    )
