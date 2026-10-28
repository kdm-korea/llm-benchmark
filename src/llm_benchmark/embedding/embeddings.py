"""고정 리비전 모델로 문서·질의를 임베딩하고, 검증된 벡터 캐시를 재사용한다."""

import gc
import importlib.metadata
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from huggingface_hub import HfApi

from llm_benchmark.common.hashing import file_sha256, fingerprint
from llm_benchmark.common.io import write_json

from .models import (
    DataMetadata,
    EncodedBatch,
    InputKind,
    ModelConfig,
    RuntimeConfig,
)
from .parsing import mapping, string


def resolve_model(config: ModelConfig) -> ModelConfig:
    """모델 설정의 리비전을 Hugging Face의 정확한 커밋 해시로 확인한다."""
    resolved = config.copy()
    resolved['revision'] = string(HfApi().model_info(config['id'], revision=config['revision']).sha)
    return resolved


class EmbeddingModel:
    """고정 모델의 토크나이저·풀링 규칙을 그대로 사용하는 밀집 벡터 인코더."""

    def __init__(self, config: ModelConfig, runtime: RuntimeConfig) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.runtime = runtime
        device = runtime['device']
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.model = SentenceTransformer(
            config['id'],
            revision=config['revision'],
            device=device,
            cache_folder=runtime['model_cache_dir'],
            trust_remote_code=False,
        )
        self.model.max_seq_length = config['max_length']
        self.device = device
        # 설정 이름만 믿지 않고 고정 모델에서 실제로 로드된 풀링 구조를 기록한다.
        self.pooling = [
            {'module': type(module).__name__, 'config': module.get_config_dict()}
            for module in self.model
            if type(module).__name__ == 'Pooling'
        ]

    def encode(self, texts: Sequence[str], kind: InputKind) -> EncodedBatch:
        """접두사를 적용한 입력을 임베딩한다.

        길이 제한 전에 토큰 수를 측정해 실제로 잘릴 문서·질의의 수와 비율을 기록한다."""
        prefix = self.config['query_prefix' if kind == InputKind.QUERY else 'document_prefix']
        inputs = [prefix + text for text in texts]
        lengths: list[int] = []
        for start in range(0, len(inputs), 128):
            tokenized = self.model.tokenizer(
                inputs[start : start + 128],
                truncation=False,
                padding=False,
                verbose=False,
            )
            lengths.extend(len(ids) for ids in tokenized['input_ids'])
        truncated_count = sum(length > self.config['max_length'] for length in lengths)
        stats: dict[str, object] = {
            'count': len(inputs),
            'max_length': self.config['max_length'],
            'truncated_count': truncated_count,
            'truncated_ratio': truncated_count / len(inputs),
            'max_input_tokens': max(lengths),
            'mean_input_tokens': sum(lengths) / len(lengths),
            'device': self.device,
            'dtype': str(next(self.model.parameters()).dtype),
            'pooling': self.pooling,
        }
        vectors = self.model.encode(
            inputs,
            batch_size=self.runtime['batch_size'],
            convert_to_numpy=True,
            normalize_embeddings=True,
            prompt='',
            show_progress_bar=True,
        )
        return EncodedBatch(vectors=np.asarray(vectors, dtype=np.float32), stats=stats)

    def close(self) -> None:
        import torch

        del self.model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Parameter(Protocol):
    @property
    def dtype(self) -> object: ...


class ParameterModel(Protocol):
    def parameters(self) -> Iterator[Parameter]: ...


class Encoder(Protocol):
    """캐시가 모델 구현에서 읽는 설정과 추론 연산의 최소 계약."""

    @property
    def config(self) -> ModelConfig: ...
    @property
    def runtime(self) -> RuntimeConfig: ...
    @property
    def device(self) -> str: ...
    @property
    def model(self) -> ParameterModel: ...
    def encode(self, texts: Sequence[str], kind: InputKind) -> EncodedBatch: ...


def cached_encode(
    encoder: Encoder,
    ids: list[str],
    texts: list[str],
    kind: InputKind,
    data_metadata: DataMetadata,
) -> EncodedBatch:
    """입력·모델·실행 조건이 같은 벡터만 재사용하고, 읽기 전에 캐시 파일 해시를 검증한다."""
    # 입력 내용과 ID 순서도 캐시 키에 넣어 다른 입력의 벡터가 재사용되지 않게 한다.
    versions = {
        name: importlib.metadata.version(name)
        for name in ('sentence-transformers', 'transformers', 'torch', 'numpy')
    }
    identity = {
        'schema_version': 1,
        'model': encoder.config,
        'kind': kind,
        'input_hash': fingerprint({'ids': ids, 'texts': texts}),
        'dataset_revision': data_metadata.revision,
        'versions': versions,
        'normalize': True,
        'prompt': '',
        'runtime': encoder.runtime,
        'device': encoder.device,
        'dtype': str(next(encoder.model.parameters()).dtype),
    }
    key = fingerprint(identity)
    folder = Path(encoder.runtime['cache_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    vector_path = folder / f'{key}.npy'
    meta_path = folder / f'{key}.json'
    if vector_path.exists() and meta_path.exists():
        metadata = mapping(json.loads(meta_path.read_text()))
        if metadata.get('vector_sha256') != file_sha256(vector_path):
            raise ValueError(f'Embedding cache checksum mismatch: {key}')
        vectors = np.load(vector_path, mmap_mode='r', allow_pickle=False)
        if metadata['identity'] != identity or vectors.ndim != 2 or len(vectors) != len(ids):
            raise ValueError(f'Invalid embedding cache: {key}')
        return EncodedBatch(
            vectors=vectors,
            stats={**mapping(metadata['stats']), 'cache_hit': True, 'cache_key': key},
        )
    encoded = encoder.encode(texts, kind)
    vectors, stats = encoded.vectors, encoded.stats
    if vectors.ndim != 2 or len(vectors) != len(ids) or not np.isfinite(vectors).all():
        raise ValueError('Invalid encoded vector shape or values')
    temporary = vector_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, vectors, allow_pickle=False)
    temporary.replace(vector_path)
    checksum = file_sha256(vector_path)
    write_json(meta_path, {'identity': identity, 'stats': stats, 'vector_sha256': checksum})
    return EncodedBatch(vectors=vectors, stats={**stats, 'cache_hit': False, 'cache_key': key})
