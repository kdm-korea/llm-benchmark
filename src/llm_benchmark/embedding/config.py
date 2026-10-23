"""YAML 입력을 검사하고 실행에 필요한 설정 필드를 명시한다."""

from pathlib import Path

import yaml

from llm_benchmark.embedding.models import (
    BenchmarkConfig, DatasetConfig, EvaluationConfig, ModelConfig, RuntimeConfig,
)
from llm_benchmark.embedding.parsing import integer, mapping, sequence, string


def load_config(path: str | Path) -> BenchmarkConfig:
    """지원하지 않는 데이터 조건과 잘못된 모델 이름·실행 크기를 미리 거부한다."""
    raw = mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    source = mapping(raw["dataset"])
    dataset = DatasetConfig(
        id=string(source["id"]), revision=string(source["revision"]),
        raw_dir=string(source["raw_dir"]), prepared_dir=string(source["prepared_dir"]),
        split=string(source["split"]), input_template=string(source["input_template"]),
        duplicate_query_policy=string(source["duplicate_query_policy"]),
    )
    if dataset["split"] != "test" or dataset["input_template"] != "title_newline_text":
        raise ValueError("Only test/title_newline_text is supported")
    evaluation_source = mapping(raw["evaluation"])
    limit = evaluation_source["query_limit"]
    evaluation = EvaluationConfig(
        k=[integer(k) for k in sequence(evaluation_source["k"])],
        query_batch_size=integer(evaluation_source["query_batch_size"]),
        query_limit=None if limit is None else integer(limit),
    )
    ks = evaluation["k"]
    if not ks or any(k < 1 for k in ks) or len(ks) != len(set(ks)):
        raise ValueError("K values must be distinct positive integers")
    runtime_source = mapping(raw["runtime"])
    runtime = RuntimeConfig(
        device=string(runtime_source["device"]), batch_size=integer(runtime_source["batch_size"]),
        cache_dir=string(runtime_source["cache_dir"]),
        model_cache_dir=string(runtime_source["model_cache_dir"]),
        results_dir=string(runtime_source["results_dir"]),
    )
    if runtime["batch_size"] < 1 or evaluation["query_batch_size"] < 1:
        raise ValueError("Batch sizes must be positive")
    models = [parse_model(value) for value in sequence(raw["models"])]
    names = [model["name"] for model in models]
    if not names or len(set(names)) != len(names):
        raise ValueError("Model names must be nonempty and unique")
    return BenchmarkConfig(dataset=dataset, evaluation=evaluation, runtime=runtime, models=models)


def parse_model(value: object) -> ModelConfig:
    source = mapping(value)
    model = ModelConfig(
        name=string(source["name"]), id=string(source["id"]), revision=string(source["revision"]),
        max_length=integer(source["max_length"]), query_prefix=string(source["query_prefix"]),
        document_prefix=string(source["document_prefix"]),
    )
    name = model["name"]
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Model name must be a safe filename")
    if model["max_length"] < 1:
        raise ValueError("max_length must be positive")
    return model
