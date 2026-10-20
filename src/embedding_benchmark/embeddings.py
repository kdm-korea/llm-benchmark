"""Encode documents and queries with pinned models, reusing validated caches."""

import gc
import importlib.metadata
import json
from pathlib import Path

from huggingface_hub import HfApi
import numpy as np

from embedding_benchmark.dataset import file_sha256, fingerprint, write_json


def resolve_model(config):
    """Resolve the configured model revision to its exact Hub commit."""
    resolved = dict(config)
    resolved["revision"] = HfApi().model_info(config["id"], revision=config["revision"]).sha
    return resolved


class EmbeddingModel:
    """Dense embeddings using the pinned model's tokenizer and pooling rules."""
    def __init__(self, config, runtime):
        import torch
        from sentence_transformers import SentenceTransformer
        self.config = config
        self.runtime = runtime
        device = runtime["device"]
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.model = SentenceTransformer(
            config["id"], revision=config["revision"], device=device,
            cache_folder=runtime["model_cache_dir"], trust_remote_code=False,
        )
        self.model.max_seq_length = config["max_length"]
        self.device = device
        # Record the actual pooling architecture loaded from the pinned model.
        self.pooling = [
            {"module": type(module).__name__, "config": module.get_config_dict()}
            for module in self.model
            if type(module).__name__ == "Pooling"
        ]

    def encode(self, texts, kind):
        """Encode inputs and measure truncation before applying the length limit."""
        prefix = self.config["query_prefix" if kind == "query" else "document_prefix"]
        inputs = [prefix + text for text in texts]
        lengths = []
        for start in range(0, len(inputs), 128):
            tokenized = self.model.tokenizer(
                inputs[start:start + 128], truncation=False, padding=False, verbose=False,
            )
            lengths.extend(len(ids) for ids in tokenized["input_ids"])
        truncated_count = sum(length > self.config["max_length"] for length in lengths)
        stats = {
            "count": len(inputs),
            "max_length": self.config["max_length"],
            "truncated_count": truncated_count,
            "truncated_ratio": truncated_count / len(inputs),
            "max_input_tokens": max(lengths),
            "mean_input_tokens": sum(lengths) / len(lengths),
            "device": self.device,
            "dtype": str(next(self.model.parameters()).dtype),
            "pooling": self.pooling,
        }
        vectors = self.model.encode(
            inputs, batch_size=self.runtime["batch_size"], convert_to_numpy=True,
            normalize_embeddings=True, prompt="", show_progress_bar=True,
        )
        return np.asarray(vectors, dtype=np.float32), stats

    def close(self):
        import torch
        del self.model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def cached_encode(encoder, ids, texts, kind, data_metadata):
    """Reuse matching embeddings only after validating the cached file's checksum."""
    # Content and ID order are part of the cache contract, not just filenames.
    versions = {
        name: importlib.metadata.version(name)
        for name in ("sentence-transformers", "transformers", "torch", "numpy")
    }
    identity = {
        "schema_version": 1,
        "model": encoder.config,
        "kind": kind,
        "input_hash": fingerprint({"ids": ids, "texts": texts}),
        "dataset_revision": data_metadata["revision"],
        "versions": versions,
        "normalize": True,
        "prompt": "",
        "runtime": encoder.runtime,
        "device": encoder.device,
        "dtype": str(next(encoder.model.parameters()).dtype),
    }
    key = fingerprint(identity)
    folder = Path(encoder.runtime["cache_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    vector_path = folder / f"{key}.npy"
    meta_path = folder / f"{key}.json"
    if vector_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("vector_sha256") != file_sha256(vector_path):
            raise ValueError(f"Embedding cache checksum mismatch: {key}")
        vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
        if metadata["identity"] != identity or vectors.ndim != 2 or len(vectors) != len(ids):
            raise ValueError(f"Invalid embedding cache: {key}")
        return vectors, {**metadata["stats"], "cache_hit": True, "cache_key": key}
    vectors, stats = encoder.encode(texts, kind)
    if vectors.ndim != 2 or len(vectors) != len(ids) or not np.isfinite(vectors).all():
        raise ValueError("Invalid encoded vector shape or values")
    temporary = vector_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, vectors, allow_pickle=False)
    temporary.replace(vector_path)
    checksum = file_sha256(vector_path)
    write_json(meta_path, {"identity": identity, "stats": stats, "vector_sha256": checksum})
    return vectors, {**stats, "cache_hit": False, "cache_key": key}
