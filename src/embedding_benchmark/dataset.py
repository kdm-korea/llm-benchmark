"""Load, validate and prepare the EVE legal retrieval dataset.

This module owns legal dataset rules, IDs, source integrity and JSON storage.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil


def fingerprint(value: object) -> str:
    """Hash JSON content deterministically, preserving existing IDs and cache keys."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_qrels(corpus, query_id, grades):
    for document_id, grade in grades.items():
        if document_id not in corpus:
            raise ValueError(f"Unknown qrels document: {query_id}/{document_id}")
        valid_grade = (
            isinstance(grade, (int, float))
            and not isinstance(grade, bool)
            and math.isfinite(grade)
            and grade >= 0
        )
        if not valid_grade:
            raise ValueError(f"Invalid relevance grade: {query_id}/{document_id}")


@dataclass
class RetrievalData:
    """Corpus, queries and relevance labels used by one evaluation."""
    corpus: dict[str, dict[str, str]]
    queries: dict[str, str]
    qrels: dict[str, dict[str, float]]
    metadata: dict

    def validate(self) -> None:
        if not self.corpus or not self.queries:
            raise ValueError("Corpus and queries must not be empty")
        for docid, document in self.corpus.items():
            if not docid or not document["text"].strip():
                raise ValueError(f"Empty document: {docid!r}")
        if set(self.queries) != set(self.qrels):
            raise ValueError("Query and qrels ID sets differ")
        for qid, query in self.queries.items():
            if not query.strip():
                raise ValueError(f"Empty query: {qid}")
            _validate_qrels(self.corpus, qid, self.qrels[qid])

    def document_texts(self) -> list[str]:
        texts = []
        for document in self.corpus.values():
            parts = (document["title"].strip(), document["text"].strip())
            texts.append("\n".join(part for part in parts if part))
        return texts

    def subset_queries(self, limit: int | None) -> "RetrievalData":
        """Limit queries while retaining every searchable corpus document."""
        if limit is not None and limit < 1:
            raise ValueError("query_limit must be positive")
        ids = list(self.queries)[:limit]
        return RetrievalData(
            corpus=self.corpus,
            queries={qid: self.queries[qid] for qid in ids},
            qrels={qid: self.qrels[qid] for qid in ids},
            metadata=self.metadata,
        )


def write_json(path, value):
    """Replace a JSON file only after its complete contents have been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.write_text(content + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                raise ValueError(f"Invalid JSON: {path}:{number}") from exc


def identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty string ID, got {value!r}")
    return value  # Preserve leading zeros; do not coerce IDs through integers.


def load_eve(raw_dir, duplicate_query_policy="error"):
    """Map EVE rows to retrieval data; audit any explicitly allowed query merges."""
    if duplicate_query_policy not in ("error", "merge_max"):
        raise ValueError("Unknown duplicate_query_policy")
    raw_dir = Path(raw_dir)
    source = json.loads((raw_dir / "source.json").read_text())
    actual_hashes = {
        name: file_sha256(raw_dir / name)
        for name in ("corpus.jsonl", "qrels.jsonl")
    }
    for name, expected in source.get("file_sha256", {}).items():
        if name in actual_hashes and actual_hashes[name] != expected:
            raise ValueError(f"Source file hash mismatch: {name}")
    corpus, queries, qrels = {}, {}, {}
    duplicate_rows = []
    source_rows = 0
    for row in read_jsonl(raw_dir / "corpus.jsonl"):
        docid = identifier(row["docid"])
        if docid in corpus:
            raise ValueError(f"Duplicate corpus ID: {docid}")
        if not isinstance(row["text"], str) or not isinstance(row.get("title", ""), str):
            raise ValueError(f"Invalid document text: {docid}")
        corpus[docid] = {
            "title": row.get("title", ""),
            "text": row["text"],
            "kind": row.get("kind", ""),
        }
    for row in read_jsonl(raw_dir / "qrels.jsonl"):
        source_rows += 1
        query = row["query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Empty/invalid query")
        qid = "q-" + fingerprint(query)[:24]
        if not isinstance(row["grades"], dict):
            raise ValueError(f"Invalid grades: {qid}")
        grades = {identifier(d): grade for d, grade in row["grades"].items()}
        # Validate before merging so invalid duplicate rows cannot be masked.
        _validate_qrels(corpus, qid, grades)
        if qid in queries:
            if queries[qid] != query or duplicate_query_policy == "error":
                raise ValueError(f"Duplicate query text or ID collision: {qid}")
            duplicate_rows.append({
                "query_id": qid,
                "source_row": source_rows,
                "previous_grades": dict(qrels[qid]),
                "incoming_grades": grades,
            })
            for docid, grade in grades.items():
                qrels[qid][docid] = max(grade, qrels[qid].get(docid, 0))
        else:
            queries[qid] = query
            qrels[qid] = grades
    data = RetrievalData(
        corpus=dict(sorted(corpus.items())),
        queries=dict(sorted(queries.items())),
        qrels=qrels,
        metadata={
            **source,
            "split": "test",
            "query_id_policy": "sha256-json-text-first24",
            "source_qrels_rows": source_rows,
            "duplicate_query_policy": duplicate_query_policy,
            "actual_file_sha256": actual_hashes,
            "duplicate_rows": duplicate_rows,
        },
    )
    data.validate()
    data.metadata["data_hash"] = fingerprint({
        "corpus": data.corpus, "queries": data.queries, "qrels": data.qrels,
    })
    return data


def download_dataset(config):
    """Download a pinned source snapshot and record each file's checksum."""
    from huggingface_hub import HfApi, hf_hub_download
    revision = HfApi().dataset_info(config["id"], revision=config["revision"]).sha
    raw = Path(config["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("README.md", "corpus.jsonl", "qrels.jsonl"):
        cached = hf_hub_download(
            config["id"], name, repo_type="dataset", revision=revision,
            cache_dir="cache/huggingface",
        )
        shutil.copyfile(cached, raw / name)
        hashes[name] = file_sha256(raw / name)
    source = {"dataset_id": config["id"], "revision": revision, "file_sha256": hashes}
    (raw / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    return source


def prepare_dataset(config):
    """Validate local source data and save normalized data plus an audit report."""
    dataset = config["dataset"]
    data = load_eve(dataset["raw_dir"], dataset["duplicate_query_policy"])
    if (
        data.metadata["dataset_id"] != dataset["id"]
        or data.metadata["revision"] != dataset["revision"]
    ):
        raise ValueError("Local dataset source differs from configured pinned revision")
    target = Path(dataset["prepared_dir"])
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "corpus.json", data.corpus)
    write_json(target / "queries.json", data.queries)
    write_json(target / "qrels.json", data.qrels)
    positives = [sum(g > 0 for g in grades.values()) for grades in data.qrels.values()]
    report = {
        **data.metadata,
        "corpus_count": len(data.corpus),
        "query_count": len(data.queries),
        "no_positive_query_count": sum(count == 0 for count in positives),
        "relevant_docs_min": min(positives),
        "relevant_docs_max": max(positives),
        "relevant_docs_mean": sum(positives) / len(positives),
    }
    write_json(target / "validation.json", report)
    return data, report
