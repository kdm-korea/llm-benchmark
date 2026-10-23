"""EVE 법률 검색 데이터를 읽고 정합성을 검증해 실험 입력으로 준비한다.

이 모듈은 EVE 데이터 규칙과 ID를 관리하며, 공통 파일 처리는 common에 둔다.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil

from llm_benchmark.common.hashing import file_sha256, fingerprint
from llm_benchmark.common.io import read_jsonl, write_json


def _validate_qrels(corpus, query_id, grades):
    for document_id, grade in grades.items():
        if document_id not in corpus:
            raise ValueError(f"Unknown qrels document: {query_id}/{document_id}")
        # bool도 int로 판정되므로 별도로 제외하고 유한한 비음수 점수만 허용한다.
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
    """한 번의 검색 평가에서 사용하는 문서·질의·관련성 정답과 출처 정보."""
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
        """질의 수만 제한하고 검색 대상 문서는 모두 유지한다.

        문서까지 줄이면 전체 코퍼스 검색과 평가 조건이 달라지므로 질의만 잘라낸다."""
        if limit is not None and limit < 1:
            raise ValueError("query_limit must be positive")
        ids = list(self.queries)[:limit]
        return RetrievalData(
            corpus=self.corpus,
            queries={qid: self.queries[qid] for qid in ids},
            qrels={qid: self.qrels[qid] for qid in ids},
            metadata=self.metadata,
        )


def identifier(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty string ID, got {value!r}")
    return value  # 앞자리 0도 ID의 일부이므로 정수 변환 없이 원본 문자열을 유지한다.


def load_eve(raw_dir, duplicate_query_policy="error"):
    """EVE 원본 행을 검색 평가 입력으로 변환한다.

    중복 질의 병합은 설정에서 허용한 경우에만 수행하고, 병합 전 정답을 기록한다."""
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
        # 잘못된 중복 행이 병합 과정에서 가려지지 않도록 각 행을 먼저 검증한다.
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
    """지정한 원본 리비전을 내려받고 파일별 해시를 기록해 데이터 출처를 고정한다."""
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
    """로컬 원본의 출처·해시·정답을 검증하고 정규화 데이터와 검증 보고서를 저장한다."""
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
