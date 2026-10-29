import pytest

from llm_benchmark.koblex.mapping import build_units as _build_units, match_context
from llm_benchmark.koblex.splitting import split_queries as _split_queries
from llm_benchmark.koblex.config import SplitSettings
from llm_benchmark.koblex.evaluation import chunk_coverage_rankings, evaluate_evidence
from llm_benchmark.koblex.models import (
    EvidenceMapping, MappingStatus, ParentStatus, Query, QueryStatus, SplitName, Unit, UnitKind,
)


def build_units(rows):
    return _build_units(rows, revision='test-revision')


def split_queries(queries):
    return _split_queries(queries, SplitSettings(20261023, 0.2, 0.65, 5), source_split='test')


def statute(hierarchy, content, index='법 1조 목적'):
    return {'index': index, 'hierarchy': hierarchy, 'content': content}


def test_duplicates_preserve_source_positions_and_conflicting_parent_is_ambiguous():
    root = statute('법 1조 목적', '제1조(목적)')
    units, _ = build_units([root, root, statute('법 1조 목적', '제1조(목적) 다른 본문'),
                            statute('법 1조 목적 1항', '① 본문')])
    assert len(units) == 3
    assert units[0].source_row_ids == [0, 1]
    assert units[-1].parent_id is None
    assert units[-1].parent_status is ParentStatus.AMBIGUOUS


def test_exact_matching_requires_body_and_hierarchy_and_never_guesses_punctuation():
    context = statute('법 1조 목적 1항', '① 본문')
    _, index = build_units([context, context])
    assert match_context(context, index).status is MappingStatus.EXACT
    assert match_context(dict(context, content='① 본문.'), index).status is MappingStatus.UNRESOLVED
    assert match_context(dict(context, hierarchy='법 1조 목적 2항'), index).status is MappingStatus.UNRESOLVED


def test_composite_evidence_accepts_full_item_reordering_but_rejects_cross_article_text():
    root = statute('법 1조 목적', '제1조(목적) 다음과 같다.')
    one = statute('법 1조 목적 1호', '1. 첫째')
    two = statute('법 1조 목적 2호', '2. 둘째')
    _, index = build_units([root, one, two])
    context = dict(root, content=root['content'] + '\n' + two['content'] + '\n' + one['content'])
    result = match_context(context, index)
    assert result.status is MappingStatus.ASSEMBLED_REORDERED
    assert len(result.unit_ids) == 3
    assert match_context(dict(context, content=context['content'] + ' 다른 조문'), index).status is MappingStatus.UNRESOLVED


def test_whitespace_collisions_are_not_silently_resolved():
    a = statute('법 1조 목적 1항', '① 본문')
    _, index = build_units([a, dict(a, content='①  본문')])
    assert match_context(a, index).status is MappingStatus.AMBIGUOUS


def test_metrics_require_all_components_and_deduplicate_evidence_across_chunks():
    qrels = {'a': {'e1': ['u1', 'u2'], 'e2': ['u3']}, 'b': {'e3': ['u4']}}
    ranks = {'a': [{'u1'}, {'u1'}, {'u2'}, {'u3'}], 'b': [{'u4'}]}
    summary, rows = evaluate_evidence(qrels, ranks, (1, 3, 4))
    assert summary['evidence_recall_at_1'] == .5
    assert summary['evidence_recall_at_3'] == .75
    assert summary['all_evidence_hit_at_3'] == .5
    assert summary['all_evidence_hit_at_4'] == 1
    assert summary['mrr'] == pytest.approx((1 / 3 + 1) / 2)
    assert rows[0]['hit_at_1'] == 0


def test_metrics_reject_missing_rankings_and_unresolved_qrels():
    with pytest.raises(ValueError, match='Query IDs differ'):
        evaluate_evidence({'q': {'e': ['u']}}, {})
    with pytest.raises(ValueError, match='Unresolved'):
        evaluate_evidence({'q': {'e': []}}, {'q': []})
    with pytest.raises(ValueError, match='No evaluable'):
        evaluate_evidence({}, {})


def test_fragment_coverage_requires_full_body_without_gaps_and_ignores_titles():
    units = {
        'u': Unit(
            index='법 1조', hierarchy='법 1조 1항', content='abcdef',
            unit_id='u', kind=UnitKind.PARAGRAPH, source_revision='test',
        ),
        't': Unit(
            index='법 1조', hierarchy='법 1조', content='title',
            unit_id='t', kind=UnitKind.ARTICLE_TITLE, source_revision='test',
        ),
    }
    def span(uid, start, end):
        return {'unit_id': uid, 'start': start, 'end': end, 'text': units[uid].content[start:end]}
    chunks = [[span('u', 0, 2), span('t', 0, 5)], [span('u', 3, 6)],
              [span('u', 1, 4)], [span('u', 0, 6)]]
    assert chunk_coverage_rankings(units, chunks) == [set(), set(), {'u'}, set()]
    with pytest.raises(ValueError, match='Invalid source span'):
        chunk_coverage_rankings(units, [[dict(span('u', 0, 3), text='fake')]])


def test_similar_scenarios_and_source_families_stay_in_one_group():
    queries = [
        Query(
            query_id=query_id,
            source_row_id=position,
            source_split='test',
            scenario=scenario,
            question=question,
            retrieval_query=scenario + question,
            reference_answer='참고 답변',
            n_hops=1,
            source_revision='test',
        )
        for position, (query_id, scenario, question) in enumerate([
            ('qa_1_a', '동일 시나리오입니다.', '질문1'),
            ('qa_2_b', '동일 시나리오입니다.', '질문2'),
            ('qa_2_c', '다른 사실관계', '질문3'),
        ])
    ]
    rows, _ = split_queries(queries)
    assert len({row.group_id for row in rows}) == 1
    assert len({row.split for row in rows}) == 1
    assert split_queries(queries) == split_queries(queries)


def test_configured_sources_paths_and_split_work_without_network(tmp_path, monkeypatch):
    import hashlib
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from llm_benchmark.koblex import config
    from llm_benchmark.koblex.download import download
    from llm_benchmark.koblex.prepare import (
        load_sources, map_sources, partition_dataset, prepare,
    )
    from llm_benchmark.koblex.audit import validate_evaluation_data
    from llm_benchmark.koblex.models import MappedDataset, PreparedDataset, SourceDataset
    from pathlib import Path

    monkeypatch.setattr(config, 'PROJECT_ROOT', tmp_path)
    context = statute('법 1조 목적 1항', '① 본문')
    qa = [{'id': 'qa_1_a', 'background': '사실관계', 'question': '질문',
           'answer': '답변', 'contexts': [context], 'n_hops': 1}]
    sources = []
    for repo, rows, revision in [('owner/qa', qa, 'qa-revision'),
                                 ('owner/statute', [context], 'corpus-revision')]:
        directory = tmp_path / 'input' / repo.split('/')[-1]
        directory.mkdir(parents=True)
        path = directory / 'selected.parquet'
        pq.write_table(pa.Table.from_pylist(rows), path)
        duplicate = directory / 'copy.parquet'
        duplicate.write_bytes(path.read_bytes())
        entries = [{'path': f'pinned/{repo.split("/")[-1]}/{p.name}',
                    'url': 'https://example.invalid/not-used', 'size_bytes': p.stat().st_size,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [path, duplicate]]
        sources.append({'repo': repo, 'revision': revision, 'files': entries})
    (tmp_path / 'sources.json').write_text(json.dumps({'raw_root': 'pinned', 'sources': sources}))
    settings_text = '''normalization_version = "fixture-v1"
source_manifest = "sources.json"
[paths]
raw = "input"
output = "prepared"
audit = "audit"
[qa]
repo = "owner/qa"
file = "selected.parquet"
duplicate_file = "copy.parquet"
split = "validation"
[corpus]
repo = "owner/statute"
file = "selected.parquet"
duplicate_file = "copy.parquet"
[split]
seed = 7
dev_ratio = 0.4
similarity_threshold = 0.9
shingle_size = 3
[download]
timeout_seconds = 2
'''
    path = tmp_path / 'settings.toml'
    path.write_text(settings_text)
    settings = config.load_settings(path)
    monkeypatch.chdir(tmp_path.parent)

    def reject_network(*args, **kwargs):
        raise AssertionError('해시가 맞는 원본은 네트워크에 접근하면 안 됩니다')

    monkeypatch.setattr('llm_benchmark.koblex.download.urlopen', reject_network)
    download(settings)
    sources = load_sources(settings.raw, settings)
    assert isinstance(sources, SourceDataset)

    def reject_file_access(*args, **kwargs):
        raise AssertionError('매핑·분할·데이터 검증 단계는 파일에 접근하면 안 됩니다')

    with monkeypatch.context() as phase_guard:
        phase_guard.setattr(Path, 'open', reject_file_access)
        mapped = map_sources(sources, settings)
        dataset = partition_dataset(mapped, settings)
        validate_evaluation_data(dataset)

    assert isinstance(mapped, MappedDataset)
    assert isinstance(dataset, PreparedDataset)
    assert dataset.mapped is mapped
    assert mapped.qrels['qa_1_a'][mapped.mappings[0].evidence_id] == [mapped.units[0].unit_id]
    prepare(settings=settings)
    output = tmp_path / 'prepared'
    query = json.loads((output / 'queries.jsonl').read_text())
    unit = json.loads((output / 'units.jsonl').read_text())
    split = json.loads((output / 'splits.jsonl').read_text())
    assert query['source_revision'] == 'qa-revision'
    assert query['source_split'] == split['source_split'] == 'validation'
    assert unit['source_revision'] == 'corpus-revision'
    assert unit['unit_id'] == _build_units([context], 'corpus-revision')[0][0].unit_id
    assert json.loads((output / 'manifest.json').read_text())['normalization_version'] == 'fixture-v1'
    assert (tmp_path / 'audit/mapping-ledger.json').is_file()
    path.write_text(settings_text.replace('dev_ratio = 0.4', 'dev_ratio = 1.5'))
    with pytest.raises(ValueError, match='dev_ratio'):
        config.load_settings(path)


def test_mapping_objects_and_enum_values_preserve_json_contract():
    from dataclasses import asdict
    import json

    from llm_benchmark.koblex.mapping import map_queries

    context = statute('법 1조 목적 1항', '① 본문')
    units, index = build_units([context])
    queries, mappings = map_queries(
        [{'id': 'q1', 'background': '사실관계', 'question': '질문',
          'answer': '답변', 'contexts': [context], 'n_hops': 1}],
        index, revision='qa-revision', source_split='test',
    )
    assert isinstance(queries[0], Query)
    assert isinstance(mappings[0], EvidenceMapping)
    assert units[0].kind is UnitKind.PARAGRAPH
    assert queries[0].evaluation_status is QueryStatus.READY
    assert mappings[0].status is MappingStatus.EXACT
    assert mappings[0].unit_ids == [units[0].unit_id]
    assert json.loads(json.dumps(asdict(mappings[0])))['status'] == 'exact'
    assert json.loads(json.dumps(asdict(queries[0])))['evaluation_status'] == 'ready'

    splits, _ = split_queries(queries)
    assert splits[0].split is SplitName.TEST
    assert json.loads(json.dumps(asdict(splits[0])))['split'] == 'test'
