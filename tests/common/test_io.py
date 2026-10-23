import pytest

from llm_benchmark.common.io import read_jsonl, write_json, write_jsonl


@pytest.mark.parametrize('writer', [write_json, write_jsonl])
def test_serialization_failure_preserves_existing_file_and_cleans_temporary_files(tmp_path, writer):
    target = tmp_path / 'data.json'
    target.write_text('previous data')
    with pytest.raises(ValueError):
        writer(target, [{'valid': '한글'}, {'invalid': float('nan')}])
    assert target.read_text() == 'previous data'
    assert list(tmp_path.iterdir()) == [target]


def test_jsonl_reports_source_position_and_preserves_unicode(tmp_path):
    target = tmp_path / 'nested/data.jsonl'
    write_jsonl(target, [{'값': '원문'}, {'number': 2}])
    assert list(read_jsonl(target)) == [{'값': '원문'}, {'number': 2}]
    with target.open('a') as stream:
        stream.write('\ninvalid\n')
    with pytest.raises(ValueError, match='data.jsonl:4'):
        list(read_jsonl(target))
