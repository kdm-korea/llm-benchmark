from llm_benchmark.common.hashing import compact_digest, file_sha256, fingerprint


def test_existing_hash_profiles_remain_unchanged():
    value = {'한글': '값', 'a': [1, 2]}
    assert fingerprint(value) == '43922a2c8b2c7c7dd3d2b49092c40e03926c1ea0d7f841abc582edf35814bd93'
    assert compact_digest(value) == '6fa9d368ba0aaa04a1e93903fa9bc8e7825dc36ad55c9eb8ef3ae03376d99341'
    assert fingerprint(dict(reversed(list(value.items())))) == fingerprint(value)
    assert compact_digest(dict(reversed(list(value.items())))) != compact_digest(value)


def test_file_checksum_uses_original_bytes(tmp_path):
    target = tmp_path / 'file'
    target.write_bytes(b'abc')
    assert file_sha256(target) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
