"""테스트 수집 순서와 관계없이 실험 사이의 import 경계가 지켜지는지 검증한다."""

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / 'src/llm_benchmark'
EXPERIMENTS = tuple(sorted(
    path.name for path in SOURCE_ROOT.iterdir()
    if path.is_dir() and path.name != 'common' and (path / '__init__.py').exists()
))
MODULES = {
    area: [
        '.'.join(path.relative_to(SOURCE_ROOT).with_suffix('').parts)
        for path in sorted((SOURCE_ROOT / area).rglob('*.py'))
        if path.name != '__init__.py'
    ]
    for area in ('common', *EXPERIMENTS)
}
OPTIONAL_LIBRARIES = (
    'numpy', 'yaml', 'huggingface_hub', 'torch',
    'sentence_transformers', 'mteb', 'pyarrow',
)


def _imported_modules(node):
    if isinstance(node, ast.ImportFrom):
        return [node.module or '']
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    return []


def _is_blocked(module_name, blocked_modules):
    return any(
        module_name == blocked or module_name.startswith(blocked + '.')
        for blocked in blocked_modules
    )


def test_experiments_and_their_tests_never_import_each_other():
    for area in (*EXPERIMENTS, 'common'):
        other_experiments = [name for name in EXPERIMENTS if name != area]
        blocked_modules = [
            prefix + name
            for name in other_experiments
            for prefix in ('llm_benchmark.', 'tests.')
        ]
        source_roots = [ROOT / 'src/llm_benchmark' / area, ROOT / 'tests' / area]
        for source_root in source_roots:
            for path in source_root.rglob('*.py'):
                for node in ast.walk(ast.parse(path.read_text())):
                    for module_name in _imported_modules(node):
                        assert not _is_blocked(module_name, blocked_modules), path
                        assert not module_name.startswith('test_'), path
                    if isinstance(node, ast.ImportFrom) and node.level:
                        # 상대 import가 현재 실험 폴더를 벗어나 다른 실험을 참조하지 못하게 한다.
                        package_depth = len(path.relative_to(source_root).parent.parts) + 1
                        assert node.level <= package_depth, path


# 앞선 테스트의 import 캐시가 의존 경계를 가리지 않도록 새 인터프리터에서 검사한다.
IMPORT_PROBE = '''
import importlib
import importlib.abc
import json
import sys

source_root, blocked_json, modules_json = sys.argv[1:]
sys.path.insert(0, source_root)
blocked_modules = json.loads(blocked_json)

class BlockOtherExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for blocked in blocked_modules:
            if fullname == blocked or fullname.startswith(blocked + '.'):
                raise ImportError('Forbidden dependency: ' + fullname)

sys.meta_path.insert(0, BlockOtherExperiments())
for module_name in json.loads(modules_json):
    importlib.import_module('llm_benchmark.' + module_name)
'''


@pytest.mark.parametrize('area', ['common', *EXPERIMENTS])
def test_imports_in_fresh_process_with_other_experiments_blocked(area):
    import json

    if area == 'embedding' and importlib.util.find_spec('numpy') is None:
        pytest.skip('embedding extra is not installed')

    blocked_modules = ['llm_benchmark.' + name for name in EXPERIMENTS if name != area]
    if area != 'embedding':
        blocked_modules.extend(OPTIONAL_LIBRARIES)

    command = [
        sys.executable, '-I', '-c', IMPORT_PROBE,
        str(ROOT / 'src'), json.dumps(blocked_modules), json.dumps(MODULES[area]),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_embedding_data_models_import_without_optional_libraries() -> None:
    import json

    command = [
        sys.executable, '-I', '-c', IMPORT_PROBE,
        str(ROOT / 'src'), json.dumps(OPTIONAL_LIBRARIES),
        json.dumps(['embedding.models', 'embedding.dataset']),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
