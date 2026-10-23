"""Check published metric tables and images against native and independently verified data."""
import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/20261020T020620+0900-de87f657'
K = [1, 5, 10, 20, 50, 100]
LABELS = {'KURE-v1': 'kure-v1', 'BGE-M3': 'bge-m3',
          'Multilingual E5 Large': 'multilingual-e5-large'}
DOCS = ['README.md', 'docs/benchmarks/results/2026-10-20-embedding-recall.md',
        'docs/posts/2026-10-23-legal-embedding-recall-benchmark.md']


def load(path):
    return json.loads(path.read_text())


def verify():
    audit = load(ROOT / 'assets/images/legal-embedding-recall/verification.json')
    csv_rows = {r['model']: r for r in csv.DictReader((RUN / 'recall_summary.csv').open())}
    native = {m: load(RUN / f'raw/{m}.json') for m in LABELS.values()}
    cells = links = 0
    for name in DOCS:
        p = ROOT / name
        text = p.read_text()
        assert '이진' in text and '0~1' in text
        for label, model in LABELS.items():
            rows = [[c.strip().replace('**', '') for c in line.strip('|').split('|')]
                    for line in text.splitlines() if line.startswith('|')]
            metric_rows = [r for r in rows if len(r) == 7 and r[0] == label]
            assert len(metric_rows) == 2, (name, label, metric_rows)
            recall = next(r[1:] for r in metric_rows if r[1].endswith('%'))
            ndcg = next(r[1:] for r in metric_rows if not r[1].endswith('%'))
            # First pass against original artifacts, including native rounded nDCG.
            assert recall == [f"{float(csv_rows[model][f'recall_at_{k}']) * 100:.2f}%" for k in K]
            assert ndcg == [f"{native[model]['ndcg'][f'NDCG@{k}']:.5f}" for k in K]
            # Second pass against full precision values independently recomputed by render script.
            assert recall == [f"{audit['recall'][model][str(k)] * 100:.2f}%" for k in K]
            assert ndcg == [f"{audit['ndcg'][model][str(k)]:.5f}" for k in K]
            cells += 12
        assert not any(phrase in text for phrase in ['Recall@K만 평가', '청킹, nDCG,'])
        for liquid_path in re.findall(r"'(/assets/images/[^']+)'", text):
            assert (ROOT / liquid_path.lstrip('/')).is_file()
            links += 1
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            if link.startswith(('https://', 'http://', '#', '{{')):
                continue
            assert (p.parent / link.split('#')[0]).exists(), (name, link)
            links += 1
    for model in LABELS.values():
        # Binary nDCG@1 must equal Hit@1; check with independently stored success values.
        hit = mean(v['success_1'] for v in native[model]['all_scores'].values())
        assert abs(audit['ndcg'][model]['1'] - hit) < 1e-12
    for relative, expected in audit['source_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    for name, expected in audit['image_sha256'].items():
        assert hashlib.sha256((ROOT / 'assets/images/legal-embedding-recall' / name).read_bytes()).hexdigest() == expected
    print(f'PASS: {cells} metric cells checked twice, {links} links, nDCG@1=Hit@1, source/image hashes')


if __name__ == '__main__':
    verify()
