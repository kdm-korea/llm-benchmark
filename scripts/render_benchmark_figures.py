"""Validate saved benchmark outputs twice, then render publication figures.

Usage: MPLCONFIGDIR=/private/tmp/legal-mpl .venv/bin/python scripts/render_benchmark_figures.py
Requires matplotlib (plotting only; no model inference).
"""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/20261020T020620+0900-de87f657'
OUT = ROOT / 'assets/images/legal-embedding-recall'
K = [1, 5, 10, 20, 50, 100]
MODELS = ['kure-v1', 'bge-m3', 'multilingual-e5-large']
LABELS = ['KURE-v1', 'BGE-M3', 'Multilingual E5 Large']
COLORS = ['#147D78', '#4263A3', '#C17B32']


def load(path):
    return json.loads(path.read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12), (a, b)


def main():
    rows = {r['model']: r for r in csv.DictReader((RUN / 'recall_summary.csv').open())}
    manifest = load(RUN / 'manifest.json')
    qrels = load(ROOT / 'data/prepared/eve_legal/qrels.json')
    values, ndcg_values, auxiliary = {}, {}, {}
    for model in MODELS:
        queries = load(RUN / f'per_query/{model}.json')['queries']
        native = load(RUN / f'raw/{model}.json')
        raw = native['all_scores']
        ndcg_values[model] = [mean(v[f'ndcg_cut_{k}'] for v in raw.values()) for k in K]
        for k, score in zip(K, ndcg_values[model]):
            assert abs(score - native['ndcg'][f'NDCG@{k}']) <= 0.0000051
        assert len(queries) == len(qrels) == manifest['query_count'] == 5998
        assert {q['query_id'] for q in queries} == set(qrels) == set(raw)
        values[model] = [float(rows[model][f'recall_at_{k}']) for k in K]
        # Pass 1: independently aggregate stored per-query and MTEB scores.
        for k, expected in zip(K, values[model]):
            close(mean(q[f'recall_at_{k}'] for q in queries), expected)
            close(mean(s[f'recall_{k}'] for s in raw.values()), expected)
        # Pass 2: recompute from ranked document IDs and prepared positive qrels.
        computed = {k: [] for k in K}
        computed_ndcg = {k: [] for k in K}
        first_hits = no_hits = full_hits = 0
        for q in queries:
            relevant = {d for d, grade in qrels[q['query_id']].items() if grade > 0}
            assert relevant == set(q['relevant_doc_ids'])
            ranking = q['retrieved_doc_ids_top100']
            assert len(ranking) == len(set(ranking)) == 100
            for k in K:
                score = len(relevant.intersection(ranking[:k])) / len(relevant)
                close(score, q[f'recall_at_{k}'])
                computed[k].append(score)
                dcg = sum(1 / math.log2(rank + 2) for rank, doc in enumerate(ranking[:k]) if doc in relevant)
                ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
                ndcg = dcg / ideal
                close(ndcg, raw[q['query_id']][f'ndcg_cut_{k}'])
                computed_ndcg[k].append(ndcg)
            first_hits += ranking[0] in relevant
            no_hits += not relevant.intersection(ranking)
            full_hits += relevant.issubset(ranking)
        for k, expected in zip(K, values[model]):
            close(mean(computed[k]), expected)
        for k, expected in zip(K, ndcg_values[model]):
            close(mean(computed_ndcg[k]), expected)
        auxiliary[model] = dict(hit_at_1_count=first_hits,
                                hit_at_1_percent=first_hits / len(queries) * 100,
                                no_hit_at_100_count=no_hits,
                                full_recall_at_100_count=full_hits,
                                full_recall_at_100_percent=full_hits / len(queries) * 100)

    for metric in [values, ndcg_values]:
        for i in range(len(K)):
            assert metric['kure-v1'][i] > metric['bge-m3'][i] > metric['multilingual-e5-large'][i]

    font = Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
    assert font.exists(), 'Korean font required'
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font)).get_name(),
                         'font.size': 12, 'axes.unicode_minus': False,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    for ax, scores, title, xlabel, upper, percent in [
        (axes[0], [values[m][3] * 100 for m in MODELS], '관련 문서 회수: Recall@20', '평균 Recall@20 (%)', 100, True),
        (axes[1], [ndcg_values[m][2] for m in MODELS], '상위 순위 품질: 이진 nDCG@10', '평균 이진 nDCG@10 (0~1)', 1, False),
    ]:
        bars = ax.barh(LABELS, scores, color=COLORS, height=.52)
        ax.invert_yaxis()
        ax.set(xlim=(0, upper), xlabel=xlabel, title=title)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=.16)
        for bar, expected in zip(bars, scores):
            close(bar.get_width(), expected)
            formatted = f'{expected:.2f}%' if percent else f'{expected:.5f}'
            label = ax.text(expected + upper * .018, bar.get_y() + bar.get_height()/2,
                            formatted, va='center', weight='bold')
            assert label.get_text() == formatted
    fig.suptitle('판례 59,786개 · 질의 5,998개 · 전체 문서 정확 검색', fontsize=11, color='#555555')
    fig.savefig(OUT / 'metrics-overview.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    for model, label, color, marker in zip(MODELS, LABELS, COLORS, ['o', 's', '^']):
        ys = [v * 100 for v in values[model]]
        line, = ax.plot(K, ys, label=label, color=color, marker=marker, linewidth=2.4, markersize=7)
        for actual, expected in zip(line.get_ydata(), ys):
            close(actual, expected)
        tag = ax.annotate(f'{ys[-1]:.2f}%', (100, ys[-1]), xytext=(9, 0),
                          textcoords='offset points', va='center', color=color, weight='bold')
        assert tag.get_text() == f'{ys[-1]:.2f}%'
    ax.set(xscale='log', xlim=(.85, 155), ylim=(0, 100),
           xlabel='검색 결과 수 K (로그 간격)', ylabel='평균 Recall@K (%)',
           title='검색 결과 수에 따른 Recall@K 비교')
    ax.set_xticks(K, labels=[str(k) for k in K])
    ax.grid(alpha=.16)
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(OUT / 'recall-curve.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    for model, label, color, marker in zip(MODELS, LABELS, COLORS, ['o', 's', '^']):
        ys = ndcg_values[model]
        line, = ax.plot(K, ys, label=label, color=color, marker=marker, linewidth=2.4, markersize=7)
        for actual, expected in zip(line.get_ydata(), ys):
            close(actual, expected)
        tag = ax.annotate(f'{ys[-1]:.5f}', (100, ys[-1]), xytext=(9, 0),
                          textcoords='offset points', va='center', color=color, weight='bold')
        assert tag.get_text() == f'{ys[-1]:.5f}'
    ax.set(xscale='log', xlim=(.85, 180), ylim=(0, 1),
           xlabel='검색 결과 수 K (로그 간격)', ylabel='평균 이진 nDCG@K (0~1)',
           title='검색 결과 수에 따른 이진 nDCG@K 비교')
    ax.set_xticks(K, labels=[str(k) for k in K])
    ax.grid(alpha=.16)
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(OUT / 'ndcg-curve.png', dpi=200)
    plt.close(fig)

    audit = {'source_run': RUN.name, 'verification': ['stored per-query + MTEB means vs CSV',
             'recomputed Recall and binary DCG/IDCG from rankings and positive qrels', 'plotted coordinates and labels vs verified values'],
             'recall': {m: dict(zip(map(str, K), values[m])) for m in MODELS},
             'ndcg': {m: dict(zip(map(str, K), ndcg_values[m])) for m in MODELS},
             'ndcg_relevance': 'binary_grade_gt_zero',
             'rendering': {'matplotlib': matplotlib.__version__, 'dpi': 200},
             'auxiliary': auxiliary,
             'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [RUN/'recall_summary.csv', RUN/'manifest.json', ROOT/'data/prepared/eve_legal/qrels.json']
                 + [RUN/f'{folder}/{m}.json' for folder in ['raw', 'per_query'] for m in MODELS]},
             'image_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.png'))}}
    (OUT / 'verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print('PASS: two independent numerical checks + plot coordinates/labels; 3 PNGs saved')


if __name__ == '__main__':
    main()
