import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'docs/benchmarks/koblex/data-flow.md'
OUTPUT = ROOT / 'assets/images/koblex-legal-evaluation-dataset/data-relationship.png'
FONT = font_manager.FontProperties(fname='/System/Library/Fonts/Supplemental/AppleGothic.ttf')


def draw_table(
    axes: Axes, left: float, top: float, width: float, title: str, fields: list[str]
) -> float:
    row_height = 0.38
    header_height = 0.62
    bottom = top - header_height - len(fields) * row_height
    axes.add_patch(
        Rectangle(
            (left, bottom),
            width,
            top - bottom,
            facecolor='white',
            edgecolor='#64748b',
            linewidth=1.4,
        )
    )
    axes.add_patch(
        Rectangle(
            (left, top - header_height),
            width,
            header_height,
            facecolor='#e8effa',
            edgecolor='#64748b',
            linewidth=1.4,
        )
    )
    axes.text(left + 0.16, top - header_height / 2, title, va='center', fontsize=13, weight='bold')
    for position, field in enumerate(fields):
        row_top = top - header_height - position * row_height
        if position % 2 == 0:
            axes.add_patch(
                Rectangle(
                    (left + 0.01, row_top - row_height),
                    width - 0.02,
                    row_height,
                    facecolor='#f7f9fc',
                    edgecolor='none',
                )
            )
        axes.text(left + 0.16, row_top - row_height / 2, field, va='center', fontsize=12)
    return bottom


def connect(
    axes: Axes,
    points: list[tuple[float, float]],
    source_count: tuple[float, float],
    target_count: tuple[float, float],
) -> None:
    for start, end in zip(points[:-2], points[1:-1], strict=True):
        axes.plot([start[0], end[0]], [start[1], end[1]], color='#475569', linewidth=1.5)
    axes.annotate(
        '',
        xy=points[-1],
        xytext=points[-2],
        arrowprops={'arrowstyle': '->', 'color': '#475569', 'lw': 1.5},
    )
    axes.text(*source_count, '1', ha='center', va='center', fontsize=11, backgroundcolor='white')
    axes.text(*target_count, '0..*', ha='center', va='center', fontsize=11, backgroundcolor='white')


def main() -> None:
    source = SOURCE.read_text()
    tables = {
        name: (title, fields.split())
        for name, title, fields in re.findall(r'class (\w+)\["([^"]+)"\] \{([^}]+)\}', source)
    }
    assert set(tables) == {'QUERIES', 'EVIDENCE', 'EVIDENCE_UNIT', 'UNITS', 'SPLITS'}
    plt.rcParams.update(
        {'font.family': FONT.get_name(), 'axes.unicode_minus': False, 'text.color': '#172033'}
    )
    figure, axes = plt.subplots(figsize=(14, 10))
    figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    axes.set(xlim=(0, 14), ylim=(0, 10))
    axes.axis('off')
    axes.text(0.4, 9.55, 'KoBLEX 데이터의 필드와 연결 관계', fontsize=23, weight='bold')
    axes.text(
        0.4,
        9.03,
        'JSON·JSONL을 테이블 형태로 표현한 관계도 · 화살표의 숫자는 연결 개수',
        fontsize=12,
        color='#475569',
    )
    positions = {
        'QUERIES': (0.4, 8.5, 3.6),
        'EVIDENCE': (5.05, 8.5, 3.8),
        'UNITS': (10.05, 8.5, 3.55),
        'SPLITS': (0.4, 3.7, 3.6),
        'EVIDENCE_UNIT': (5.05, 3.7, 4.5),
    }
    bottoms = {}
    for name, (left, top, width) in positions.items():
        title, fields = tables[name]
        bottoms[name] = draw_table(axes, left, top, width, title, fields)
    connect(axes, [(4.0, 7.1), (5.05, 7.1)], (4.15, 7.35), (4.83, 7.35))
    axes.text(4.53, 6.74, 'query_id', ha='center', fontsize=10)
    axes.annotate(
        '',
        xy=(2.2, 3.7),
        xytext=(2.2, bottoms['QUERIES']),
        arrowprops={'arrowstyle': '->', 'color': '#475569', 'lw': 1.5},
    )
    axes.text(2.0, bottoms['QUERIES'] - 0.22, '1', ha='center', fontsize=11)
    axes.text(2.0, 3.9, '1', ha='center', fontsize=11)
    axes.text(2.4, 4.7, 'query_id', fontsize=11)
    connect(
        axes,
        [(6.95, bottoms['EVIDENCE']), (6.95, 3.7)],
        (6.73, bottoms['EVIDENCE'] - 0.2),
        (6.6, 3.92),
    )
    axes.text(7.18, 4.38, 'query_id\n+ context_position', fontsize=10, linespacing=1.5)
    connect(
        axes,
        [(11.82, bottoms['UNITS']), (11.82, 2.8), (9.55, 2.8)],
        (12.04, bottoms['UNITS'] - 0.2),
        (9.85, 3.04),
    )
    axes.text(12.02, 4.4, 'unit_id', fontsize=11)
    axes.text(
        0.4,
        1.27,
        '질문 하나에 여러 근거, 근거 하나에 여러 법령 항목이 연결될 수 있다.',
        fontsize=12,
    )
    axes.text(
        0.4,
        0.8,
        '‘근거와 법령 연결’은 evidence.unit_ids 배열을 펼친 표현이며 별도 저장 파일이 아니다.',
        fontsize=12,
    )
    axes.text(
        0.4,
        0.35,
        'qrels.json은 질문 → 근거 → 법령 ID 목록을 담는 검색 평가용 정답표다.',
        fontsize=12,
        color='#475569',
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor='white')
    plt.close(figure)
    print(OUTPUT)


if __name__ == '__main__':
    main()
