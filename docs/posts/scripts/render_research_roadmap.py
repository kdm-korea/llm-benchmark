from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'assets/images/legal-rag-research-plan/research-roadmap.png'
STAGES = (
    ('임베딩 모델 선정', '산출물: 문서 단위 검색 기준선', '완료'),
    ('평가 데이터와 정답 기준', '산출물: 질문·근거·기대 답변과 평가 규칙', '연구 항목'),
    ('법률 텍스트 청킹', '산출물: 길이·문단·법률 구조별 비교', '연구 항목'),
    ('PDF 파싱', '산출물: 문단·표·이미지 추출 품질 비교', '연구 항목'),
    ('PDF 청킹·임베딩', '산출물: 페이지 경계·문서 구조 처리 비교', '연구 항목'),
    ('검색·재정렬', '산출물: 검색 방식·reranker 효과 비교', '연구 항목'),
    ('근거 기반 답변', '산출물: 사실·법률 적용·인용 검증', '연구 항목'),
)


def main() -> None:
    font = font_manager.FontProperties(fname='/System/Library/Fonts/Supplemental/AppleGothic.ttf')
    plt.rcParams.update({'font.family': font.get_name(), 'axes.unicode_minus': False})
    figure, axes = plt.subplots(figsize=(9, 12))
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.87)
    axes.set(xlim=(0, 1), ylim=(0, 1))
    axes.axis('off')
    figure.text(0.08, 0.95, '법률 RAG 연구 순서', fontsize=27, color='#172554')
    figure.text(0.08, 0.91, '모델 선정부터 근거 기반 답변 검증까지', fontsize=15, color='#475569')
    for index, (title, output, status) in enumerate(STAGES):
        bottom = 0.87 - index * 0.13
        color = '#ecfdf5' if index == 0 else '#eff6ff'
        axes.add_patch(
            FancyBboxPatch(
                (0.04, bottom),
                0.92,
                0.10,
                boxstyle='round,pad=0.008',
                facecolor=color,
                edgecolor='#cbd5e1',
            )
        )
        axes.text(
            0.075,
            bottom + 0.06,
            f'{index + 1:02d}  {title}',
            fontsize=18,
            va='center',
            color='#172554',
        )
        axes.text(0.075, bottom + 0.022, output, fontsize=12, va='center', color='#475569')
        axes.text(
            0.92,
            bottom + 0.06,
            status,
            fontsize=11,
            ha='right',
            va='center',
            color='#047857' if index == 0 else '#64748b',
        )
        if index < len(STAGES) - 1:
            axes.annotate(
                '',
                xy=(0.50, bottom - 0.020),
                xytext=(0.50, bottom - 0.009),
                arrowprops={'arrowstyle': '->', 'color': '#64748b', 'lw': 1.7},
            )
    figure.text(
        0.08,
        0.035,
        '실험 진행 순서 · 서비스 처리 순서나 성능 측정 결과가 아님',
        fontsize=11,
        color='#64748b',
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180, facecolor='white')
    plt.close(figure)
    print(OUTPUT)


if __name__ == '__main__':
    main()
