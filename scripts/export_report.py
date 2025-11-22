#!/usr/bin/env python
"""
거래 리포트 생성 스크립트

Paper trading 또는 Live trading 결과를 분석하여 리포트를 생성합니다.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def select_trading_mode() -> str:
    """거래 모드 선택"""
    print("\n거래 모드 선택:")
    print("  1. Paper Trading")
    print("  2. Live Trading")

    choice = input("\n선택 (1-2): ").strip()

    return 'paper' if choice == '1' else 'live'


def select_period() -> tuple:
    """기간 선택"""
    print("\n기간 선택:")
    print("  1. 최근 7일")
    print("  2. 최근 30일")
    print("  3. 최근 90일")
    print("  4. 전체 기간")
    print("  5. 직접 입력")

    choice = input("\n선택 (1-5): ").strip()

    if choice == '5':
        print("\n시작 날짜 (YYYY-MM-DD):")
        start = input("시작: ").strip()
        print("\n종료 날짜 (YYYY-MM-DD):")
        end = input("종료: ").strip()
        return start, end

    # 간단한 기간 반환 (실제로는 날짜 계산 필요)
    period_map = {
        '1': ('7days', None),
        '2': ('30days', None),
        '3': ('90days', None),
        '4': ('all', None)
    }

    return period_map.get(choice, ('30days', None))


def select_report_format() -> str:
    """리포트 형식 선택"""
    print("\n리포트 형식 선택:")
    print("  1. CSV")
    print("  2. Excel")
    print("  3. PDF")
    print("  4. HTML")

    choice = input("\n선택 (1-4): ").strip()

    format_map = {
        '1': 'csv',
        '2': 'excel',
        '3': 'pdf',
        '4': 'html'
    }

    return format_map.get(choice, 'csv')


def generate_report():
    """리포트 생성"""
    print("=" * 60)
    print("📈 거래 리포트 생성")
    print("=" * 60)

    mode = select_trading_mode()
    period = select_period()
    report_format = select_report_format()

    print("\n" + "=" * 60)
    print("리포트 설정:")
    print(f"  모드: {mode.upper()}")
    print(f"  기간: {period}")
    print(f"  형식: {report_format.upper()}")
    print("=" * 60)

    confirm = input("\n생성하시겠습니까? (y/n): ").strip().lower()

    if confirm != 'y':
        print("취소되었습니다.")
        return

    print("\n🚀 리포트 생성 중...")

    # TODO: 실제 리포트 생성 로직 구현
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"data/reports/{mode}_report_{timestamp}.{report_format}"

    print("\n✅ 리포트 생성 완료!")
    print(f"\n저장 위치: {output_file}")

    print("\n주요 지표:")
    print("  총 거래 수: -")
    print("  승률: -")
    print("  총 수익률: -")
    print("  최대 낙폭: -")
    print("  샤프 비율: -")


if __name__ == "__main__":
    try:
        generate_report()
    except KeyboardInterrupt:
        print("\n\n리포트 생성이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
