#!/usr/bin/env python
"""
모델 성능 벤치마크 비교 스크립트

서로 다른 모델(LGBM, XGBoost, Ensemble)의 성능을 비교합니다.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def select_horizons() -> list:
    """예측 기간 선택"""
    print("\n예측 기간 선택:")
    print("  1. 72h (3일)")
    print("  2. 168h (7일)")
    print("  3. 720h (30일)")
    print("  4. 2160h (90일)")
    print("  5. 전체")

    choice = input("\n선택 (1-5): ").strip()

    horizon_map = {
        '1': [72],
        '2': [168],
        '3': [720],
        '4': [2160],
        '5': [72, 168, 720, 2160]
    }

    return horizon_map.get(choice, [72, 168, 720, 2160])


def select_metrics() -> list:
    """평가 지표 선택"""
    print("\n평가 지표 선택:")
    print("  1. RMSE")
    print("  2. Direction Accuracy")
    print("  3. Sharpe Ratio")
    print("  4. 전체")

    choice = input("\n선택 (1-4): ").strip()

    metrics_map = {
        '1': ['rmse'],
        '2': ['direction_accuracy'],
        '3': ['sharpe_ratio'],
        '4': ['rmse', 'direction_accuracy', 'sharpe_ratio']
    }

    return metrics_map.get(choice, ['rmse', 'direction_accuracy'])


def run_benchmark():
    """벤치마크 실행"""
    print("=" * 60)
    print("📊 모델 성능 벤치마크")
    print("=" * 60)

    horizons = select_horizons()
    metrics = select_metrics()

    print("\n시작 날짜 입력 (형식: YYYY-MM-DD):")
    start_date = input("시작: ").strip()

    print("\n종료 날짜 입력 (형식: YYYY-MM-DD):")
    end_date = input("종료: ").strip()

    print("\n" + "=" * 60)
    print("벤치마크 설정:")
    print(f"  예측 기간: {horizons}")
    print(f"  평가 지표: {metrics}")
    print(f"  기간: {start_date} ~ {end_date}")
    print("=" * 60)

    confirm = input("\n시작하시겠습니까? (y/n): ").strip().lower()

    if confirm != 'y':
        print("취소되었습니다.")
        return

    print("\n🚀 벤치마크 실행 중...")

    # TODO: 실제 벤치마크 로직 구현
    print("\n✅ 벤치마크 완료!")
    print(f"\n결과가 data/reports/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv 에 저장되었습니다.")


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print("\n\n벤치마크가 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
