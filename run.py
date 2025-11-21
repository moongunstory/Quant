# run.py
"""
BTC Jarvis Manager - Main Entry Point

자동 실행:
- 필수 데이터 파일이 없으면 → 누락된 모듈만 540일치 초기 수집
- 모든 파일이 있으면 → 각 데이터의 갱신 주기에 따라 필요한 항목만 '스마트 업데이트'
- 데이터 수집/가공 후 → 최근 540일 기준 데일리 롤링 학습 + 예측 + 예측 로그/실적 업데이트
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple

from ingest.orchestrator import DataOrchestrator
from process.builder import build_all_features
from model.daily.config import DailyConfig          # ✅ 여기서 가져와야 함
from model.daily.pipeline import run_daily_cycle       # ✅ 이건 그대로


def check_data_status() -> Tuple[List[str], int]:
    """
    어떤 모듈의 필수 파일이 누락되었는지 확인하고, 가장 최신 데이터의 날짜를 찾습니다.
    (smart_update 도입 후 days_old는 사용되지 않지만, 초기 진단용으로 유지)

    Returns:
        (missing_modules_list, days_since_latest_date)
        missing_modules_list가 비어있으면 데이터가 완전하다는 의미입니다.
    """
    data_dir = Path("data/raw")
    # 센티먼트는 현재 파이프라인에서 사용하지 않으므로 제외
    all_modules = ['binance', 'macro', 'news', 'onchain', 'derivatives']
    if not data_dir.exists():
        return all_modules, 0

    # 모듈별 필수 파일 목록
    essential_files = {
        'binance': [
            'binance/ohlcv_futures_1h.parquet',
            'binance/ohlcv_spot_1h.parquet',
            'binance/oi_1h.parquet',
            'binance/ls_ratio_top_1h.parquet',
            'binance/funding_rate.parquet',
        ],
        'macro': [
            'macro/fred_dgs10.parquet',
            'macro/yahoo_gspc.parquet',
        ],
        'news': [
            'news/news_raw.parquet',
        ],
        'onchain': [
            'onchain/blockchain_com_n-transactions.parquet',
        ],
        'derivatives': [
            'derivatives/deribit_btc_dvol.parquet',
        ],
    }

    missing_modules = []
    for module, files in essential_files.items():
        for f in files:
            if not (data_dir / f).exists():
                print(f"⚠️ 필수 데이터 파일 누락: {f} ({module} 모듈 재수집 필요)")
                if module not in missing_modules:
                    missing_modules.append(module)
    
    if missing_modules:
        return missing_modules, 0

    # 모든 필수 파일이 존재하면, 가장 최신 날짜를 찾음
    latest_date = None
    for parquet_file in data_dir.rglob('*.parquet'):
        try:
            df = pd.read_parquet(parquet_file)
            date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            if date_col in df.columns and not df.empty:
                file_latest = pd.to_datetime(df[date_col]).max()
                if latest_date is None or file_latest > latest_date:
                    latest_date = file_latest
        except Exception:
            continue
    
    if latest_date is None:
        return all_modules, 0

    days_old = (pd.Timestamp.now().normalize() - latest_date.normalize()).days
    return [], days_old


def _print_today_predictions(daily_pred: pd.DataFrame) -> None:
    if daily_pred is None or daily_pred.empty:
        print("\n(오늘 생성된 예측 레코드가 없습니다.)")
        return

    daily_pred = daily_pred.sort_values("horizon_days")

    print("\n=== 오늘 BTC 방향 예측 요약 ===")
    as_of_ts = daily_pred["as_of_ts"].iloc[0]
    print(f"기준 시각 (as_of_ts): {as_of_ts}")

    label_str = {-1: "하락(-1)", 0: "중립(0)", 1: "상승(+1)"}

    for _, row in daily_pred.iterrows():
        lbl = row.pred_label
        lbl_txt = label_str.get(lbl, f"알 수 없음({lbl})")

        line = (
            f"- Horizon {row.horizon_days:>2}일 "
            f"→ 예측: {lbl_txt}, "
            f"P(하락)={row.proba_down:.3f}, "
            f"P(중립)={row.proba_flat:.3f}, "
            f"P(상승)={row.proba_up:.3f}"
        )

        # exp_return 컬럼이 있고 값이 있으면 퍼센트로 같이 출력
        if "exp_return" in daily_pred.columns and pd.notna(row.get("exp_return", np.nan)):
            # 비율(0.04) → 퍼센트(4.0)
            er_pct = row.exp_return * 100.0
            samples = int(row.get("exp_return_samples", 0) or 0)
            line += f", 예상 수익률≈{er_pct:.2f}%, (과거 샘플 {samples}개)"

        print(line)

def main():
    """자동 실행"""
    
    print("=" * 60)
    print("🚀 BTC Jarvis Manager - 데이터 수집 & 피처 빌드 & 데일리 학습")
    print("=" * 60)
    
    # check_data_status는 이제 필수 파일 누락 여부만 확인하는 용도로 사용
    missing_modules, _ = check_data_status()
    
    orchestrator = DataOrchestrator()
    success = False
    
    # 1) 데이터 수집/업데이트
    if missing_modules:
        print(f"\n📂 데이터 불완전 ({', '.join(missing_modules)} 모듈 누락)")
        print("📥 누락된 데이터에 대해 540일치 초기 수집 시작...\n")
        success = orchestrator.initial_collection(days=540, targets=missing_modules)
    else:
        # 모든 필수 파일이 존재하면, 세부적인 갱신 여부는 smart_update에 위임
        print(f"\n📊 모든 필수 데이터 발견. 스마트 업데이트를 시작합니다...\n")
        success = orchestrator.smart_update()

    # 2) 데이터 수집이 성공했으면, 바로 피처 가공 + 병합
    if success:
        print("\n" + "=" * 60)
        print("🧮 피처 가공 및 병합 시작 (process.builder.build_all_features)")
        print("=" * 60)
        try:
            build_all_features()   # binance/onchain/macro/derivatives/news + master_1h
            print("\n✅ 피처/마스터 테이블 빌드 완료!")
        except Exception as e:
            print(f"\n❌ 피처 빌드 중 오류 발생: {e}")
            success = False

    # 3) 피처 빌드까지 성공했다면, 데일리 롤링 학습 + 예측 + 실적 업데이트
    daily_pred = None
    if success:
        print("\n" + "=" * 60)
        print("🤖 데일리 롤링 학습 + 예측 + 예측 로그/실적 업데이트 시작 (model.daily.run_daily_cycle)")
        print("=" * 60)
        try:
            master_path = Path("data/processed/master_features_1h.parquet")
            if not master_path.exists():
                raise FileNotFoundError(f"마스터 피처 파일이 없습니다: {master_path}")
            
            df_master = pd.read_parquet(master_path)

            cfg = DailyConfig()
            daily_pred = run_daily_cycle(cfg, df_master=df_master)

            if daily_pred is not None and not daily_pred.empty:
                print("\n✅ 데일리 롤링 학습 + 예측 1회 완료!")
            else:
                print("\nℹ️ 오늘은 학습이 스킵되었거나, 예측만 업데이트되었습니다.")
        except Exception as e:
            print(f"\n❌ 데일리 학습/예측 루프 중 오류 발생: {e}")
            success = False

    # 🔹 오늘 예측 결과를 즉시 로그로 출력
    if daily_pred is not None and not daily_pred.empty:
        _print_today_predictions(daily_pred)

    print("\n" + "=" * 60)
    if success:
        print("✅ 전체 파이프라인 완료! (수집 + 피처 빌드 + 데일리 학습/예측)")
    else:
        print("❌ 일부 작업 실패 (로그 확인 필요)")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
