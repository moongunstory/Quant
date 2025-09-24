# train/prepare/run.py

import sys
import os
import pandas as pd

# --- 프로젝트 루트 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

# --- 모듈 임포트 ---
from ai_binance.config.paths import PROCESSED_DATA_DIR
from ai_binance.config.settings import DEFAULT_START_DATE, DEFAULT_END_DATE
from ai_binance.train.prepare.collection.fetch_binance import fetch_binance_data
from ai_binance.train.prepare.collection.fetch_dune import fetch_dune_data
from ai_binance.train.prepare.process.data_splitter import split_data
from ai_binance.train.prepare.process.feature_engineering.ohlcv_features import load_ohlcv_data, compute_ohlcv_features
from ai_binance.train.prepare.process.feature_engineering.funding_index_features import load_funding_data, compute_funding_features, load_index_data, compute_index_features
from ai_binance.train.prepare.process.feature_engineering.dune_features import load_dune_raw_data, compute_dune_features
from ai_binance.train.prepare.process.feature_registry import fixed_features, tunable_features, get_representative_config
from ai_binance.train.prepare.process.feature_cleaning import merge_and_clean_features


def _calculate_required_periods(fixed_feats: dict, tunable_feats: dict) -> tuple[int, int]:
    """피처 설정에서 최대 lookback 기간과 chikou span 기간을 동적으로 계산합니다."""
    max_lookback = 0
    chikou_period = 0

    # 1. 튜닝 가능한 피처에서 최대 윈도우 찾기
    for group_feats in tunable_feats.values():
        for feat_cfg in group_feats.values():
            if isinstance(feat_cfg, dict) and "range" in feat_cfg:
                max_lookback = max(max_lookback, feat_cfg["range"]["max"])

    # 2. 고정 피처에서 최대 윈도우 찾기 (특히 Ichimoku)
    if "ichimoku" in fixed_feats.get("ohlcv", []):
        # Ichimoku: window1=9, window2=26, window3=52 (Senkou B)
        # Chikou: 26
        max_lookback = max(max_lookback, 52)
        chikou_period = 26

    return max_lookback, chikou_period


def run_fetch():
    """모든 원본 데이터를 수집합니다."""
    print("▶ 원본 데이터 수집 시작")

    # --- 동적 날짜 계산 ---
    # 원리: 피처 레지스트리를 분석하여 필요한 lookback/look-forward 기간을 동적으로 계산
    max_lookback, chikou_period = _calculate_required_periods(fixed_features, tunable_features)
    print(f"계산된 최대 Lookback: {max_lookback}, Chikou 기간: {chikou_period}")

    # 5분봉 기준 Timedelta 계산
    interval_minutes = 5
    start_timedelta = pd.Timedelta(minutes=interval_minutes * max_lookback)
    end_timedelta = pd.Timedelta(minutes=interval_minutes * chikou_period)

    # 실제 수집할 날짜 범위 계산
    fetch_start_date = (pd.to_datetime(DEFAULT_START_DATE) - start_timedelta).strftime('%Y-%m-%d %H:%M:%S')
    fetch_end_date = (pd.to_datetime(DEFAULT_END_DATE) + end_timedelta).strftime('%Y-%m-%d %H:%M:%S')

    print(f"동적 수집 기간: {fetch_start_date} ~ {fetch_end_date}")

    fetch_binance_data(fetch_start_date, fetch_end_date)
    fetch_dune_data()


def run_process(symbol: str):
    """피처를 계산하고 최종 데이터셋을 분할하여 저장합니다."""
    print(f"▶ 피처 계산 및 데이터셋 생성 시작 (Symbol: {symbol})")

    # 1. 원시 데이터 로딩
    df_ohlcv = load_ohlcv_data(symbol)
    df_funding = load_funding_data(symbol)
    df_index = load_index_data(symbol)  # ✅ index 데이터 로딩 추가
    df_dune = load_dune_raw_data(symbol)

    # 2. 대표 config 생성
    rep_config = get_representative_config(tunable_features)
    
    # 3. 그룹별 config 준비 + 고정 피처 포함
    ohlcv_cfg = rep_config.get("ohlcv", {})
    ohlcv_cfg.update({feat: True for feat in fixed_features["ohlcv"]})

    funding_cfg = rep_config.get("funding", {})
    funding_cfg.update({feat: True for feat in fixed_features["funding"]})

    index_cfg = rep_config.get("index", {})
    index_cfg.update({feat: True for feat in fixed_features["index"]})

    dune_cfg = rep_config.get("dune", {})
    dune_cfg.update({})  # Dune은 보통 튜닝형만?

    # 4. 피처 계산
    df_ohlcv_feats = compute_ohlcv_features(df_ohlcv, ohlcv_cfg)
    df_funding_feats = compute_funding_features(df_funding, funding_cfg)
    
    # ✅ index 피처 계산: windows만 추출
    index_windows = index_cfg.get("windows", [])
    df_index_feats = compute_index_features(df_index, index_windows)

    df_dune_feats = compute_dune_features(df_dune, dune_cfg)

    # 5. 병합 + 클리닝 (timestamp 기준)
    mega_df = merge_and_clean_features(
        df_main=df_ohlcv_feats,
        df_background_list=[df_funding_feats, df_index_feats, df_dune_feats],  # ✅ index 피처 포함
        on="timestamp"
    )

    # 6. 데이터 분할
    df_train, df_val, df_test = split_data(mega_df)

    # 7. 저장
    save_path = PROCESSED_DATA_DIR / symbol.lower()
    save_path.mkdir(parents=True, exist_ok=True)

    df_train.to_parquet(save_path / "train_set.parquet", index=False)
    df_val.to_parquet(save_path / "validation_set.parquet", index=False)
    df_test.to_parquet(save_path / "test_set.parquet", index=False)

    print(f"✅ 최종 데이터셋 저장 완료 → {save_path}")


def main():
    """데이터 준비 파이프라인 (수집 -> 가공)을 실행합니다."""
    
    # --- 설정 ---
    SYMBOLS_TO_PROCESS = ["ETHUSDT", "BTCUSDT"]

    # 1단계: 데이터 수집
    run_fetch()
    
    # 2단계: 데이터 가공
    for symbol in SYMBOLS_TO_PROCESS:
        run_process(symbol)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
