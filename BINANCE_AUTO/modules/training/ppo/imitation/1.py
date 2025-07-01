# tp_sl_hit_debugger.py

import os
import pandas as pd
import numpy as np
import logging
from datetime import timedelta

# 설정
TP_THRESHOLD = 0.008  # 0.8%
SL_THRESHOLD = -0.008  # -0.8%
LABEL_HORIZON = 20     # 15min 기준 20개 = 약 5시간
ENTRY_TF = "15min"
EVAL_TF = "5min"
PKL_PATH = r"C:\gtpbitcoin\BINANCE_AUTO\data\label\train_long.pkl"  # 또는 train_short.pkl
DIRECTION = "long"  # 또는 "short"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_tp_sl_hits(entry_df, eval_df, direction):
    logger.info("⏱️ 타임존 정렬 시작")
    entry_df.index = pd.to_datetime(entry_df.index).tz_localize("UTC") if entry_df.index.tz is None else entry_df.index.tz_convert("UTC")
    eval_df.index = pd.to_datetime(eval_df.index).tz_localize("UTC") if eval_df.index.tz is None else eval_df.index.tz_convert("UTC")
    logger.info(f"Entry 타임존: {entry_df.index.tz}, Eval 타임존: {eval_df.index.tz}")

    tp_thresh = abs(TP_THRESHOLD)
    sl_thresh = abs(SL_THRESHOLD)

    tp_hits = np.zeros(len(entry_df), dtype=bool)
    sl_hits = np.zeros(len(entry_df), dtype=bool)

    for i, (entry_time, entry_price) in enumerate(entry_df['close'].items()):
        tp_price = entry_price * (1 + tp_thresh) if direction == "long" else entry_price * (1 - tp_thresh)
        sl_price = entry_price * (1 - sl_thresh) if direction == "long" else entry_price * (1 + sl_thresh)

        future_start = entry_time  # ← 이 시점 포함
        future_end = entry_time + timedelta(minutes=15 * LABEL_HORIZON)

        future_data = eval_df[(eval_df.index > future_start) & (eval_df.index <= future_end)]
        if future_data.empty:
            continue

        high = future_data['high']
        low = future_data['low']

        # 로그: 앞 3개 샘플 확인
        if i < 3:
            logger.info(f"🟡 샘플 {i}")
            logger.info(f"  Entry Time: {entry_time}, Entry Price: {entry_price:.2f}")
            logger.info(f"  TP Price: {tp_price:.2f}, SL Price: {sl_price:.2f}")
            logger.info(f"  Future Range: {future_data.index[0]} ~ {future_data.index[-1]} ({len(future_data)}개)")
            logger.info(f"  Max High: {high.max():.2f}, Min Low: {low.min():.2f}")

        tp_hit_idx = np.where(high >= tp_price)[0] if direction == "long" else np.where(low <= tp_price)[0]
        sl_hit_idx = np.where(low <= sl_price)[0] if direction == "long" else np.where(high >= sl_price)[0]

        if len(tp_hit_idx) > 0 and len(sl_hit_idx) > 0:
            if tp_hit_idx[0] <= sl_hit_idx[0]:
                tp_hits[i] = True
            else:
                sl_hits[i] = True
        elif len(tp_hit_idx) > 0:
            tp_hits[i] = True
        elif len(sl_hit_idx) > 0:
            sl_hits[i] = True

    return tp_hits, sl_hits

def main():
    logger.info(f"📁 파일 로드: {PKL_PATH}")
    raw = pd.read_pickle(PKL_PATH)
    df_entry = raw[ENTRY_TF].copy()
    df_eval = raw[EVAL_TF].copy()

    logger.info(f"✅ Entry 샘플 수: {len(df_entry)}, Eval 샘플 수: {len(df_eval)}")
    df_entry = df_entry.sort_index()
    df_eval = df_eval.sort_index()

    tp_hits, sl_hits = calculate_tp_sl_hits(df_entry, df_eval, direction=DIRECTION)

    tp_count = tp_hits.sum()
    sl_count = sl_hits.sum()
    neutral_count = len(df_entry) - tp_count - sl_count

    logger.info("📊 TP/SL 히트 분포:")
    logger.info(f"    TP 히트: {tp_count} ({tp_count/len(df_entry)*100:.2f}%)")
    logger.info(f"    SL 히트: {sl_count} ({sl_count/len(df_entry)*100:.2f}%)")
    logger.info(f"    Neutral: {neutral_count} ({neutral_count/len(df_entry)*100:.2f}%)")

if __name__ == "__main__":
    main()
