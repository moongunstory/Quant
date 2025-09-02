# live/realtime_ingest.py
from __future__ import annotations
import os, time
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple
from datetime import datetime, timezone, timedelta

# 학습 FE 재사용
from ai_binance.train.fe import compute_features_for_tf

# 학습時 규약
REF_SUFFIXES = ["_Open", "_High", "_Low", "_Close", "_Volume", "_FundingRate", "_FundingSettle"]
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PROC_DIR = os.path.join(DATA_DIR, "processed")

class ExchangeClient:
    def fetch_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        raise NotImplementedError

class FeatureBuilder:
    def __init__(self):
        self.feat_lists = {
            tf: self._load_json(os.path.join(PROC_DIR, f"fe_feature_list_{tf}.json"))
            for tf in ["5m", "15m", "1h", "4h"]
        }
        self.scalers = {
            tf: joblib.load(os.path.join(PROC_DIR, f"scaler_{tf}.joblib"))
            for tf in ["5m", "15m", "1h", "4h"]
        }

    @staticmethod
    def _load_json(path: str):
        import json
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def build_observation(self, eth_raw: Dict[str, pd.DataFrame], btc1h_raw: pd.DataFrame) -> Tuple[pd.Series, pd.Timestamp]:
        # 1) BTC 1h 피처
        btc_feat = compute_features_for_tf(btc1h_raw, "btc1h")
        # 2) ETH 각 TF 피처(lead-lag 포함)
        eth_feats = {tf: compute_features_for_tf(eth_raw[tf], tf, btc_df=btc_feat) for tf in ["5m", "15m", "1h", "4h"]}
        # 3) TF별 스케일링 후 5m 타임스탬프 기준 asof-join
        dfs = []
        for tf in ["5m", "15m", "1h", "4h"]:
            df = eth_feats[tf]
            feat_list = self.feat_lists[tf]
            obs_cols = [c for c in feat_list if not any(c.endswith(s) for s in REF_SUFFIXES)]
            X = df.reindex(columns=obs_cols).fillna(0.0).astype(float).to_numpy()
            Xs = self.scalers[tf].transform(X)
            df_scaled = pd.DataFrame(Xs, index=df.index, columns=obs_cols).add_prefix(f"f_{tf}_")
            dfs.append(df_scaled)

        obs = dfs[0]  # 5m 기준
        for d in dfs[1:]:
            obs = pd.merge_asof(obs.sort_index(), d.sort_index(), left_index=True, right_index=True, direction="backward")
        obs = obs.sort_index()
        ts = obs.index[-1]
        return obs.iloc[-1], ts

class LiveIngest:
    """닫힌 캔들만 사용. 5분봉 마감 감지 유틸 포함."""
    def __init__(self, ex: ExchangeClient, symbol_eth="ETHUSDT", symbol_btc="BTCUSDT"):
        self.ex = ex
        self.sym_eth = symbol_eth
        self.sym_btc = symbol_btc
        self.fe = FeatureBuilder()

    def fetch_raw_closed(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        lim = {"5m": 600, "15m": 300, "1h": 200, "4h": 200, "btc1h": 200}
        eth_raw = {tf: self.ex.fetch_klines(self.sym_eth, tf, lim[tf]) for tf in ["5m", "15m", "1h", "4h"]}
        btc1h_raw = self.ex.fetch_klines(self.sym_btc, "1h", lim["btc1h"])
        return eth_raw, btc1h_raw

    @staticmethod
    def _next_5m_close(after: datetime) -> datetime:
        """다음 5분 경계(UTC) 반환"""
        m = (after.minute // 5) * 5
        base = after.replace(second=0, microsecond=0)
        candidate = base.replace(minute=m) + timedelta(minutes=5)
        if candidate <= after:
            candidate += timedelta(minutes=5)
        return candidate

    def wait_next_5m_and_build(self, poll_sec: float = 2.0, grace_sec: float = 2.0) -> Tuple[pd.Series, pd.Timestamp]:
        """다음 5분봉 마감 직후(여유 grace_sec) 관측치 생성해서 반환"""
        now = datetime.now(timezone.utc)
        target = self._next_5m_close(now)
        while True:
            now = datetime.now(timezone.utc)
            if now >= target + timedelta(seconds=grace_sec):
                # 마감 직후: 닫힌 캔들 기반으로 관측치 생성
                eth_raw, btc1h_raw = self.fetch_raw_closed()
                obs, ts = self.fe.build_observation(eth_raw, btc1h_raw)
                # safety: 마지막 5m 타임스탬프가 target과 같거나 이전이어야 함(=닫힘)
                if ts <= target:
                    return obs, ts
            time.sleep(poll_sec)
