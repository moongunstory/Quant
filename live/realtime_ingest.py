from __future__ import annotations
import os, time, json
from typing import Dict, Tuple, List
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib

# 학습 FE 재사용
from ai_binance.train.fe import compute_features_for_tf

# ===== 경로/규약 =====
DATA_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PROC_DIR   = os.path.join(DATA_DIR, "processed")
MODEL_DIR  = os.path.join(DATA_DIR, "model")
OBS_PATH        = os.path.join(MODEL_DIR, "obs_cols.json")
OBS_BEST_PATH   = os.path.join(MODEL_DIR, "obs_cols_best.json")

# 학습시 관측에서 제외했던 원본(참조) 접미사
REF_SUFFIXES = ["_Open", "_High", "_Low", "_Close", "_Volume", "_FundingRate", "_FundingSettle"]
# 원본(참조) 컬럼 이름들
REF_CANON = ["Open", "High", "Low", "Close", "Volume", "FundingRate", "FundingSettle"]

# ===== 거래소 클라이언트 인터페이스 =====
class ExchangeClient:
    def fetch_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        raise NotImplementedError

# ===== 피처 빌더 =====
class FeatureBuilder:
    """
    - ETH 각 TF(5m/15m/1h/4h): 학습 시 저장된 feature_list & scaler로 스케일 → f_{tf}_ 접두사
    - BTC 1h: compute_features_for_tf('btc1h') → REF 제외 → f_btc1h_ 접두사 → 5m 기준 asof-join
    - 마지막에 obs_cols_best.json(있으면) 또는 obs_cols.json 순서/개수에 정확히 맞춰 reindex(fill_value=0.0)
    """
    def __init__(self):
        # 1) 학습시 사용한 컬럼 순서(정답 벡터 형태) — best 우선
        meta_path = OBS_BEST_PATH if os.path.exists(OBS_BEST_PATH) else OBS_PATH
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"obs cols not found: {meta_path}\n"
                f"→ train.py 학습 시 obs_cols_best.json 또는 obs_cols.json을 생성하도록 해 주세요."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            self.obs_expected: List[str] = json.load(f)

        # 2) ETH TF별 feature list / scaler
        self.tf_list = ["5m", "15m", "1h", "4h"]
        self.feat_lists = {
            tf: self._load_json(os.path.join(PROC_DIR, f"fe_feature_list_{tf}.json"))
            for tf in self.tf_list
        }
        self.scalers = {
            tf: joblib.load(os.path.join(PROC_DIR, f"scaler_{tf}.joblib"))
            for tf in self.tf_list
        }

    @staticmethod
    def _load_json(path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _drop_ref_cols(df: pd.DataFrame) -> pd.DataFrame:
        drop = [c for c in REF_CANON if c in df.columns]
        return df.drop(columns=drop) if drop else df

    def _scale_eth_tf(self, eth_df: pd.DataFrame, tf: str, btc_feat_for_eth: pd.DataFrame | None) -> pd.DataFrame:
        """
        ETH 단일 TF 피처 생성 → 선택(feature_list) → scaler.transform → f_{tf}_ 접두사 부여
        """
        feat_full = compute_features_for_tf(eth_df, tf, btc_df=btc_feat_for_eth).sort_index()
        keep = list(self.feat_lists[tf]) or [c for c in feat_full.columns if not any(c.endswith(s) for s in REF_SUFFIXES)]
        X = feat_full.reindex(columns=keep, fill_value=0.0).to_numpy(dtype=float)
        Xs = self.scalers[tf].transform(X)
        return pd.DataFrame(Xs, index=feat_full.index, columns=[f"f_{tf}_{c}" for c in keep])

    def _prep_btc1h(self, btc1h_raw: pd.DataFrame) -> pd.DataFrame:
        """
        BTC 1h 단독 피처 생성 후 REF 제거 → f_btc1h_ 접두사 부여 (스케일 없음: 학습도 unscaled)
        """
        btc_feat = compute_features_for_tf(btc1h_raw, "btc1h").sort_index()
        btc_feat = self._drop_ref_cols(btc_feat)
        return btc_feat.add_prefix("f_btc1h_")

    def build_observation(self, eth_raw: Dict[str, pd.DataFrame], btc1h_raw: pd.DataFrame) -> Tuple[pd.Series, pd.Timestamp]:
        # 1) BTC 1h 피처 (나중에 5m 기준으로 asof-join)
        btc_pref = self._prep_btc1h(btc1h_raw)
        # 2) ETH 각 TF (BTC 리드/래그/상관 계산용 비접두사 BTC 피처)
        btc_for_eth = compute_features_for_tf(btc1h_raw, "btc1h")
        eth_scaled = {tf: self._scale_eth_tf(eth_raw[tf], tf, btc_for_eth) for tf in self.tf_list}

        # 3) 5m 기준 asof-merge + BTC 1h asof-join
        obs = eth_scaled["5m"].sort_index()
        for tf in ["15m", "1h", "4h"]:
            d = eth_scaled[tf].sort_index()
            obs = pd.merge_asof(obs, d, left_index=True, right_index=True, direction="backward")
        obs = pd.merge_asof(obs.sort_index(), btc_pref.sort_index(),
                            left_index=True, right_index=True, direction="backward").sort_index()

        # 4) 학습 시 obs_cols 순서/개수에 정확히 맞춤
        obs = obs.reindex(columns=self.obs_expected, fill_value=0.0)

        ts = obs.index[-1]
        row = obs.iloc[-1].astype(np.float32)
        return row, ts

# ===== 실시간 수집 루프 =====
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
        """다음 5분 경계(UTC)"""
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
