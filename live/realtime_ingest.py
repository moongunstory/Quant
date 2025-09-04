from __future__ import annotations
import os, time, json
from typing import Dict, Tuple, List, Any
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib

# 학습 FE 재사용 (REV-9.0 권장: leak-free / volume-consistent / BTC tolerance 일치)
from ai_binance.train.fe import compute_features_for_tf

# ===== 경로/규약 =====
DATA_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
PROC_DIR   = os.path.join(DATA_DIR, "processed")
MODEL_DIR  = os.path.join(DATA_DIR, "model")
OBS_PATH        = os.path.join(MODEL_DIR, "obs_cols.json")
OBS_BEST_PATH   = os.path.join(MODEL_DIR, "obs_cols_best.json")

# 학습시 관측에서 제외했던 원본(참조) 접미사 (정보용)
REF_SUFFIXES = ["_Open", "_High", "_Low", "_Close", "_Volume", "_FundingRate", "_FundingSettle"]
REF_CANON    = ["Open", "High", "Low", "Close", "Volume", "FundingRate", "FundingSettle"]

# ✅ 학습(fe.py)와 동일한 asof tolerance (옵션 A 기준)
TF_TOLERANCE = {
    "15m": pd.Timedelta("1H"),
    "1h":  pd.Timedelta("1H"),
    "4h":  pd.Timedelta("4H"),
}

# ===== 거래소 클라이언트 인터페이스 =====
class ExchangeClient:
    def fetch_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        raise NotImplementedError

# ===== 피처 빌더 =====
class FeatureBuilder:
    """
    옵션 A:
      - ETH 각 TF(5m/15m/1h/4h): 학습시 저장된 feature_list & scaler로 스케일 → f_{tf}_ 접두사
      - BTC 1h: '독립 관측'으로 합치지 않음. 대신 ETH 피처 생성 시 btc_df 인자로 넣어 lag/corr/beta 등 요약 신호만 사용
      - 최종 obs는 obs_cols_best.json(있으면) 또는 obs_cols.json 순서/개수와 정확히 일치
      - 결측 허용 안 함(health.ok=False)
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

    def _scale_eth_tf(self, eth_df: pd.DataFrame, tf: str, btc_feat_for_eth: pd.DataFrame | None) -> pd.DataFrame:
        """
        ETH 단일 TF 피처 생성 → 선택(feature_list) → scaler.transform → f_{tf}_ 접두사 부여
        (btc_feat_for_eth는 compute_features_for_tf(btc1h_raw,"btc1h") 결과)
        """
        feat_full = compute_features_for_tf(eth_df, tf, btc_df=btc_feat_for_eth).sort_index()
        # 학습 때 쓰던 리스트가 정답. 혹시 파일이 비었으면 REF 제외 규칙으로 백업
        keep = list(self.feat_lists[tf]) or [c for c in feat_full.columns if not any(c.endswith(s) for s in REF_SUFFIXES)]
        X = feat_full.reindex(columns=keep, fill_value=0.0).to_numpy(dtype=float)
        Xs = self.scalers[tf].transform(X)
        return pd.DataFrame(Xs, index=feat_full.index, columns=[f"f_{tf}_{c}" for c in keep])

    @staticmethod
    def _merge_asof_tol(left: pd.DataFrame, right: pd.DataFrame, tol: pd.Timedelta) -> pd.DataFrame:
        """Index 기반 asof-merge with tolerance (backward)."""
        li = left.reset_index().rename(columns={"index": "time"}) if left.index.name != "time" else left.reset_index()
        ri = right.reset_index().rename(columns={"index": "time"}) if right.index.name != "time" else right.reset_index()
        merged = pd.merge_asof(
            li.sort_values("time"),
            ri.sort_values("time"),
            on="time",
            direction="backward",
            tolerance=tol,
        )
        merged = merged.set_index("time")
        return merged

    def build_observation(self, eth_raw: Dict[str, pd.DataFrame], btc1h_raw: pd.DataFrame) -> Tuple[pd.Series, pd.Timestamp, Dict[str, Any]]:
        """
        반환: (obs_row: pd.Series, ts: pd.Timestamp(UTC), health: Dict)
        """
        # 0) BTC 1h 피처는 '한 번만' 계산 → ETH TF 생성 시 재사용 (옵션 A)
        btc_for_eth = compute_features_for_tf(btc1h_raw, "btc1h").sort_index()

        # 1) ETH 각 TF 스케일
        eth_scaled = {tf: self._scale_eth_tf(eth_raw[tf], tf, btc_for_eth) for tf in self.tf_list}

        # 2) 5m 기준 asof-merge (+tolerance) — 상위 TF만 병합 (BTC 독립 블록 없음)
        obs = eth_scaled["5m"].sort_index()
        for tf in ["15m", "1h", "4h"]:
            d = eth_scaled[tf].sort_index()
            obs = self._merge_asof_tol(obs, d, TF_TOLERANCE[tf])

        obs = obs.sort_index()

        # 3) 학습 시 obs_cols 순서/개수에 정확히 맞춤 (⚠ 결측은 채우지 않음)
        obs = obs.reindex(columns=self.obs_expected)

        ts = obs.index[-1]
        row = obs.iloc[-1]

        # ---- 헬스체크 (결측/정합/신선도) ----
        na_cols = row.index[row.isna()].tolist()
        na_ratio = float(np.mean(row.isna())) if len(row) else 1.0

        # 최근 타임스탬프 차이(신선도)
        def _age_sec(df_: pd.DataFrame) -> float:
            if df_.empty or ts is None:
                return float("inf")
            return float((ts - df_.index.max()).total_seconds())

        ages = {
            "age_15m_s": _age_sec(eth_scaled["15m"]),
            "age_1h_s":  _age_sec(eth_scaled["1h"]),
            "age_4h_s":  _age_sec(eth_scaled["4h"]),
            # BTC는 ETH 피처 생성에 실제 사용된 프레임으로 측정
            "age_btc1h_s": _age_sec(btc_for_eth),
        }

        health = {
            "ok": (na_ratio == 0.0),
            "na_ratio": na_ratio,
            "na_cols": na_cols[:20],  # 너무 길면 상위 20개만
            **ages,
            "dim": int(len(row)),
        }

        # float32 캐스팅은 ok일 때만
        if health["ok"]:
            row = row.astype(np.float32)

        return row, ts, health

# ===== 실시간 수집 루프 =====
class LiveIngest:
    """닫힌 캔들만 사용. 5분봉 마감 감지 + obs 헬스체크/결측-스킵."""
    def __init__(self, ex: ExchangeClient, symbol_eth="ETHUSDT", symbol_btc="BTCUSDT",
                 require_health_pass: bool = True, verbose: bool = True,
                 allow_sticky_last_good: bool = False, sticky_ttl_sec: int = 120):
        self.ex = ex
        self.sym_eth = symbol_eth
        self.sym_btc = symbol_btc
        self.fe = FeatureBuilder()
        self.require_health_pass = require_health_pass
        self.verbose = verbose
        self.allow_sticky_last_good = allow_sticky_last_good
        self.sticky_ttl_sec = sticky_ttl_sec
        self.last_health: Dict[str, Any] | None = None
        self._last_good: Tuple[pd.Series, pd.Timestamp] | None = None

    def fetch_raw_closed(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        # limit은 충분히 여유있게(지표/롤링/앵커드VWAP 등)
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
        """
        다음 5분봉 마감 직후(여유 grace_sec) 관측치 생성해서 반환.
        require_health_pass=True 이면 결측 발생 시 해당 스텝 스킵(다음 봉로 이월).
        allow_sticky_last_good=True 이면, 결측이어도 직전 정상 관측을 TTL 내에서 1회 재사용.
        """
        now = datetime.now(timezone.utc)
        target = self._next_5m_close(now)

        while True:
            now = datetime.now(timezone.utc)
            if now >= target + timedelta(seconds=grace_sec):
                eth_raw, btc1h_raw = self.fetch_raw_closed()
                row, ts, health = self.fe.build_observation(eth_raw, btc1h_raw)
                self.last_health = health

                if self.verbose:
                    print(f"[ingest] ts={ts} ok={health['ok']} na={health['na_ratio']:.4f} "
                          f"age(15m/1h/4h/B1h)={health['age_15m_s']:.0f}/{health['age_1h_s']:.0f}/"
                          f"{health['age_4h_s']:.0f}/{health['age_btc1h_s']:.0f}s "
                          f"dim={health['dim']}")

                if health["ok"]:
                    self._last_good = (row, ts)
                    return row, ts

                # 결측 처리 정책
                if not self.require_health_pass and len(row):
                    # 강제 통과(권장 X): 그대로 반환
                    return row.astype(np.float32, copy=False), ts

                if self.allow_sticky_last_good and self._last_good is not None:
                    lg_row, lg_ts = self._last_good
                    # TTL 내 재사용
                    if (ts - lg_ts).total_seconds() <= self.sticky_ttl_sec:
                        if self.verbose:
                            print(f"[ingest] using sticky last-good obs @ {lg_ts} (ttl {self.sticky_ttl_sec}s)")
                        return lg_row, ts  # ts는 현재 봉 기준으로 반환

                # 결측이면 이번 5분봉은 스킵하고 다음 봉을 기다림
                target = target + timedelta(minutes=5)

            time.sleep(poll_sec)
