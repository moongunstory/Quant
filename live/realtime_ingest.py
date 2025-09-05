# ai_binance/live/realtime_ingest.py
from __future__ import annotations
import os
import time
import json
from typing import Dict, Tuple, List, Any, Optional
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import joblib

# 학습 FE 재사용 (REV-9.0 권장: leak-free / volume-consistent)
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

# ===== 거래소 클라이언트 인터페이스 =====
class ExchangeClient:
    def fetch_klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """
        반환: DatetimeIndex(UTC, name='time') + OHLCV(및 부가) 칼럼. '닫힌 캔들'만 포함해야 함.
        """
        raise NotImplementedError

# ===== 피처 빌더 =====
class FeatureBuilder:
    """
    캘린더 정렬(Calendar-aligned) 방식:
      - ETH 각 TF(5m/15m/1h/4h): 학습시 저장된 feature_list & scaler로 스케일 → f_{tf}_ 접두사
      - 상위 TF는 5m 최신 시각 기준 '직전 완성 캔들(open_time)'의 한 행만 정확히 인덱싱 (tolerance 미사용)
      - BTC 1h: 독립 블록을 관측에 합치지 않음. ETH 피처 생성 시 btc_df 인자로 넣어 lag/corr/beta 요약 신호만 사용
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

    def _scale_eth_tf(self, eth_df: pd.DataFrame, tf: str, btc_feat_for_eth: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        ETH 단일 TF 피처 생성 → 선택(feature_list) → scaler.transform → f_{tf}_ 접두사 부여
        (btc_feat_for_eth는 compute_features_for_tf(btc1h_raw, "btc1h") 결과)
        """
        feat_full = compute_features_for_tf(eth_df, tf, btc_df=btc_feat_for_eth).sort_index()
        keep = list(self.feat_lists[tf]) or [c for c in feat_full.columns if not any(c.endswith(s) for s in REF_SUFFIXES)]
        X = feat_full.reindex(columns=keep, fill_value=0.0).to_numpy(dtype=float)
        Xs = self.scalers[tf].transform(X)
        return pd.DataFrame(Xs, index=feat_full.index, columns=[f"f_{tf}_{c}" for c in keep])

    @staticmethod
    def _last_completed_open(ts: pd.Timestamp, tf: str) -> pd.Timestamp:
        """
        5m 최신 시각(ts) 기준, 해당 TF의 '직전 완성 캔들'의 open_time을 계산.
        인덱스는 UTC DatetimeIndex 가정.
        """
        if tf == "15m":
            return ts.floor("15min") - pd.Timedelta(minutes=15)
        if tf == "1h":
            return ts.floor("1h") - pd.Timedelta(hours=1)
        if tf == "4h":
            return ts.floor("4h") - pd.Timedelta(hours=4)
        raise ValueError(f"unsupported timeframe for alignment: {tf}")

    def _ages_calendar(self, ts: pd.Timestamp, aligned_opens: Dict[str, pd.Timestamp], btc_df: pd.DataFrame) -> Dict[str, float]:
        out = {}
        for tf, t_open in aligned_opens.items():
            out[f"age_{tf}_s"] = float((ts - t_open).total_seconds())
        # BTC 1h는 실제 사용 인덱스(직전 1h 완성 open)가 있으면 그것 기준, 없으면 마지막 인덱스 기준
        btc_target = ts.floor("1h") - pd.Timedelta(hours=1)
        if len(btc_df) and (btc_target in btc_df.index):
            out["age_btc1h_s"] = float((ts - btc_target).total_seconds())
        else:
            last = btc_df.index.max() if len(btc_df) else ts
            out["age_btc1h_s"] = float((ts - last).total_seconds())
        return out

    def build_observation(self, eth_raw: Dict[str, pd.DataFrame], btc1h_raw: pd.DataFrame) -> Tuple[pd.Series, pd.Timestamp, Dict[str, Any]]:
        """
        반환: (obs_row: pd.Series, ts: pd.Timestamp(UTC), health: Dict)
        - 상위 TF는 tolerance 없이 캘린더 경계로 정확히 1행을 집어온다.
        """
        # 0) BTC 1h 피처는 한 번만 계산 → ETH TF 생성 시 재사용
        btc_for_eth = compute_features_for_tf(btc1h_raw, "btc1h").sort_index()

        # 1) ETH 각 TF 스케일
        eth_scaled = {tf: self._scale_eth_tf(eth_raw[tf], tf, btc_for_eth) for tf in self.tf_list}

        # 2) 5m 최신 시각 및 최신 행
        obs5 = eth_scaled["5m"].sort_index()
        if obs5.empty:
            raise RuntimeError("5m frame is empty; cannot build observation.")
        ts = obs5.index[-1]
        row5 = obs5.iloc[-1]

        # 3) 상위 TF의 '직전 완성 캔들 open_time'을 계산해 정확히 해당 시점의 행만 선택
        aligned_opens = {
            "15m": self._last_completed_open(ts, "15m"),
            "1h":  self._last_completed_open(ts, "1h"),
            "4h":  self._last_completed_open(ts, "4h"),
        }

        rows = [row5]
        for tf in ["15m", "1h", "4h"]:
            df_tf = eth_scaled[tf]
            tgt = aligned_opens[tf]
            if tgt in df_tf.index:
                rows.append(df_tf.loc[tgt])
            else:
                # 해당 시각의 상위TF 행이 없으면, 그 TF 블록 전체 NaN으로 두어 헬스 fail → sticky/skip으로 처리
                rows.append(pd.Series(index=df_tf.columns, dtype=float))

        # 4) 가로로 이어 붙여 단일 관측 벡터 구성 후, 학습 시 obs_cols 순서/개수에 정확히 맞춤
        row = pd.concat(rows)
        row = row.reindex(self.obs_expected)

        # ---- 헬스체크 ----
        na_ratio = float(np.mean(row.isna())) if len(row) else 1.0
        na_cols = row.index[row.isna()].tolist()
        ages = self._ages_calendar(ts, aligned_opens, btc_for_eth)

        health = {
            "ok": (na_ratio == 0.0),
            "na_ratio": na_ratio,
            "na_cols": na_cols[:20],
            "dim": int(len(row)),
            **ages,
        }

        if health["ok"]:
            row = row.astype(np.float32)

        return row, ts, health

# ===== 실시간 수집 루프 =====
class LiveIngest:
    """
    닫힌 캔들만 사용.
    - 최초: FE가 요구하는 '최대 롤링 창 + α' 만큼만 각 TF를 로드(가볍게)
    - 이후: TF별 주기에 따라 '증분'으로만 가져와 캐시에 append
      * 5m:   매 5분
      * 15m:  매 15분
      * 1h:   매 정시
      * 4h:   매 4시간 정시
      * BTC1h: 매 정시
    - 매 5분봉 마감 직후 관측 생성 + 헬스체크
      * require_health_pass=True: 결측이면 스킵
      * allow_sticky_last_good=True: TTL 내 직전 정상 관측 재사용
    """
    def __init__(
        self,
        ex: ExchangeClient,
        symbol_eth: str = "ETHUSDT",
        symbol_btc: str = "BTCUSDT",
        require_health_pass: bool = True,
        verbose: bool = True,
        allow_sticky_last_good: bool = False,
        sticky_ttl_sec: int = 120,
    ):
        self.ex = ex
        self.sym_eth = symbol_eth
        self.sym_btc = symbol_btc
        self.fe = FeatureBuilder()
        self.require_health_pass = require_health_pass
        self.verbose = verbose
        self.allow_sticky_last_good = allow_sticky_last_good
        self.sticky_ttl_sec = int(sticky_ttl_sec)
        self.last_health: Dict[str, Any] | None = None
        self._last_good: Tuple[pd.Series, pd.Timestamp] | None = None

        # 🔹 TF별 캐시 (최초 1회만 가볍게 로드)
        self._eth_cache: Dict[str, pd.DataFrame] = {"5m": pd.DataFrame(), "15m": pd.DataFrame(), "1h": pd.DataFrame(), "4h": pd.DataFrame()}
        self._btc_cache: Dict[str, pd.DataFrame] = {"1h": pd.DataFrame()}

        self._initial_load()  # ✅ 최초에만 필요한 최소 개수 로드

    # ===== 최초 로드: FE 요구치 기반 최소 개수만 =====
    def _initial_load(self):
        # FE 최대 창 + 알파(10~20%) — 현재 fe.py 기준
        WARMUP = {"5m": 380, "15m": 120, "1h": 96, "4h": 220, "btc1h": 96}
        self._eth_cache["5m"]  = self.ex.fetch_klines(self.sym_eth, "5m",  WARMUP["5m"])
        self._eth_cache["15m"] = self.ex.fetch_klines(self.sym_eth, "15m", WARMUP["15m"])
        self._eth_cache["1h"]  = self.ex.fetch_klines(self.sym_eth, "1h",  WARMUP["1h"])
        self._eth_cache["4h"]  = self.ex.fetch_klines(self.sym_eth, "4h",  WARMUP["4h"])
        self._btc_cache["1h"]  = self.ex.fetch_klines(self.sym_btc, "1h",  WARMUP["btc1h"])

        # 인덱스 정리
        for k in list(self._eth_cache.keys()):
            self._eth_cache[k] = self._eth_cache[k].sort_index()
        self._btc_cache["1h"] = self._btc_cache["1h"].sort_index()

    @staticmethod
    def _append_new(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if old is None or old.empty:
            return new.sort_index()
        if new is None or new.empty:
            return old
        cat = pd.concat([old, new]).sort_index()
        # 인덱스 기준 최신값 유지
        cat = cat[~cat.index.duplicated(keep="last")]
        return cat

    def _update_caches_incremental(self):
        """
        현재 시각 기준 TF별 주기에 맞게 '소량(1~3개)'만 증분 수신하여 캐시에 append.
        """
        now = datetime.now(timezone.utc)

        # 5m: 매 5분
        if now.minute % 5 == 0:
            df = self.ex.fetch_klines(self.sym_eth, "5m", 3)
            self._eth_cache["5m"] = self._append_new(self._eth_cache["5m"], df)

        # 15m: 매 15분
        if now.minute % 15 == 0:
            df = self.ex.fetch_klines(self.sym_eth, "15m", 3)
            self._eth_cache["15m"] = self._append_new(self._eth_cache["15m"], df)

        # 1h & BTC1h: 매 정시
        if now.minute == 0:
            df1 = self.ex.fetch_klines(self.sym_eth, "1h", 3)
            self._eth_cache["1h"] = self._append_new(self._eth_cache["1h"], df1)
            dfb = self.ex.fetch_klines(self.sym_btc, "1h", 3)
            self._btc_cache["1h"] = self._append_new(self._btc_cache["1h"], dfb)

        # 4h: 매 4시간 정시
        if now.minute == 0 and (now.hour % 4 == 0):
            df4 = self.ex.fetch_klines(self.sym_eth, "4h", 3)
            self._eth_cache["4h"] = self._append_new(self._eth_cache["4h"], df4)

        self._inject_live_funding()

    def _inject_live_funding(self):
        try:
            # python-binance 사용 가정 (실제 클라이언트 접근 경로에 맞춰 조정)
            info = self.ex.client.futures_premium_index(symbol=self.sym_eth)
            rate = float(info.get("lastFundingRate", 0.0))
            next_ms = int(info.get("nextFundingTime", 0))
            if not self._eth_cache["5m"].empty:
                self._eth_cache["5m"]["FundingRate"] = rate
                if next_ms:
                    next_ts = pd.to_datetime(next_ms, unit="ms", utc=True)
                    # 정산 바 표시(존재하면 1)
                    if next_ts in self._eth_cache["5m"].index:
                        self._eth_cache["5m"]["FundingSettle"] = (self._eth_cache["5m"].index == next_ts).astype("int8")
        except Exception:
            pass

    # (호환 유지) 현재 캐시를 그대로 반환
    def fetch_raw_closed(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        return (self._eth_cache.copy(), self._btc_cache["1h"].copy())

    @staticmethod
    def _next_5m_close(after: datetime) -> datetime:
        """다음 5분 경계(UTC)"""
        base = after.replace(second=0, microsecond=0)
        m = (after.minute // 5) * 5
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
                # 🔹 TF별 증분 업데이트만 수행
                self._update_caches_incremental()
                eth_raw, btc1h_raw = self.fetch_raw_closed()

                row, ts, health = self.fe.build_observation(eth_raw, btc1h_raw)
                self.last_health = health

                if self.verbose:
                    print(
                        f"[ingest] ts={ts} ok={health['ok']} na={health['na_ratio']:.4f} "
                        f"age(15m/1h/4h/B1h)={int(health.get('age_15m_s',-1))}/"
                        f"{int(health.get('age_1h_s',-1))}/{int(health.get('age_4h_s',-1))}/"
                        f"{int(health.get('age_btc1h_s',-1))}s dim={health.get('dim',-1)}"
                    )

                if health["ok"]:
                    self._last_good = (row, ts)
                    return row, ts

                # 결측 처리 정책
                if not self.require_health_pass and len(row):
                    return row.astype(np.float32, copy=False), ts

                if self.allow_sticky_last_good and self._last_good is not None:
                    lg_row, lg_ts = self._last_good
                    if (ts - lg_ts).total_seconds() <= self.sticky_ttl_sec:
                        if self.verbose:
                            print(f"[ingest] using sticky last-good obs @ {lg_ts} (ttl {self.sticky_ttl_sec}s)")
                        return lg_row, ts  # ts는 현재 봉 기준으로 반환

                # 결측이면 이번 5분봉은 스킵하고 다음 봉을 기다림
                target = target + timedelta(minutes=5)

            time.sleep(poll_sec)
