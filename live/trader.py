# ai_binance/live/trader.py
from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import gymnasium as gym
import time
from gymnasium import spaces
import importlib.util
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET

# ===== 공통 인터페이스 =====
class ExchangeClient:
    def fetch_klines(self, symbol: str, interval: str, limit: int): ...

class ExecClient:
    def position(self) -> int: ...
    def market_long(self, qty: float): ...
    def market_short(self, qty: float): ...
    def market_close(self): ...

# ===== Gym(dummy env) 유지 =====
class _ObsEnv(gym.Env):
    metadata = {}
    def __init__(self, obs_dim: int):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if hasattr(super(), "reset"):
            super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}
    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated, truncated = True, False
        return obs, reward, terminated, truncated, {}

# ===== cloudpickle가 참조하는 'policy' 모듈 강제 로드 =====
def _import_policy_for_cloudpickle() -> None:
    """
    cloudpickle로 저장된 모델을 로드하기 위해 'policy' 모듈을 sys.modules에 등록합니다.
    학습 시점의 `from policy import ...` 구문을 재현하기 위함입니다.
    """
    # 이미 로드되었다면 아무것도 하지 않음
    if "policy" in sys.modules:
        return

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, "..", "train", "reinforce", "policy.py")),
        os.path.abspath(os.path.join(here, "..", "reinforce", "policy.py")),
        os.path.abspath(os.path.join(here, "policy.py")),
    ]
    for p in candidates:
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("policy", p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["policy"] = mod
                spec.loader.exec_module(mod)
                return
    raise ModuleNotFoundError(
        "Unable to import 'policy'. 학습에 사용한 policy.py를 다음 중 한 경로에 두세요:\n"
        + "\n".join(f" - {c}" for c in candidates)
    )

# ===== Binance 라이브 어댑터 =====
_INTERVAL = {
    "5m": Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "1h": Client.KLINE_INTERVAL_1HOUR,
    "4h": Client.KLINE_INTERVAL_4HOUR,
}

class BinanceExchange(ExchangeClient, ExecClient):
    def __init__(self,
                 symbol_eth: str = "ETHUSDT",
                 symbol_btc: str = "BTCUSDT",
                 api_key: str = "",
                 api_secret: str = "",
                 use_testnet: bool = False,
                 recv_window: int = 5_000):
        if not api_key or not api_secret:
            raise RuntimeError("BinanceExchange: api_key/api_secret must be provided (no env read).")

        # --- Binance 클라이언트 ---
        self.client = Client(api_key=api_key, api_secret=api_secret)
        if use_testnet:
            self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"

        self.recv_window = recv_window
        self.symbol_eth = symbol_eth
        self.symbol_btc = symbol_btc

        # --- 서버 시간 동기화 (−1021 방지) ---
        try:
            srv_ms = int(self.client.get_server_time()["serverTime"])
            loc_ms = int(time.time() * 1000)
            offset = srv_ms - loc_ms
            # python-binance는 `timestamp_offset`를 사용합니다.
            self.client.timestamp_offset = offset
        except Exception:
            # 실패해도 치명적이지 않으니 그냥 진행
            self.client.timestamp_offset = 0

        # --- 심볼 필터 로드(수량 스텝/최소수량) ---
        self._filters: Dict[str, Dict[str, float]] = {}
        self._load_filters()

    def set_leverage(self, lev: int):
        try:
            self.client.futures_change_leverage(
                symbol=self.symbol_eth, leverage=int(lev), recvWindow=self.recv_window
            )
        except Exception:
            pass

    # --- 시세 (닫힌 캔들만) ---
    def fetch_klines(self, symbol: str, interval: str, limit: int):
        bi = _INTERVAL[interval]
        raw = self.client.futures_klines(symbol=symbol, interval=bi, limit=limit)
        import pandas as pd, time as _t
        cols = ["open_time","Open","High","Low","Close","Volume","close_time",
                "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]
        df = pd.DataFrame(raw, columns=cols)
        now_ms = int(_t.time() * 1000)
        df = df[df["close_time"] <= now_ms - 1000]  # 미확정 봉 제거
        for c in ["Open","High","Low","Close","Volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index(idx).drop(columns=["open_time","close_time","ignore"])
        df.index.name = "time"
        df["FundingRate"] = 0.0
        return df

    # --- 주문/포지션 ---
    def _load_filters(self):
        info = self.client.futures_exchange_info()
        for s in info["symbols"]:
            if s.get("contractType") != "PERPETUAL":
                continue
            sym = s["symbol"]
            lot = next(f for f in s["filters"] if f["filterType"] == "LOT_SIZE")
            self._filters[sym] = {"stepSize": float(lot["stepSize"]), "minQty": float(lot["minQty"])}

    def _round_qty(self, symbol: str, qty: float) -> float:
        f = self._filters.get(symbol)
        if not f:
            return float(qty)
        step, minq = f["stepSize"], f["minQty"]
        q = np.floor(float(qty) / step) * step
        if q < minq:
            q = 0.0
        return float(np.round(q, 8))

    def position(self) -> int:
        arr = self.client.futures_position_information(symbol=self.symbol_eth, recvWindow=self.recv_window)
        if not arr:
            return 0
        amt = float(arr[0]["positionAmt"])
        return 1 if amt > 0 else (-1 if amt < 0 else 0)

    def _abs_position_qty(self) -> float:
        arr = self.client.futures_position_information(symbol=self.symbol_eth, recvWindow=self.recv_window)
        if not arr:
            return 0.0
        return abs(float(arr[0]["positionAmt"]))

    def market_long(self, qty: float):
        q = self._round_qty(self.symbol_eth, qty)
        if q <= 0:
            return
        self.client.futures_create_order(
            symbol=self.symbol_eth, side=SIDE_BUY, type=ORDER_TYPE_MARKET,
            quantity=q, recvWindow=self.recv_window
        )

    def market_short(self, qty: float):
        q = self._round_qty(self.symbol_eth, qty)
        if q <= 0:
            return
        self.client.futures_create_order(
            symbol=self.symbol_eth, side=SIDE_SELL, type=ORDER_TYPE_MARKET,
            quantity=q, recvWindow=self.recv_window
        )

    def market_close(self):
        pos = self.position()
        if pos == 0:
            return
        q = self._abs_position_qty()
        if q <= 0:
            return
        side = SIDE_SELL if pos > 0 else SIDE_BUY
        self.client.futures_create_order(
            symbol=self.symbol_eth, side=side, type=ORDER_TYPE_MARKET,
            quantity=self._round_qty(self.symbol_eth, q),
            reduceOnly=True, recvWindow=self.recv_window
        )

# ===== Paper 모드 =====
class PublicBinanceData(ExchangeClient):
    def __init__(self, testnet: bool = False):
        self.client = Client(None, None)
        if testnet:
            self.client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
    def fetch_klines(self, symbol: str, interval: str, limit: int):
        bi = _INTERVAL[interval]
        raw = self.client.futures_klines(symbol=symbol, interval=bi, limit=limit)
        import pandas as pd, time as _t
        cols = ["open_time","Open","High","Low","Close","Volume","close_time",
                "quote_asset_volume","number_of_trades","taker_buy_base","taker_buy_quote","ignore"]
        df = pd.DataFrame(raw, columns=cols)
        now_ms = int(_t.time() * 1000)
        df = df[df["close_time"] <= now_ms - 1000]
        for c in ["Open","High","Low","Close","Volume","quote_asset_volume","taker_buy_base","taker_buy_quote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index(idx).drop(columns=["open_time","close_time","ignore"])
        df.index.name = "time"
        df["FundingRate"] = 0.0
        return df

class PaperBroker(ExecClient):
    def __init__(self): self._pos = 0
    def position(self) -> int: return self._pos
    def market_long(self, qty: float):  self._pos = 1
    def market_short(self, qty: float): self._pos = -1
    def market_close(self):             self._pos = 0

# ===== Step 결과 컨테이너 =====
@dataclass
class StepResult:
    logs: List[Dict[str, Any]]
    report_snapshot: Dict[str, Any]
    summary: Dict[str, Any]

# ===== 공통 트레이더 베이스 =====
class BaseTrader:
    def __init__(self,
                 exec_client: ExecClient,
                 data_client: ExchangeClient,
                 model_path: Optional[str],
                 vec_path: Optional[str],
                 norm_reward_at_train: bool,
                 symbol_eth: str,
                 leverage: float,
                 risk_fraction: float,
                 init_equity: float):

        self.exec = exec_client
        self.data = data_client
        self.symbol_eth = symbol_eth
        self.leverage = float(leverage)
        self.risk_fraction = float(risk_fraction)
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "model"))
        self.model_path = model_path or os.path.join(base, "best_model.zip")
        self.vec_path   = vec_path   or os.path.join(base, "unified_vecnorm.pkl")

        # 1) cloudpickle이 'policy' 찾도록
        _import_policy_for_cloudpickle()

        # 2) env 없이 모델 먼저 로드 → obs_dim 취득
        model = MaskablePPO.load(self.model_path, env=None)
        # SB3/Contrib는 policy에 obs_space 보관
        obs_space = getattr(model.policy, "observation_space", None) or getattr(model, "observation_space", None)
        if obs_space is None:
            raise RuntimeError("Loaded model has no observation_space; re-check saved model.")
        obs_dim = int(np.prod(obs_space.shape))

        # 3) 같은 차원의 더미 VecEnv 만들고 VecNormalize 통계 로드
        venv = DummyVecEnv([lambda: _ObsEnv(obs_dim)])
        venv = VecNormalize(venv, training=False, norm_obs=True, norm_reward=norm_reward_at_train)
        venv = VecNormalize.load(self.vec_path, venv)
        venv.training = False
        venv.norm_reward = False

        # 4) 모델에 VecNormalize env 장착
        model.set_env(venv)

        self.venv = venv
        self.model = model
        self.initial_equity = float(init_equity)

    @staticmethod
    def _mask_from_pos(pos: int) -> np.ndarray:
        # 액션: 0 WAIT, 1 LONG, 2 SHORT, 3 CLOSE
        m = np.ones(4, dtype=np.int8)
        if pos == 0: m[3] = 0
        if pos > 0:  m[1] = 0
        if pos < 0:  m[2] = 0
        return m

    def _predict_action(self, obs_vec: np.ndarray) -> Dict[str, Any]:
        """ 모델을 사용해 액션을 예측하고, 부가 정보(가치, 로짓)를 함께 반환합니다. """
        o = np.asarray(obs_vec, dtype=np.float32).reshape(1, -1)
        normalized_obs = self.venv.normalize_obs(o)
        mask = self._mask_from_pos(self.exec.position())

        # SB3 모델에서 상세 정보를 얻기 위해 policy 직접 사용
        import torch
        obs_tensor = torch.as_tensor(normalized_obs).to(self.model.policy.device)
        with torch.no_grad():
            # get_distribution()으로 액션 확률 분포(로짓) 획득
            dist = self.model.policy.get_distribution(obs_tensor)
            logits = dist.distribution.logits.cpu().numpy().flatten()
            # predict_values()로 상태 가치(기대 보상) 획득
            value = self.model.policy.predict_values(obs_tensor).cpu().numpy().flatten()[0]

        # 최종 액션은 마스크를 적용하여 결정
        action, _ = self.model.predict(normalized_obs, deterministic=True, action_masks=mask)

        return {
            "action": int(action),
            "value": float(value),
            "logits": logits.tolist(),
        }

    def _format_prediction_string(self, prediction: Dict[str, Any], before_pos: int) -> str:
        action = prediction["action"]
        logits = prediction["logits"]
        
        labels = ['W', 'L', 'S', 'C']
        mask = self._mask_from_pos(before_pos) 

        pred_parts = []
        for i, logit in enumerate(logits):
            label = labels[i]
            
            if mask[i] == 0:
                part = f"{label}:X"
            else:
                part = f"{label}:{logit:.2f}"

            if i == action:
                part = f"({part})"
            
            pred_parts.append(part)

        return f"[{', '.join(pred_parts)}]"

    def _last_close(self) -> float:
        df = self.data.fetch_klines(self.symbol_eth, "5m", limit=2)
        return float(df["Close"].iloc[-1])

    # 포지션 사이징: 전액 * 레버리지 / 가격 * 비율
    def _size_full(self, equity: float, price: float) -> float:
        # 가용 증거금을 100% 쓰지 않도록 안전 캡 + 초기증거금(IM) 검증
        SAFETY = 0.95   # 가용 증거금의 95%만 사용
        FEEBUF = 1.005  # 수수료/슬리피지 여유
        px = max(float(price), 1e-9)
        eq = max(float(equity), 0.0)
        # 1차 제안 수량
        qty = (eq * self.leverage * self.risk_fraction * SAFETY) / px
        if qty <= 0:
            return 0.0
        # 초기증거금 추산: notional/lev (보수적 버퍼 포함)
        required_im = (qty * px / max(self.leverage, 1e-9)) * FEEBUF
        if required_im > eq:
            scale = eq / required_im  # 살짝만 초과해도 자동 축소
            qty *= max(min(scale, 1.0), 0.0)
        return float(qty)

    def step(self, obs_vec: np.ndarray, ts: datetime) -> StepResult:
        raise NotImplementedError

# ===== 라이브 트레이더 =====
class LiveTrader(BaseTrader):
    def __init__(self, exec_client: BinanceExchange,
                 model_path: Optional[str] = None, vec_path: Optional[str] = None,
                 norm_reward_at_train: bool = False, symbol_eth: str = "ETHUSDT",
                 leverage: float = 5.0, risk_fraction: float = 1.0,
                 prob_threshold_entry: float = 0.65,
                 prob_threshold_switch: float = 0.67):
        acc = exec_client.client.futures_account()
        total_eq = float(acc.get("totalMarginBalance", "0"))
        super().__init__(exec_client, exec_client, model_path, vec_path, norm_reward_at_train,
                         symbol_eth, leverage, risk_fraction, init_equity=total_eq)
        self.prob_threshold_entry = prob_threshold_entry
        self.prob_threshold_switch = prob_threshold_switch

    def step(self, obs_vec: np.ndarray, ts: datetime) -> StepResult:
        before_pos = self.exec.position()
        prediction = self._predict_action(obs_vec)
        action = prediction["action"]
        
        logs: List[Dict[str, Any]] = []
        filter_info = None
        final_action = action

        # 상황별 임계값 필터
        if action in (1, 2): # 롱 또는 숏 진입/전환 시도
            import torch
            logits = torch.tensor(prediction["logits"])
            probs = torch.nn.functional.softmax(logits, dim=0)
            action_prob = probs[action].item()

            context = ""
            threshold = 0.0

            if before_pos == 0: # 1. 신규 진입
                context, threshold = "ENTRY", self.prob_threshold_entry
            else: # 2. 포지션 전환
                context, threshold = "SWITCH", self.prob_threshold_switch

            if action_prob < threshold:
                final_action = 0 # WAIT으로 강제 변경
                filter_info = f"OVERRIDE({context} prob {action_prob:.2f} < {threshold})"
            else:
                filter_info = f"PASS({context} prob {action_prob:.2f} >= {threshold})"
            
            logs.append({"timestamp": ts.isoformat(), "type": "FILTER", "info": filter_info})

        # PREDICT 로그는 필터링 결정 후에 추가 (실제 수행될 action을 기록하기 위함)
        prediction["action"] = final_action # 필터링이 적용된 최종 액션으로 업데이트
        logs.insert(0, {
            "timestamp": ts.isoformat(),
            "type": "PREDICT",
            "value": f"{prediction['value']:.4f}",
            "logits[W,L,S,C]": [f"{l:.3f}" for l in prediction['logits']],
            "masked_action": final_action,
        })

        price = self._last_close()

        def _account_equity() -> float:
            a = self.exec.client.futures_account()
            return float(a.get("availableBalance", "0"))

        if final_action == 3:
            self.exec.market_close()
            logs.append({"timestamp": ts.isoformat(), "type": "CLOSE", "position": "FLAT", "price": f"{price:.2f}"})
        elif final_action in (1, 2):
            if (final_action == 1 and before_pos < 0) or (final_action == 2 and before_pos > 0):
                self.exec.market_close()
                logs.append({"timestamp": ts.isoformat(), "type": "CLOSE", "position": "FLAT", "price": f"{price:.2f}"})
            equity = _account_equity()
            qty = self._size_full(equity, price)
            if final_action == 1:
                self.exec.market_long(qty)
                logs.append({"timestamp": ts.isoformat(), "type": "ENTRY_LONG", "position": "LONG", "price": f"{price:.2f}"})
            else:
                self.exec.market_short(qty)
                logs.append({"timestamp": ts.isoformat(), "type": "ENTRY_SHORT", "position": "SHORT", "price": f"{price:.2f}"})
        
        acc = self.exec.client.futures_account()
        total_eq = float(acc.get("totalMarginBalance", "0"))
        unrl    = float(acc.get("totalUnrealizedProfit", "0"))
        
        after_pos = self.exec.position()
        pos_map = {0: "FLAT", 1: "LONG", -1: "SHORT"}
        pos_str = pos_map[after_pos]
        if before_pos != after_pos:
            pos_str = f"{pos_map[before_pos]}->{pos_map[after_pos]}"

        report = {
            "total_equity": total_eq,
            "unrealized_pnl_amount": unrl,
            "unrealized_pnl_percent": (unrl / total_eq * 100.0) if total_eq > 0 else 0.0,
            "position": pos_map[after_pos],
        }
        
        prediction_str = self._format_prediction_string(prediction, before_pos)
        summary = {
            "pos": pos_str, 
            "price": f"{price:.2f}", 
            "equity": f"{total_eq:.2f}", 
            "value": f"{prediction['value']:.4f}",
            "prediction": prediction_str
        }
        if filter_info:
            summary["filter"] = filter_info

        return StepResult(logs=logs, report_snapshot=report, summary=summary)

# ===== 페이퍼 트레이더 =====
class PaperTrader(BaseTrader):
    def __init__(self, data_client: PublicBinanceData, exec_client: PaperBroker,
                 model_path: Optional[str] = None, vec_path: Optional[str] = None,
                 norm_reward_at_train: bool = False, symbol_eth: str = "ETHUSDT",
                 leverage: float = 5.0, risk_fraction: float = 1.0, init_equity: float = 10_000.0,
                 prob_threshold_entry: float = 0.65,
                 prob_threshold_switch: float = 0.67):
        super().__init__(exec_client, data_client, model_path, vec_path, norm_reward_at_train,
                         symbol_eth, leverage, risk_fraction, init_equity)
        self.realized = 0.0
        self.entry_side = 0
        self.entry_price: Optional[float] = None
        self.entry_qty: float = 0.0
        self.prob_threshold_entry = prob_threshold_entry
        self.prob_threshold_switch = prob_threshold_switch

    def _unrealized(self, price: float) -> float:
        if self.entry_side == 0 or self.entry_price is None:
            return 0.0
        sign = 1.0 if self.entry_side > 0 else -1.0
        return (price - self.entry_price) * sign * self.entry_qty

    def step(self, obs_vec: np.ndarray, ts: datetime) -> StepResult:
        before_pos = self.exec.position()
        prediction = self._predict_action(obs_vec)
        action = prediction["action"]

        logs: List[Dict[str, Any]] = []
        filter_info = None
        final_action = action

        # 상황별 임계값 필터
        if action in (1, 2): # 롱 또는 숏 진입/전환 시도
            import torch
            logits = torch.tensor(prediction["logits"])
            probs = torch.nn.functional.softmax(logits, dim=0)
            action_prob = probs[action].item()

            context = ""
            threshold = 0.0

            if before_pos == 0: # 1. 신규 진입
                context, threshold = "ENTRY", self.prob_threshold_entry
            else: # 2. 포지션 전환
                context, threshold = "SWITCH", self.prob_threshold_switch

            if action_prob < threshold:
                final_action = 0 # WAIT으로 강제 변경
                filter_info = f"OVERRIDE({context} prob {action_prob:.2f} < {threshold})"
            else:
                filter_info = f"PASS({context} prob {action_prob:.2f} >= {threshold})"
            
            logs.append({"timestamp": ts.isoformat(), "type": "FILTER", "info": filter_info})

        # PREDICT 로그는 필터링 결정 후에 추가 (실제 수행될 action을 기록하기 위함)
        prediction["action"] = final_action # 필터링이 적용된 최종 액션으로 업데이트
        logs.insert(0, {
            "timestamp": ts.isoformat(),
            "type": "PREDICT",
            "value": f"{prediction['value']:.4f}",
            "logits[W,L,S,C]": [f"{l:.3f}" for l in prediction['logits']],
            "masked_action": final_action,
        })

        price = self._last_close()

        if final_action == 3:
            if self.entry_side != 0:
                pnl = self._unrealized(price)
                self.realized += pnl
                logs.append({"timestamp": ts.isoformat(), "type": "CLOSE", "position": "FLAT",
                             "price": f"{price:.2f}", "profit": f"{pnl:.2f}"})
                self.entry_side, self.entry_price, self.entry_qty = 0, None, 0.0
            self.exec.market_close()
        elif final_action in (1, 2):
            if self.entry_side != 0 and ((final_action == 1 and self.entry_side < 0) or (final_action == 2 and self.entry_side > 0)):
                pnl = self._unrealized(price)
                self.realized += pnl
                logs.append({"timestamp": ts.isoformat(), "type": "CLOSE", "position": "FLAT",
                             "price": f"{price:.2f}", "profit": f"{pnl:.2f}"})
            equity_now = self.initial_equity + self.realized + self._unrealized(price)
            qty = self._size_full(equity_now, price)
            if final_action == 1:
                self.exec.market_long(qty);  self.entry_side, self.entry_price, self.entry_qty = +1, price, qty
                logs.append({"timestamp": ts.isoformat(), "type": "ENTRY_LONG", "position": "LONG", "price": f"{price:.2f}"})
            else:
                self.exec.market_short(qty); self.entry_side, self.entry_price, self.entry_qty = -1, price, qty
                logs.append({"timestamp": ts.isoformat(), "type": "ENTRY_SHORT", "position": "SHORT", "price": f"{price:.2f}"})
        
        unrl = self._unrealized(price)
        total_eq = self.initial_equity + self.realized + unrl
        
        after_pos = self.exec.position()
        pos_map = {0: "FLAT", 1: "LONG", -1: "SHORT"}
        pos_str = pos_map[after_pos]
        if before_pos != after_pos:
            pos_str = f"{pos_map[before_pos]}->{pos_map[after_pos]}"

        report = {
            "total_equity": total_eq,
            "unrealized_pnl_amount": unrl,
            "unrealized_pnl_percent": (unrl / total_eq * 100.0) if total_eq > 0 else 0.0,
            "position": pos_map[after_pos],
        }
        
        prediction_str = self._format_prediction_string(prediction, before_pos)
        summary = {
            "pos": pos_str, 
            "price": f"{price:.2f}", 
            "equity": f"{total_eq:.2f}", 
            "value": f"{prediction['value']:.4f}",
            "prediction": prediction_str
        }
        if filter_info:
            summary["filter"] = filter_info

        return StepResult(logs=logs, report_snapshot=report, summary=summary)

# ===== 페이퍼 트레이더 =====
class PaperTrader(BaseTrader):
    def __init__(self, data_client: PublicBinanceData, exec_client: PaperBroker,
                 model_path: Optional[str] = None, vec_path: Optional[str] = None,
                 norm_reward_at_train: bool = False, symbol_eth: str = "ETHUSDT",
                 leverage: float = 5.0, risk_fraction: float = 1.0, init_equity: float = 10_000.0,
                 prob_threshold: float = 0.65):
        super().__init__(exec_client, data_client, model_path, vec_path, norm_reward_at_train,
                         symbol_eth, leverage, risk_fraction, init_equity)
        self.realized = 0.0
        self.entry_side = 0
        self.entry_price: Optional[float] = None
        self.entry_qty: float = 0.0
        self.prob_threshold = prob_threshold

    def _unrealized(self, price: float) -> float:
        if self.entry_side == 0 or self.entry_price is None:
            return 0.0
        sign = 1.0 if self.entry_side > 0 else -1.0
        return (price - self.entry_price) * sign * self.entry_qty

    def step(self, obs_vec: np.ndarray, ts: datetime) -> StepResult:
        before_pos = self.exec.position()
        prediction = self._predict_action(obs_vec)
        action = prediction["action"]

        logs: List[Dict[str, Any]] = []
        filter_info = None

        # 확률 임계값 필터: 무포지션일 때, 진입 액션(1:Long, 2:Short)에만 적용
        if before_pos == 0 and action in (1, 2):
            import torch
            logits = torch.tensor(prediction["logits"])
            probs = torch.nn.functional.softmax(logits, dim=0)
            action_prob = probs[action].item()
            
            if action_prob < self.prob_threshold:
                filter_info = f"OVERRIDE(prob {action_prob:.2f} < {self.prob_threshold})"
                logs.append({"timestamp": ts.isoformat(), "type": "FILTER", "info": filter_info})
                action = 0  # 액션을 WAIT으로 강제 변경

        # PREDICT 로그는 필터링 결정 후에 추가 (실제 수행될 action을 기록하기 위함)
        prediction["action"] = action # 필터링이 적용된 최종 액션으로 업데이트
        logs.insert(0, {
            "timestamp": ts.isoformat(),
            "type": "PREDICT",
            "value": f"{prediction['value']:.4f}",
            "logits[W,L,S,C]": [f"{l:.3f}" for l in prediction['logits']],
            "masked_action": action,
        })

        price = self._last_close()

        if action == 3:
            if self.entry_side != 0:
                pnl = self._unrealized(price)
                self.realized += pnl
                logs.append({"timestamp": ts.isoformat(), "type": "CLOSE", "position": "FLAT",
                             "price": f"{price:.2f}", "profit": f"{pnl:.2f}"})
                self.entry_side, self.entry_price, self.entry_qty = 0, None, 0.0
            self.exec.market_close()
        elif action in (1, 2):
            if self.entry_side != 0 and ((action == 1 and self.entry_side < 0) or (action == 2 and self.entry_side > 0)):
                pnl = self._unrealized(price)
                self.realized += pnl
                logs.append({"timestamp": ts.isoformat(), "type": "CLOSE", "position": "FLAT",
                             "price": f"{price:.2f}", "profit": f"{pnl:.2f}"})
            equity_now = self.initial_equity + self.realized + self._unrealized(price)
            qty = self._size_full(equity_now, price)
            if action == 1:
                self.exec.market_long(qty);  self.entry_side, self.entry_price, self.entry_qty = +1, price, qty
                logs.append({"timestamp": ts.isoformat(), "type": "ENTRY_LONG", "position": "LONG", "price": f"{price:.2f}"})
            else:
                self.exec.market_short(qty); self.entry_side, self.entry_price, self.entry_qty = -1, price, qty
                logs.append({"timestamp": ts.isoformat(), "type": "ENTRY_SHORT", "position": "SHORT", "price": f"{price:.2f}"})
        
        unrl = self._unrealized(price)
        total_eq = self.initial_equity + self.realized + unrl
        pos_str = "LONG" if self.exec.position() > 0 else ("SHORT" if self.exec.position() < 0 else "FLAT")
        report = {
            "total_equity": total_eq,
            "unrealized_pnl_amount": unrl,
            "unrealized_pnl_percent": (unrl / total_eq * 100.0) if total_eq > 0 else 0.0,
            "position": pos_str,
        }
        
        prediction_str = self._format_prediction_string(prediction, before_pos)
        summary = {
            "pos": pos_str, 
            "price": f"{price:.2f}", 
            "equity": f"{total_eq:.2f}", 
            "value": f"{prediction['value']:.4f}",
            "prediction": prediction_str
        }
        if filter_info:
            summary["filter"] = filter_info

        return StepResult(logs=logs, report_snapshot=report, summary=summary)
