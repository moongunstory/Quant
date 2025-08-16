# ai_binance/live/execution.py
"""
Binance UM Futures Execution Adapter
- 기능 요약:
  1) 진입 전: 심볼의 기존 TP/SL(모든 미체결 주문) 전부 취소
  2) 진입 후: 3.5% 고정 SL을 STOP_MARKET(closePosition)로 예약
- 옵션: 마크가격 트리거(workingType="MARK_PRICE") 기본
- 포지션 모드는 원웨이(단일 포지션) 전제

사용 예:
    ex = BinanceExecutor(api_key, secret_key)
    # 수량은 트레이더에서 계산해 전달
    resp = ex.entry_with_stop(
        symbol="ETHUSDT",
        side="BUY",             # 진입 방향 (BUY: Long, SELL: Short)
        quantity=0.5,           # 계약 수량
        last_price=3250.0,      # 직전 체결/관찰 가격(진입가 근사)
        sl_rate=0.035           # 3.5% 고정
    )
"""

from __future__ import annotations

import hmac
import time
import math
import hashlib
from typing import Optional, Tuple, Dict, Any

import requests


class BinanceExecutor:
    BASE_URL = "https://fapi.binance.com"

    def __init__(self, api_key: str, secret_key: str, recv_window: int = 5000, timeout: int = 15):
        self.api_key = api_key
        self.secret_key = secret_key.encode("utf-8")
        self.recv_window = int(recv_window)
        self.timeout = int(timeout)

        # 심볼별 필터 캐시 (tickSize/stepSize)
        self._filters: dict[str, dict[str, float]] = {}
        # 서버 타임 오프셋(밀리초) — 서명 실패 가드용
        self._time_offset_ms = self._calc_time_offset_ms()

    # ---------- 내부 HTTP/서명 ----------

    def _calc_time_offset_ms(self) -> int:
        try:
            r = requests.get(self.BASE_URL + "/fapi/v1/time", timeout=self.timeout)
            srv = int(r.json()["serverTime"])
            loc = int(time.time() * 1000)
            return srv - loc
        except Exception:
            return 0

    def _ts(self) -> int:
        return int(time.time() * 1000 + self._time_offset_ms)

    def _headers(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    def _sign(self, params: dict) -> dict:
        qs = "&".join([f"{k}={params[k]}" for k in sorted(params.keys()) if params[k] is not None])
        sig = hmac.new(self.secret_key, qs.encode("utf-8"), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = dict(params or {})
        if signed:
            params.setdefault("timestamp", self._ts())
            params.setdefault("recvWindow", self.recv_window)
            params = self._sign(params)
        url = self.BASE_URL + path
        fn = getattr(requests, method.lower())
        r = fn(url, params=params if method in ("GET", "DELETE") else None,
               data=params if method in ("POST", "PUT") else None,
               headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---------- 메타/필터 ----------

    def _load_filters(self, symbol: str) -> dict:
        if symbol in self._filters:
            return self._filters[symbol]
        data = self._request("GET", "/fapi/v1/exchangeInfo", {"symbol": symbol})
        info = data["symbols"][0]
        tick = 0.01
        step = 0.001
        for f in info["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                tick = float(f["tickSize"])
            elif f["filterType"] == "LOT_SIZE":
                step = float(f["stepSize"])
        self._filters[symbol] = {"tickSize": tick, "stepSize": step}
        return self._filters[symbol]

    @staticmethod
    def _round_step(x: float, step: float) -> float:
        return math.floor(x / step + 1e-12) * step

    @staticmethod
    def _round_tick_near(x: float, tick: float) -> float:
        return round(x / tick) * tick

    @staticmethod
    def _round_tick_down(x: float, tick: float) -> float:
        return math.floor(x / tick + 1e-12) * tick

    @staticmethod
    def _round_tick_up(x: float, tick: float) -> float:
        return math.ceil(x / tick - 1e-12) * tick

    @staticmethod
    def _opp(side: str) -> str:
        return "SELL" if side.upper() == "BUY" else "BUY"

    # ---------- 공개 API: 취소/진입/SL ----------

    def cancel_all_orders(self, symbol: str) -> dict:
        """해당 심볼의 모든 미체결 주문(TP/SL 포함) 취소."""
        try:
            return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
        except requests.HTTPError as e:
            # 미체결 없음 등은 무시
            if e.response is not None and e.response.status_code in (400, 404):
                return {"code": "NO_OPEN_ORDERS"}
            raise

    def place_market(self, symbol: str, side: str, quantity: float, reduce_only: bool = False) -> dict:
        """시장가 진입/청산. reduce_only=True면 포지션 감소만."""
        f = self._load_filters(symbol)
        qty = max(self._round_step(abs(quantity), f["stepSize"]), f["stepSize"])
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": f"{qty:.8f}",
            "reduceOnly": "true" if reduce_only else "false",
            # "newClientOrderId": ...  # 필요시
        }
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_stop_loss_close(self, symbol: str, entry_side: str, entry_price: float,
                              sl_rate: float = 0.035, working_type: str = "MARK_PRICE") -> dict:
        """
        3.5% SL을 STOP_MARKET + closePosition으로 예약.
        - Long(BUY 진입): SELL STOP_MARKET at entry*(1 - sl_rate)
        - Short(SELL 진입): BUY  STOP_MARKET at entry*(1 + sl_rate)
        """
        f = self._load_filters(symbol)
        opp = self._opp(entry_side)
        if entry_side.upper() == "BUY":
            raw = entry_price * (1.0 - sl_rate)
            stop = self._round_tick_down(raw, f["tickSize"])  # 약간 더 보수적으로 아래로
        else:
            raw = entry_price * (1.0 + sl_rate)
            stop = self._round_tick_up(raw, f["tickSize"])    # 약간 더 보수적으로 위로

        params = {
            "symbol": symbol,
            "side": opp,
            "type": "STOP_MARKET",
            "stopPrice": f"{stop:.8f}",
            "closePosition": "true",           # 전체 포지션 청산
            "workingType": working_type,       # MARK_PRICE 권장
        }
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    # ---------- 고수준 편의: 취소→진입→SL ----------

    def entry_with_stop(self, symbol: str, side: str, quantity: float, last_price: float,
                        sl_rate: float = 0.035) -> dict:
        """
        1) 기존 TP/SL 포함 모든 미체결 주문 취소
        2) 시장가 진입
        3) 3.5% SL(STOP_MARKET closePosition) 예약
        반환: {"cancel": ..., "entry": ..., "sl": ...}
        """
        out: dict[str, Any] = {}
        out["cancel"] = self.cancel_all_orders(symbol)
        out["entry"] = self.place_market(symbol, side, quantity, reduce_only=False)

        # 진입 체결 가격 근사: 전달받은 last_price 사용(실거래는 거래소 fillPrice로 보강 가능)
        try:
            fill_px = float(last_price)
        except Exception:
            fill_px = last_price

        out["sl"] = self.place_stop_loss_close(symbol, side, fill_px, sl_rate=sl_rate)
        return out
