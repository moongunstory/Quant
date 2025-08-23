# ai_binance/live/execution.py
"""
Binance UM Futures Execution Adapter (revamped)
- 유지: 취소 → 시장가 진입 → SL(STOP_MARKET closePosition) 예약
- 추가: testnet 지원, 레버리지/마진/포지션모드 설정 API, 정밀도/최소주문가드,
        레이트리밋/시계오프셋 자동보정, 에러코드 핸들링

사용 예:
    ex = BinanceExecutor(api_key, secret_key, testnet=False)
    ex.ensure_oneway_mode()
    ex.set_margin_type("ISOLATED")
    ex.set_leverage("ETHUSDT", 5)
    resp = ex.entry_with_stop("ETHUSDT", "BUY", quantity=0.5, last_price=3250.0, sl_rate=0.035)
    ex.close_position_market("ETHUSDT")
"""

from __future__ import annotations

import hmac
import time
import math
import hashlib
from typing import Optional, Tuple, Dict, Any

import requests


class BinanceExecutor:
    PROD_URL = "https://fapi.binance.com"
    TEST_URL = "https://testnet.binancefuture.com"

    def __init__(self, api_key: str, secret_key: str, *, recv_window: int = 5000, timeout: int = 15, testnet: bool = False, max_retries: int = 3):
        self.api_key = api_key
        self.secret_key = secret_key.encode("utf-8")
        self.recv_window = int(recv_window)
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)
        self.base_url = self.TEST_URL if testnet else self.PROD_URL

        # 심볼별 필터 캐시
        self._filters: dict[str, dict[str, float]] = {}
        # 서버 타임 오프셋(밀리초)
        self._time_offset_ms = self._calc_time_offset_ms()

        self._session = requests.Session()

    # ---------- 내부 HTTP/서명/요청 ----------

    def _calc_time_offset_ms(self) -> int:
        try:
            r = self._session.get(self.base_url + "/fapi/v1/time", timeout=self.timeout)
            r.raise_for_status()
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

    def _request(self, method: str, path: str, params: dict | None = None, *, signed: bool = False):
        params = dict(params or {})
        attempt = 0
        while True:
            try:
                if signed:
                    params.setdefault("timestamp", self._ts())
                    params.setdefault("recvWindow", self.recv_window)
                    params = self._sign(params)

                url = self.base_url + path
                fn = getattr(self._session, method.lower())
                r = fn(
                    url,
                    params=params if method in ("GET", "DELETE") else None,
                    data=params if method in ("POST", "PUT") else None,
                    headers=self._headers(),
                    timeout=self.timeout
                )
                r.raise_for_status()
                data = r.json()
                # 바이낸스는 200이어도 에러코드가 payload에 있을 수 있음
                if isinstance(data, dict) and "code" in data and isinstance(data["code"], int) and data["code"] < 0:
                    code = data.get("code")
                    msg = data.get("msg", "")
                    # 타임스탬프 범위 오류 → 오프셋 재계산 후 재시도
                    if code == -1021 and attempt < self.max_retries:
                        self._time_offset_ms = self._calc_time_offset_ms()
                        attempt += 1
                        continue
                    raise requests.HTTPError(f"Binance API error {code}: {msg}")
                return data
            except requests.HTTPError as e:
                # 레이트리밋 계열: 418/429, 코드 -1003/-1121 등 → 단순 백오프 재시도
                status = getattr(e.response, "status_code", None)
                if (status in (418, 429) or "API error -1003" in str(e)) and attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    attempt += 1
                    continue
                raise
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    attempt += 1
                    continue
                raise

    # ---------- 메타/필터 ----------

    def _load_filters(self, symbol: str) -> dict:
        if symbol in self._filters:
            return self._filters[symbol]
        data = self._request("GET", "/fapi/v1/exchangeInfo", {"symbol": symbol})
        info = data["symbols"][0]

        tick = 0.01
        step = 0.001
        min_qty = 0.0
        min_notional = 0.0
        for f in info["filters"]:
            t = f.get("filterType")
            if t == "PRICE_FILTER":
                tick = float(f["tickSize"])
            elif t in ("LOT_SIZE", "MARKET_LOT_SIZE"):
                step = float(f["stepSize"])
                min_qty = max(min_qty, float(f.get("minQty", 0.0)))
            elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                # UM futures는 NOTIONAL 필터를 사용
                mn = float(f.get("notional", f.get("minNotional", 0.0)))
                min_notional = max(min_notional, mn)

        self._filters[symbol] = {
            "tickSize": tick,
            "stepSize": step,
            "minQty": min_qty,
            "minNotional": min_notional,
        }
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

    # ---------- 계정 설정(옵션) ----------

    def ensure_oneway_mode(self) -> dict:
        """
        포지션 모드를 단일(ONEWAY)로 설정. 이미 설정이면 OK.
        """
        try:
            st = self._request("GET", "/fapi/v1/positionSide/dual", signed=True)
            dual = bool(st.get("dualSidePosition", False))
        except Exception:
            dual = False
        if dual:
            return self._request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "false"}, signed=True)
        return {"ok": True, "dualSidePosition": False}

    def set_margin_type(self, margin_type: str = "ISOLATED") -> dict:
        """
        교차/격리 마진 설정: margin_type ∈ {"CROSSED","ISOLATED"}
        """
        mt = margin_type.upper()
        if mt not in ("CROSSED", "ISOLATED"):
            raise ValueError("margin_type must be 'CROSSED' or 'ISOLATED'")
        try:
            return self._request("POST", "/fapi/v1/marginType", {"marginType": mt}, signed=True)
        except requests.HTTPError as e:
            # 이미 설정되어 있는 경우 코드(-4046 등)는 무시
            return {"warn": str(e)}

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        lev = max(1, min(int(leverage), 125))
        return self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": lev}, signed=True)

    # ---------- 공개 API: 취소/진입/SL/평청 ----------

    def cancel_all_orders(self, symbol: str) -> dict:
        """해당 심볼의 모든 미체결 주문(TP/SL 포함) 취소."""
        try:
            return self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
        except requests.HTTPError as e:
            # 미체결 없음 등은 무시
            if e.response is not None and e.response.status_code in (400, 404):
                return {"code": "NO_OPEN_ORDERS"}
            raise

    def place_market(self, symbol: str, side: str, quantity: float, *, reduce_only: bool = False, new_client_order_id: Optional[str] = None) -> dict:
        """
        시장가 진입/청산. reduce_only=True면 포지션 감소만.
        - 수량 정밀도/최소주문/최소명목가드 적용
        """
        f = self._load_filters(symbol)
        qty = max(self._round_step(abs(float(quantity)), f["stepSize"]), f["minQty"])
        if qty <= 0:
            raise ValueError("quantity too small after precision/filters")

        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": f"{qty}",
            "reduceOnly": "true" if reduce_only else "false",
        }
        if new_client_order_id:
            params["newClientOrderId"] = str(new_client_order_id)
        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def place_stop_loss_close(self, symbol: str, entry_side: str, entry_price: float,
                              *, sl_rate: float = 0.035, working_type: str = "MARK_PRICE", price_protect: Optional[bool] = None) -> dict:
        """
        3.5% SL을 STOP_MARKET + closePosition으로 예약.
        - Long(BUY 진입): SELL STOP_MARKET at entry*(1 - sl_rate)
        - Short(SELL 진입): BUY  STOP_MARKET at entry*(1 + sl_rate)
        """
        f = self._load_filters(symbol)
        opp = self._opp(entry_side)
        if entry_side.upper() == "BUY":
            raw = float(entry_price) * (1.0 - float(sl_rate))
            stop = self._round_tick_down(raw, f["tickSize"])
        else:
            raw = float(entry_price) * (1.0 + float(sl_rate))
            stop = self._round_tick_up(raw, f["tickSize"])

        params = {
            "symbol": symbol,
            "side": opp,
            "type": "STOP_MARKET",
            "stopPrice": f"{stop:.8f}",
            "closePosition": "true",
            "workingType": working_type,  # MARK_PRICE 권장
        }
        if price_protect is not None:
            params["priceProtect"] = "true" if price_protect else "false"

        return self._request("POST", "/fapi/v1/order", params, signed=True)

    def entry_with_stop(self, symbol: str, side: str, quantity: float, *, last_price: float, sl_rate: float = 0.035) -> dict:
        """
        1) 기존 TP/SL 포함 모든 미체결 주문 취소
        2) 시장가 진입
        3) 3.5% SL(STOP_MARKET closePosition) 예약
        반환: {"cancel": ..., "entry": ..., "sl": ...}
        """
        out: dict[str, Any] = {}
        out["cancel"] = self.cancel_all_orders(symbol)
        out["entry"] = self.place_market(symbol, side, quantity, reduce_only=False)

        # 체결가 근사: 전달받은 last_price 사용
        fill_px = float(last_price)
        out["sl"] = self.place_stop_loss_close(symbol, side, fill_px, sl_rate=sl_rate)
        return out

    # ---------- 포지션 조회/즉시 평청 ----------

    def get_position_qty(self, symbol: str) -> tuple[float, str]:
        """
        현재 포지션 수량과 방향을 반환.
        반환: (abs_qty, side)  side ∈ {"LONG","SHORT","FLAT"}
        """
        data = self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
        pos = data[0] if isinstance(data, list) and data else data
        amt = float(pos.get("positionAmt", "0"))
        if amt > 0:
            return abs(amt), "LONG"
        elif amt < 0:
            return abs(amt), "SHORT"
        else:
            return 0.0, "FLAT"

    def close_position_market(self, symbol: str) -> dict:
        """
        현재 포지션을 MARKET + reduceOnly=True로 즉시 평청.
        - LONG이면 SELL, SHORT이면 BUY로 실행
        - 포지션이 없으면 {"code":"FLAT"} 반환
        """
        qty, side = self.get_position_qty(symbol)
        if qty <= 0 or side == "FLAT":
            return {"code": "FLAT"}
        side_api = "SELL" if side == "LONG" else "BUY"
        return self.place_market(symbol, side_api, quantity=qty, reduce_only=True)
