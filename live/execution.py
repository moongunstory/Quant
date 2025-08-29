# ai_binance/live/execution.py
"""
Binance UM Futures Execution Adapter (revamped)
- Signature integrity: timestamp/recvWindow 자동 주입 + 서버시각 동기화 + -1021 1회 재시도
- Ordered params → exact querystring signing
- Precision-safe formatting for quantity/price to avoid -1111
"""

from __future__ import annotations

import hmac
import time
import math
import hashlib
import json
from typing import List, Tuple, Dict, Any
from urllib.parse import urlencode

import requests

# --- precision utils ---
from decimal import Decimal, getcontext
getcontext().prec = 28  # 충분히 크게

def _decimals_from_step(step: float) -> int:
    s = f"{step:.16f}".rstrip('0').rstrip('.')
    return 0 if '.' not in s else len(s.split('.')[1])

def _fmt_down(value: float, step: float) -> str:
    dval = Decimal(str(value))
    dstep = Decimal(str(step))
    v = (dval // dstep) * dstep  # 정확한 내림
    return f"{v:.{_decimals_from_step(step)}f}"

def _fmt_up(value: float, step: float) -> str:
    dval = Decimal(str(value))
    dstep = Decimal(str(step))
    q = (dval // dstep)
    v = dval if dval == q * dstep else (q + 1) * dstep
    return f"{v:.{_decimals_from_step(step)}f}"


class BinanceExecutor:
    PROD_URL = "https://fapi.binance.com"
    TEST_URL = "https://testnet.binancefuture.com"

    # ===================== init =====================
    def __init__(self, api_key: str, secret_key: str, *, recv_window: int = 5000, timeout: int = 15, testnet: bool = False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.recv_window = int(recv_window)
        self.timeout = int(timeout)
        self.base_url = self.TEST_URL if testnet else self.PROD_URL
        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": api_key})
        self._filters: Dict[str, Dict[str, float]] = {}

        # 서버 시각 동기화용 오프셋(ms)
        self._time_offset_ms = 0
        self._sync_time()  # 시작 시 1회 동기화

    # ===================== time & signing utils =====================
    def _sync_time(self) -> None:
        """Sync local time with Binance server and store offset in ms."""
        r = self._session.get(f"{self.base_url}/fapi/v1/time", timeout=10)
        r.raise_for_status()
        server_ms = int(r.json()["serverTime"])
        local_ms = int(time.time() * 1000)
        self._time_offset_ms = server_ms - local_ms

    def _now_ms(self) -> int:
        """Return current ms with server offset applied."""
        return int(time.time() * 1000) + self._time_offset_ms

    def _build_signed_qs(self, params_pairs: List[Tuple[str, str | int | float]]):
        """
        Ensure timestamp/recvWindow are always present, then sign the exact query string.
        Keeps original order from params_pairs.
        """
        pairs = list(params_pairs)

        if not any(k == "timestamp" for k, _ in pairs):
            pairs.append(("timestamp", self._now_ms()))
        if not any(k == "recvWindow" for k, _ in pairs):
            pairs.append(("recvWindow", self.recv_window))

        query_string = urlencode(pairs, doseq=False)
        sig = hmac.new(self.secret_key.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
        return query_string, sig

    # ===================== core request =====================
    def _send_request(self, http_method: str, path: str, params_pairs: List[Tuple[str, str | int | float]]):
        def _do_once():
            qs, sig = self._build_signed_qs(params_pairs)
            headers = self._session.headers  # 이미 X-MBX-APIKEY 포함

            if http_method.upper() == 'GET':
                url = f"{self.base_url}{path}?{qs}&signature={sig}"
                return self._session.get(url, timeout=self.timeout, headers=headers)
            elif http_method.upper() == 'POST':
                url = f"{self.base_url}{path}"
                body = f"{qs}&signature={sig}"
                h = dict(headers)
                h["Content-Type"] = "application/x-www-form-urlencoded"
                return self._session.post(url, data=body, timeout=self.timeout, headers=h)
            elif http_method.upper() == 'DELETE':
                url = f"{self.base_url}{path}?{qs}&signature={sig}"
                return self._session.delete(url, timeout=self.timeout, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {http_method}")

        try:
            r = _do_once()

            # -1021(Timestamp outside recvWindow) 대응: 1회 재동기화 후 재시도
            if r.status_code >= 400:
                try:
                    err = r.json()
                except json.JSONDecodeError:
                    err = {"code": None, "msg": r.text}

                if err.get("code") == -1021:
                    self._sync_time()
                    r = _do_once()  # 재시도

            # 에러 처리
            if r.status_code >= 400:
                try:
                    error_data = r.json()
                    print(f"[EXECUTOR] HTTP Error: {r.status_code} {error_data}")
                except json.JSONDecodeError:
                    print(f"[EXECUTOR] HTTP Error: {r.status_code} {r.text}")
                r.raise_for_status()

            return r.json() if r.text else {"success": True}

        except Exception as e:
            print(f"[EXECUTOR] Request Failed: {e}")
            raise

    # ===================== helpers =====================
    @staticmethod
    def _round_step(x: float, step: float) -> float:
        return math.floor(x / step + 1e-12) * step

    def _load_filters(self, symbol: str) -> dict:
        if symbol in self._filters:
            return self._filters[symbol]

        # Public endpoint
        try:
            r = self._session.get(f"{self.base_url}/fapi/v1/exchangeInfo", timeout=10)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[EXECUTOR] Failed to load exchange info: {e}")
            # fallback defaults
            return {"tickSize": 0.01, "stepSize": 0.001, "minQty": 0.001, "minNotional": 5.0}

        info = next((s for s in data["symbols"] if s["symbol"] == symbol), None)
        if not info:
            raise ValueError(f"Could not find symbol info for {symbol}")

        filters: Dict[str, float] = {}
        for f in info["filters"]:
            t = f.get("filterType")
            if t == "PRICE_FILTER":
                filters["tickSize"] = float(f["tickSize"])
            elif t == "LOT_SIZE":
                filters["stepSize"] = float(f["stepSize"])
                filters["minQty"] = float(f["minQty"])
            elif t == "MIN_NOTIONAL":
                # 키 이름이 배포 시점에 따라 다를 수 있음
                filters["minNotional"] = float(f.get("notional", f.get("minNotional", 5.0)))

        self._filters[symbol] = filters
        return self._filters[symbol]

    # ===================== public methods =====================
    def get_usdt_balance(self) -> float:
        data = self._send_request("GET", "/fapi/v2/balance", [])
        if isinstance(data, list):
            for asset_data in data:
                if isinstance(asset_data, dict) and asset_data.get("asset") == "USDT":
                    return float(asset_data.get("availableBalance", "0.0"))
        return 0.0

    def place_market(self, symbol: str, side: str, quantity: float, *, reduce_only: bool = False):
        f = self._load_filters(symbol)
        step = float(f["stepSize"])
        min_qty = float(f["minQty"])

        qty_raw = abs(float(quantity))
        # step 기준 내림 + 최소수량 보정
        qty_step = max(min_qty, math.floor(qty_raw / step + 1e-12) * step)
        # 자릿수 고정 문자열(핵심)
        qty_str = _fmt_down(qty_step, step)

        params = [
            ("symbol", symbol),
            ("side", side.upper()),
            ("type", "MARKET"),
            ("quantity", qty_str),
            ("reduceOnly", "true" if reduce_only else "false"),
        ]
        return self._send_request("POST", "/fapi/v1/order", params)

    def place_stop_loss_close(self, symbol: str, entry_side: str, entry_price: float, *, sl_rate: float = 0.035):
        f = self._load_filters(symbol)
        tick = float(f["tickSize"])
        opp_side = "SELL" if entry_side.upper() == "BUY" else "BUY"

        if entry_side.upper() == "BUY":
            raw = float(entry_price) * (1.0 - float(sl_rate))
            stop_price_str = _fmt_down(raw, tick)   # 롱 SL: 아래로 내림
        else:
            raw = float(entry_price) * (1.0 + float(sl_rate))
            stop_price_str = _fmt_up(raw, tick)     # 숏 SL: 위로 올림

        params = [
            ("symbol", symbol),
            ("side", opp_side),
            ("type", "STOP_MARKET"),
            ("stopPrice", stop_price_str),
            ("closePosition", "true"),
        ]
        return self._send_request("POST", "/fapi/v1/order", params)

    def entry_with_stop(self, symbol: str, side: str, quantity: float, *, last_price: float, sl_rate: float = 0.035):
        out: Dict[str, Any] = {}
        out["cancel"] = self.cancel_all_orders(symbol)
        out["entry"] = self.place_market(symbol, side, quantity, reduce_only=False)
        out["sl"] = self.place_stop_loss_close(symbol, side, last_price, sl_rate=sl_rate)
        return out

    def cancel_all_orders(self, symbol: str):
        try:
            return self._send_request("DELETE", "/fapi/v1/allOpenOrders", [("symbol", symbol)])
        except Exception as e:
            if "No open orders" in str(e):
                return {"code": "NO_OPEN_ORDERS"}
            raise
