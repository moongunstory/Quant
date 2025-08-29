# ai_binance/live/execution.py
"""
Binance UM Futures Execution Adapter (revamped)
- Refactored based on user feedback to ensure signature integrity.
- Uses ordered list of tuples for params to guarantee query string order.
- Manually builds URL with signature to bypass requests lib dictionary serialization.
"""

from __future__ import annotations

import hmac
import time
import math
import hashlib
from typing import Optional, List, Tuple, Dict, Any
from urllib.parse import urlencode

import requests

class BinanceExecutor:
    PROD_URL = "https://fapi.binance.com"
    TEST_URL = "https://testnet.binancefuture.com"

    def __init__(self, api_key: str, secret_key: str, *, recv_window: int = 5000, timeout: int = 15, testnet: bool = False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.recv_window = str(recv_window)
        self.timeout = int(timeout)
        self.base_url = self.TEST_URL if testnet else self.PROD_URL
        self._session = requests.Session()
        self._filters: Dict[str, Dict[str, float]] = {}

    # --- Core Request/Signature Logic (User's Design) ---

    def _send_request(self, http_method: str, path: str, params_pairs: List[Tuple[str, str]]):
        # Add recvWindow and timestamp to all signed requests, maintaining order
        params_pairs.append(("recvWindow", self.recv_window))
        params_pairs.append(("timestamp", str(int(time.time() * 1000))))
        
        query_string = urlencode(params_pairs, doseq=False)
        signature = hmac.new(self.secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()
        
        headers = {"X-MBX-APIKEY": self.api_key}
        
        try:
            if http_method.upper() == 'GET':
                url = f"{self.base_url}{path}?{query_string}&signature={signature}"
                r = self._session.get(url, headers=headers, timeout=self.timeout)
            elif http_method.upper() == 'POST':
                url = f"{self.base_url}{path}"
                body = f"{query_string}&signature={signature}"
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                r = self._session.post(url, data=body, headers=headers, timeout=self.timeout)
            elif http_method.upper() == 'DELETE':
                url = f"{self.base_url}{path}?{query_string}&signature={signature}"
                r = self._session.delete(url, headers=headers, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {http_method}")

            r.raise_for_status()
            return r.json() if r.text else {"success": True}
        except requests.exceptions.HTTPError as e:
            print(f"[EXECUTOR] HTTP Error: {e.response.status_code} {e.response.text}")
            raise
        except Exception as e:
            print(f"[EXECUTOR] Request Failed: {e}")
            raise

    # ---------- Public Methods (Refactored) ----------

    def get_usdt_balance(self) -> float:
        """선물 계정의 사용 가능한 USDT 잔액을 조회."""
        data = self._send_request("GET", "/fapi/v2/balance", [])
        if isinstance(data, list):
            for asset_data in data:
                if isinstance(asset_data, dict) and asset_data.get("asset") == "USDT":
                    return float(asset_data.get("availableBalance", "0.0"))
        return 0.0

    def place_market(self, symbol: str, side: str, quantity: float, *, reduce_only: bool = False):
        params = [
            ("symbol", symbol),
            ("side", side.upper()),
            ("type", "MARKET"),
            ("quantity", str(quantity)),
            ("reduceOnly", "true" if reduce_only else "false"),
        ]
        return self._send_request("POST", "/fapi/v1/order", params)

    def place_stop_loss_close(self, symbol: str, entry_side: str, entry_price: float, *, sl_rate: float = 0.035):
        f = self._load_filters(symbol)
        opp_side = "SELL" if entry_side.upper() == "BUY" else "BUY"
        if entry_side.upper() == "BUY":
            stop_price = float(entry_price) * (1.0 - float(sl_rate))
            stop_price = math.floor(stop_price / f["tickSize"]) * f["tickSize"]
        else:
            stop_price = float(entry_price) * (1.0 + float(sl_rate))
            stop_price = math.ceil(stop_price / f["tickSize"]) * f["tickSize"]

        params = [
            ("symbol", symbol),
            ("side", opp_side),
            ("type", "STOP_MARKET"),
            ("stopPrice", f"{stop_price:.8f}".rstrip('0').rstrip('.')),
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
        return self._send_request("DELETE", "/fapi/v1/allOpenOrders", [("symbol", symbol)])

    def _load_filters(self, symbol: str) -> dict:
        if symbol in self._filters:
            return self._filters[symbol]
        
        # This is a public endpoint, no signature needed
        r = self._session.get(f"{self.base_url}/fapi/v1/exchangeInfo")
        r.raise_for_status()
        data = r.json()
        
        info = next((s for s in data["symbols"] if s["symbol"] == symbol), None)
        if not info: raise ValueError(f"Could not find symbol info for {symbol}")

        filters = {}
        for f in info["filters"]:
            t = f.get("filterType")
            if t == "PRICE_FILTER":
                filters["tickSize"] = float(f["tickSize"])
            elif t == "LOT_SIZE":
                filters["stepSize"] = float(f["stepSize"])
                filters["minQty"] = float(f["minQty"])
            elif t == "MIN_NOTIONAL":
                filters["minNotional"] = float(f["minNotional"])
        
        self._filters[symbol] = filters
        return self._filters[symbol]