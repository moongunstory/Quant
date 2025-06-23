import time
from math import floor
from binance.um_futures import UMFutures
from modules.config import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    TP_THRESHOLD,
    SL_THRESHOLD,
    FUTURES_SYMBOL,
    FUTURES_LEVERAGE,
    FUTURES_MARGIN_TYPE,
)


def calculate_futures_quantity(usdt_balance: float, price: float, leverage: int = FUTURES_LEVERAGE, step_size: float = 0.001):
    notional = usdt_balance * leverage
    qty = notional / price
    return round(floor(qty / step_size) * step_size, 3)


class FuturesTradeExecutor:
    def __init__(self, client=None, symbol=FUTURES_SYMBOL, leverage=FUTURES_LEVERAGE):
        self.client = client or UMFutures(key=BINANCE_API_KEY, secret=BINANCE_SECRET_KEY)
        self.symbol = symbol
        self.leverage = leverage
        self._setup_leverage()

        self.position = None
        self.entry_price = None
        self.tp_order_id = None
        self.sl_order_id = None

    def _setup_leverage(self):
        try:
            self.client.change_margin_type(symbol=self.symbol, marginType=FUTURES_MARGIN_TYPE)
        except Exception as e:
            if "-4046" in str(e) or "No need to change margin type" in str(e):
                pass  # 이미 설정된 경우 → 조용히 무시
            else:
                print("⚠️ Margin type setup failed:", e)

        try:
            self.client.change_leverage(symbol=self.symbol, leverage=self.leverage)
        except Exception as e:
            print("⚠️ Leverage setup failed:", e)

    def get_balance(self, asset="USDT"):
        balances = self.client.balance()
        for b in balances:
            if b["asset"] == asset:
                return float(b["availableBalance"])
        raise ValueError("No USDT balance")

    def market_entry(self, side: str, quantity: float):
        if quantity <= 0:
            raise ValueError(f"🚫 진입 수량이 0 이하: qty={quantity}")
        return self.client.new_order(
            symbol=self.symbol,
            side=side.upper(),
            type="MARKET",
            quantity=quantity,
        )

    def cancel_existing_orders(self):
        if self.position is not None:
            return  # ✅ 포지션 보유 중이면 건드리지 않음
        if not self.tp_order_id and not self.sl_order_id:
            return  # ✅ 취소할 주문이 명시적으로 없음

        try:
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            for order in open_orders:
                if "orderId" in order:
                    self.client.cancel_order(symbol=self.symbol, orderId=order["orderId"])
            self.tp_order_id = None
            self.sl_order_id = None
            print("🧹 기존 TP/SL 주문 전부 취소됨")
        except Exception as e:
            print(f"❌ 주문 취소 중 오류: {e}")

    def calculate_full_quantity(self, price: float):
        try:
            usdt_balance = self.get_balance()
            return calculate_futures_quantity(usdt_balance, price, self.leverage)
        except Exception as e:
            print(f"❌ 수량 계산 실패: {e}")
            return None

    def enter_position(self, direction, current_price):
        if direction not in ["long", "short"]:
            print(f"🚫 잘못된 방향: {direction}")
            return

        self.cancel_existing_orders()

        side = "BUY" if direction == "long" else "SELL"
        exit_side = "SELL" if side == "BUY" else "BUY"

        quantity = self.calculate_full_quantity(current_price)
        if quantity is None:
            print("🚫 수량 계산 실패 → 진입 스킵")
            return

        try:
            self.market_entry(side, quantity)
            print(f"🚀 시장가 진입: {direction.upper()} @ {current_price:.2f}, 수량: {quantity}")
            self.position = direction
        except Exception as e:
            print(f"❌ 시장가 진입 실패: {e}")
            self.position = None       # ✅ 실패 시 상태 초기화
            self.entry_price = None
            return
        
        time.sleep(0.5)

        tp_price = round(current_price * (1 + TP_THRESHOLD if direction == "long" else 1 - TP_THRESHOLD), 2)
        sl_price = round(current_price * (1 + SL_THRESHOLD if direction == "long" else 1 - SL_THRESHOLD), 2)

        try:
            tp_order = self.client.new_order(
                symbol=self.symbol,
                side=exit_side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp_price,
                closePosition=False,
                quantity=quantity,
                reduceOnly=True,
            )
            self.tp_order_id = tp_order["orderId"]

            sl_order = self.client.new_order(
                symbol=self.symbol,
                side=exit_side,
                type="STOP_MARKET",
                stopPrice=sl_price,
                closePosition=False,
                quantity=quantity,
                reduceOnly=True,
            )
            self.sl_order_id = sl_order["orderId"]

            print(f"📌 TP/SL 주문 등록 완료 → TP: {tp_price:.2f}, SL: {sl_price:.2f}")
        except Exception as e:
            print(f"❌ TP/SL 주문 실패: {e}")

    def monitor_position(self):
        if self.position is None:
            return
        try:
            orders = self.client.get_all_orders(symbol=self.symbol)
            tp_filled = any(o["orderId"] == self.tp_order_id and o["status"] == "FILLED" for o in orders)
            sl_filled = any(o["orderId"] == self.sl_order_id and o["status"] == "FILLED" for o in orders)
            if tp_filled or sl_filled:
                print(f"💥 {self.position.upper()} 종료 감지됨 (TP or SL 체결)")
                self.position = None
                self.entry_price = None
                self.tp_order_id = None
                self.sl_order_id = None
        except Exception as e:
            print(f"❌ 포지션 모니터링 오류: {e}")
            
    def should_enter(self):
        """
        현재 포지션이 없을 때만 진입 가능하다는 신호를 반환
        """
        if self.position is not None:
            print(f"⏸️ 이미 {self.position.upper()} 포지션 보유 중 → 진입 판단 생략")
            return False
        return True
