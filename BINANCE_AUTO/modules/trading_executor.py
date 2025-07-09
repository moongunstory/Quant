import time
from math import floor
from binance.client import Client
from modules.config import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
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
        self.client = client or Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
        self.symbol = symbol
        self.leverage = leverage
        self._setup_leverage()

        self.position = None
        self.entry_price = None
        self.tp_order_id = None
        self.sl_order_id = None

    def _setup_leverage(self):
        try:
            self.client.futures_change_margin_type(symbol=self.symbol, marginType=FUTURES_MARGIN_TYPE)
        except Exception as e:
            if "-4046" in str(e) or "No need to change margin type" in str(e):
                pass  # 이미 설정된 경우 → 조용히 무시
            else:
                print("⚠️ Margin type setup failed:", e)

        try:
            self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
        except Exception as e:
            print("⚠️ Leverage setup failed:", e)

    def get_balance(self, asset="USDT"):
        balances = self.client.futures_account_balance()
        for b in balances:
            if b["asset"] == asset:
                return float(b["availableBalance"])
        raise ValueError("No USDT balance")

    def market_entry(self, side: str, quantity: float):
        if quantity <= 0:
            raise ValueError(f"🚫 진입 수량이 0 이하: qty={quantity}")
        return self.client.futures_create_order(
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
            open_orders = self.client.futures_get_open_orders(symbol=self.symbol)
            for order in open_orders:
                if "orderId" in order:
                    self.client.futures_cancel_order(symbol=self.symbol, orderId=order["orderId"])
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
            self.entry_price = current_price # Set entry price upon successful entry
        except Exception as e:
            print(f"❌ 시장가 진입 실패: {e}")
            self.position = None       # ✅ 실패 시 상태 초기화
            self.entry_price = None
            return

    def close_position(self, current_price):
        if self.position is None:
            print("🚫 닫을 포지션 없음")
            return

        side = "SELL" if self.position == "long" else "BUY"
        quantity = self.calculate_full_quantity(current_price) # Use current price for quantity calculation
        if quantity is None:
            print("🚫 수량 계산 실패 → 청산 스킵")
            return

        try:
            self.market_entry(side, quantity)
            print(f"✅ 포지션 청산: {self.position.upper()} @ {current_price:.2f}, 수량: {quantity}")
            self.position = None
            self.entry_price = None
        except Exception as e:
            print(f"❌ 포지션 청산 실패: {e}")

    def monitor_position(self):
        if self.position is None:
            return
        try:
            # Check actual position from Binance API
            account_info = self.client.futures_account()
            current_position_info = None
            for p in account_info['positions']:
                if p['symbol'] == self.symbol:
                    current_position_info = p
                    break
            
            if current_position_info and float(current_position_info['positionAmt']) == 0:
                print(f"💥 {self.position.upper()} 포지션 종료 감지됨 (잔고 0)")
                self.position = None
                self.entry_price = None
            
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

    def execute_rl_action(self, action_str: str, current_price: float):
        """
        Executes the trading action determined by the RL agent.
        Args:
            action_str: The action string from the RL agent (e.g., 'attempt_long', 'attempt_short', 'close_position', 'no_action').
            current_price: The current market price.
        """
        print(f"[EXECUTE RL ACTION] Received action: {action_str} at price: {current_price:.2f}")

        if action_str == 'attempt_long':
            if self.position == 'none':
                self.enter_position("long", current_price)
            elif self.position == 'short':
                self.close_position(current_price)
                self.enter_position("long", current_price)
            else: # Already long
                print("ℹ️ 이미 롱 포지션 보유 중, 롱 진입 시도 무시.")

        elif action_str == 'attempt_short':
            if self.position == 'none':
                self.enter_position("short", current_price)
            elif self.position == 'long':
                self.close_position(current_price)
                self.enter_position("short", current_price)
            else: # Already short
                print("ℹ️ 이미 숏 포지션 보유 중, 숏 진입 시도 무시.")

        elif action_str == 'close_position':
            self.close_position(current_price)

        elif action_str == 'no_action':
            print("ℹ️ RL 에이전트: NO_ACTION 선택.")
            pass
        else:
            print(f"⚠️ 알 수 없는 RL 액션: {action_str}")