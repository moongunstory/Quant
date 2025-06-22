import time
from binance.client import Client
from modules.config import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    TP_THRESHOLD,
    SL_THRESHOLD,
    TRADE_SYMBOL,
    TRADE_BALANCE_RATIO
)

class TradeExecutor:
    def __init__(self):
        self.client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
        self.symbol = TRADE_SYMBOL
        self.balance_ratio = TRADE_BALANCE_RATIO

        self.position = None
        self.entry_price = None
        self.tp_order_id = None
        self.sl_order_id = None

    def cancel_existing_orders(self):
        try:
            open_orders = self.client.get_open_orders(symbol=self.symbol)
            for order in open_orders:
                self.client.cancel_order(symbol=self.symbol, orderId=order['orderId'])
            self.tp_order_id = None
            self.sl_order_id = None
            print("🧹 기존 TP/SL 주문 전부 취소됨")
        except Exception as e:
            print(f"❌ 주문 취소 중 오류: {e}")

    def calculate_full_quantity(self):
        try:
            usdt_balance = float(self.client.get_asset_balance(asset='USDT')['free'])
            eth_price = float(self.client.get_symbol_ticker(symbol=self.symbol)['price'])

            order_usdt = usdt_balance * self.balance_ratio
            quantity = round(order_usdt / eth_price, 5)  # 소수점은 바이낸스 기준으로 맞춰야 함
            return quantity
        except Exception as e:
            print(f"❌ 수량 계산 실패: {e}")
            return None

    def enter_position(self, direction, current_price):
        if direction not in ['long', 'short']:
            print(f"🚫 잘못된 방향: {direction}")
            return

        self.cancel_existing_orders()

        self.position = direction
        self.entry_price = current_price

        side = 'BUY' if direction == 'long' else 'SELL'
        exit_side = 'SELL' if side == 'BUY' else 'BUY'

        quantity = self.calculate_full_quantity()
        if quantity is None:
            print("🚫 수량 계산 실패 → 진입 스킵")
            return

        try:
            self.client.create_order(
                symbol=self.symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )
            print(f"🚀 시장가 진입: {direction.upper()} @ {current_price:.2f}, 수량: {quantity}")
        except Exception as e:
            print(f"❌ 시장가 진입 실패: {e}")
            return

        time.sleep(0.5)

        tp_price = round(current_price * (1 + TP_THRESHOLD if direction == 'long' else 1 - TP_THRESHOLD), 2)
        sl_price = round(current_price * (1 + SL_THRESHOLD if direction == 'long' else 1 - SL_THRESHOLD), 2)

        try:
            tp_order = self.client.create_order(
                symbol=self.symbol,
                side=exit_side,
                type='TAKE_PROFIT',
                stopPrice=tp_price,
                quantity=quantity,
                timeInForce='GTC'
            )
            self.tp_order_id = tp_order['orderId']

            sl_order = self.client.create_order(
                symbol=self.symbol,
                side=exit_side,
                type='STOP_LOSS',
                stopPrice=sl_price,
                quantity=quantity,
                timeInForce='GTC'
            )
            self.sl_order_id = sl_order['orderId']

            print(f"📌 TP/SL 주문 등록 완료 → TP: {tp_price:.2f}, SL: {sl_price:.2f}")

        except Exception as e:
            print(f"❌ TP/SL 주문 실패: {e}")

    def monitor_position(self):
        if self.position is None:
            return

        try:
            orders = self.client.get_all_orders(symbol=self.symbol)
            tp_filled = any(o['orderId'] == self.tp_order_id and o['status'] == 'FILLED' for o in orders)
            sl_filled = any(o['orderId'] == self.sl_order_id and o['status'] == 'FILLED' for o in orders)

            if tp_filled or sl_filled:
                print(f"💥 {self.position.upper()} 종료 감지됨 (TP or SL 체결)")
                self.position = None
                self.entry_price = None
                self.tp_order_id = None
                self.sl_order_id = None

        except Exception as e:
            print(f"❌ 포지션 모니터링 오류: {e}")
