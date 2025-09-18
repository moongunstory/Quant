# train/reinforce/core/crypto_trading_env.py

import numpy as np

class CryptoTradingEnv:
    def __init__(
        self,
        data: dict,
        seq_lens: dict,
        maker_fee=0.0002,
        taker_fee=0.0005,
        take_profit_pct=0.02,
        stop_loss_pct=0.01
    ):
        """
        data: {
            "ohlcv": (N, dim),
            "funding": (N, dim),
            ...
        }
        seq_lens: {
            "ohlcv": 48,
            "funding": 7,
            "dune": 7,
            ...
        }
        """
        self.data = data
        self.seq_lens = seq_lens
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct

        self.length = len(next(iter(data.values())))  # assumes all have same length
        self.init_cash = 1.0  # 시작 포트폴리오 가치 (기본 1.0)
        self.reset()

    def reset(self):
        self.t = max(self.seq_lens.values())
        self.position = 0.0
        self.entry_price = None
        self.done = False
        self.portfolio_value = self.init_cash
        self.portfolio_history = [self.init_cash]
        return self._get_obs()

    def _get_obs(self):
        obs = {
            k: v[self.t - self.seq_lens[k]:self.t] for k, v in self.data.items()
        }
        return obs

    def step(self, action: float, is_forced_exit=False):
        action = np.clip(action, -1, 1)
        price_now = self.data["ohlcv"][self.t][-1]
        reward = 0.0
        info = {"tp_hit": False, "sl_hit": False}

        # --- SL/TP 자동 청산 감지 ---
        if self.position != 0 and self.entry_price is not None:
            tp_price = self.entry_price * (1 + self.take_profit_pct * np.sign(self.position))
            sl_price = self.entry_price * (1 - self.stop_loss_pct * np.sign(self.position))

            if (self.position > 0 and (price_now >= tp_price or price_now <= sl_price)) or \
            (self.position < 0 and (price_now <= tp_price or price_now >= sl_price)):
                price_diff = price_now / self.entry_price - 1
                reward += self.position * price_diff
                reward -= self.maker_fee

                self.portfolio_value *= (1 + self.position * price_diff)
                self.portfolio_history.append(float(self.portfolio_value))

                info["tp_hit"] = price_now >= tp_price if self.position > 0 else price_now <= tp_price
                info["sl_hit"] = not info["tp_hit"]

                self.position = 0.0
                self.entry_price = None
                self.t += 1
                self.done = self.t >= self.length
                return self._get_obs(), reward, self.done, info

        # --- 일반 로직 ---
        if self.position == 0 and action != 0:
            reward -= self.taker_fee
            self.entry_price = price_now

        elif self.position != 0 and action == 0:
            reward -= self.taker_fee

        elif self.position != 0 and self.position != action:
            reward -= self.taker_fee * 2
            self.entry_price = price_now

        price_diff = 0
        if self.position != 0:
            price_diff = price_now / self.entry_price - 1
            reward += self.position * price_diff

        if is_forced_exit and self.position != 0:
            price_diff = price_now / self.entry_price - 1
            reward += self.position * price_diff
            reward -= self.taker_fee
            self.position = 0.0
            self.entry_price = None

        self.portfolio_value *= (1 + self.position * price_diff if self.position != 0 else 1)
        self.portfolio_history.append(float(self.portfolio_value))  # ← 여기도 float 처리

        self.position = action if not is_forced_exit else 0.0
        self.t += 1
        self.done = self.t >= self.length

        return self._get_obs(), reward, self.done, info

    def render(self):
        print(f"t={self.t}, position={self.position}")
        if self.position != 0 and self.entry_price is not None:
            tp = self.entry_price * (1 + self.take_profit_pct * np.sign(self.position))
            sl = self.entry_price * (1 - self.stop_loss_pct * np.sign(self.position))
            print(f"  Entry Price: {self.entry_price:.2f}")
            print(f"  TP Target  : {tp:.2f}")
            print(f"  SL Target  : {sl:.2f}")
