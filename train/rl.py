# ai_binance/train/rl_hrl.py
from __future__ import annotations
import os, json
from typing import List, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "model"))
os.makedirs(MODEL_DIR, exist_ok=True)

# ===== Core Config (전액 진입/전액 청산, 예산 없음) =====
FEE_RATE   = 0.0005     # 수수료 (교차/왕복 구조에 맞춰 조정)
SLIP_BP    = 1.0        # 슬리피지(bp)
TIMING_K   = 4          # 5m 기준 20분
TIMING_K_COEF = 0.6     # 타점 보상 강화
TURN_PENALTY  = 1.5 * FEE_RATE
FLIP_PENALTY  = 3.0 * FEE_RATE

# Manager 보상 가중치
M_W1 = 1.0
M_W3 = 0.5
M_FLIP = 0.2
M_MIS  = 0.2

# Worker 탐색/게이트 워밍업
GATE_WARMUP_STEPS = 20_000
ALIGN_EPS         = 0.08
OPPORTUNITY_COST  = 0.50 * FEE_RATE
GATE_SOFT_PENALTY_MULT = 1.0   # ← 15m 게이트 불일치 시, fee_rate의 배수만큼 패널티

# ===== Utils =====
def _path_fe(split: str, tf: str) -> str:
    fn = f"fe_{split}_{tf}.parquet" if tf != "btc1h" else f"fe_{split}_btc1h.parquet"
    return os.path.join(PROC_DIR, fn)

def _load_fe(split: str, tf: str) -> pd.DataFrame:
    p = _path_fe(split, tf)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

def _feat_list(tf: str) -> List[str]:
    if tf == "btc1h":
        return []
    p = os.path.join(PROC_DIR, f"fe_feature_list_{tf}.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _zscore(s: pd.Series, win: int = 100) -> pd.Series:
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std().replace(0, np.nan)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def _heikin_ashi_ohlc(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    O = df["Open"].astype(float).to_numpy()
    H = df["High"].astype(float).to_numpy()
    L = df["Low"].astype(float).to_numpy()
    C = df["Close"].astype(float).to_numpy()
    n = len(df)
    HA_C = (O+H+L+C)/4.0
    HA_O = np.empty(n, dtype=float)
    HA_O[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n):
        HA_O[i] = (HA_O[i-1] + HA_C[i-1]) / 2.0
    return pd.Series(HA_C, index=df.index), pd.Series(HA_O, index=df.index)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# ===== Data assembly =====
def build_worker_inputs(split: str):
    df5  = _load_fe(split, "5m")
    df15 = _load_fe(split, "15m")
    df1h = _load_fe(split, "1h")
    df4h = _load_fe(split, "4h")
    btc1 = _load_fe(split, "btc1h")

    f5  = df5[_feat_list("5m")]
    f15 = df15[_feat_list("15m")].reindex(df5.index, method="ffill")
    f1h = df1h[_feat_list("1h")].reindex(df5.index, method="ffill")
    f4h = df4h[_feat_list("4h")].reindex(df5.index, method="ffill")
    cols_b = [c for c in btc1.columns if c.endswith("_btc1h")]
    fb  = btc1[cols_b].reindex(df5.index, method="ffill")

    X = pd.concat([f5, f15, f1h, f4h, fb], axis=1).replace([np.inf,-np.inf],0.0).fillna(0.0)

    # OHLC for rewards & gates
    ohlc5  = df5[["Open","High","Low","Close"]]
    ohlc15 = df15[["Open","High","Low","Close"]].reindex(df5.index, method="ffill")
    ohlc4h = df4h[["Open","High","Low","Close"]].reindex(df5.index, method="ffill")

    # 15m gate (HA_BC sign)
    ha_c_15, ha_o_15 = _heikin_ashi_ohlc(ohlc15)
    gate15_sign = np.sign(ha_c_15 - ha_o_15).astype(int)

    # 4h regime (완화된 약레짐 기준)
    ha_c_4h, ha_o_4h = _heikin_ashi_ohlc(ohlc4h)
    bc4h  = ha_c_4h - ha_o_4h
    atr4h = _atr(ohlc4h)
    weak4h = (bc4h.abs() < (bc4h.abs().rolling(200, min_periods=50).median()*0.15).fillna(np.inf)) \
             & (_zscore(atr4h, 200) < -0.8)
    reg4h_sign = np.sign(bc4h).astype(int)

    return dict(
        X=X,
        price=ohlc5["Close"].astype(float),
        gate15_sign=gate15_sign.reindex(df5.index).fillna(0),
        reg4h_weak=weak4h.reindex(df5.index).fillna(True),
        reg4h_sign=reg4h_sign.reindex(df5.index).fillna(0)
    )

def build_manager_inputs(split: str):
    df1h = _load_fe(split, "1h")
    df4h = _load_fe(split, "4h")

    XH = pd.concat(
        [df1h[_feat_list("1h")], df4h[_feat_list("4h")].reindex(df1h.index, method="ffill")],
        axis=1
    ).replace([np.inf,-np.inf],0.0).fillna(0.0)

    ohlc4h = df4h[["Open","High","Low","Close"]]
    ha_c_4h, ha_o_4h = _heikin_ashi_ohlc(ohlc4h)
    bc4h  = ha_c_4h - ha_o_4h
    atr4h = _atr(ohlc4h)
    weak4h = (bc4h.abs() < (bc4h.abs().rolling(200, min_periods=50).median()*0.15).fillna(np.inf)) \
             & (_zscore(atr4h, 200) < -0.8)
    reg4h_sign = np.sign(bc4h).astype(int)

    return dict(
        XH=XH,
        price=df1h["Close"].astype(float),
        reg4h_weak=weak4h.reindex(XH.index, method="ffill").fillna(True),
        reg4h_sign=reg4h_sign.reindex(XH.index, method="ffill").fillna(0)
    )

# ===== Goal Bridge =====
class Goal:
    __slots__ = ("dir","conf")
    def __init__(self, dir_: int = 0, conf: float = 0.0):
        self.dir = int(np.sign(dir_))
        self.conf = float(np.clip(conf, 0.0, 1.0))

class GoalBridge:
    def __init__(self):
        self.cur = Goal(0, 0.0)
    def set(self, g: Goal): self.cur = g
    def vec(self) -> np.ndarray:
        return np.array([self.cur.dir, self.cur.conf], dtype=np.float32)

# ===== Worker Env (5m) =====
class WorkerEnv(gym.Env):
    """
    Obs: [X_5m+15m+1h+4h+btc1h (scaled), goal_dir, goal_conf]
    Act: 0 Hold, 1 Long, 2 Short, 3 Flat
    Mask: dir=+1 → {Hold,Long,Flat}, dir=-1 → {Hold,Short,Flat}, dir=0 → {Hold,Flat}
    Gate: (소프트) 15m 방향 불일치면 패널티만 부여, 차단하지 않음
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str, gb: GoalBridge,
                 fee_rate=FEE_RATE, slip_bp=SLIP_BP):
        super().__init__()
        self.gb = gb
        data = build_worker_inputs(split)
        self.X = data["X"]; self.price = data["price"]
        self.gate15 = data["gate15_sign"].astype(int)
        self.regweak = data["reg4h_weak"].astype(bool)
        self.regsign = data["reg4h_sign"].astype(int)

        self.idx = self.X.index
        self.t = 0
        self.pos = 0  # -1,0,1
        self.fee_rate = fee_rate
        self.slip_bp = slip_bp

        # exploration helpers
        self.total_steps = 0
        self.gate_warmup_steps = GATE_WARMUP_STEPS
        self.align_eps = ALIGN_EPS
        self.opp_cost = OPPORTUNITY_COST

        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(self.X.shape[1] + 2,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _mask(self, d: int) -> np.ndarray:
        if d > 0: return np.array([1,1,0,1], dtype=np.int8)
        if d < 0: return np.array([1,0,1,1], dtype=np.int8)
        return np.array([1,0,0,1], dtype=np.int8)

    def _obs(self) -> np.ndarray:
        x = self.X.iloc[self.t].to_numpy(dtype=np.float32, copy=False)
        g = self.gb.vec()
        return np.concatenate([x, g], axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0; self.pos = 0
        return self._obs(), {}

    def step(self, a: int):
        px = float(self.price.iloc[self.t])

        dir_, conf = int(self.gb.vec()[0]), float(self.gb.vec()[1])
        mask = self._mask(dir_)
        pen = 0.0
        if mask[a] == 0:
            a = 0; pen -= 0.01

        gate_active = (self.total_steps >= self.gate_warmup_steps)

        # 15m gate: 워밍업 이후엔 불일치에 '패널티'만 부여(하드 차단 금지)
        wants_entry = (a in (1, 2)) and (self.pos == 0)
        if gate_active and wants_entry:
            gsig = int(self.gate15.iloc[self.t])
            mismatch = (dir_ > 0 and gsig <= 0) or (dir_ < 0 and gsig >= 0)
            if mismatch:
                pen -= GATE_SOFT_PENALTY_MULT * self.fee_rate  # ← 소프트 패널티만

        target = {0: self.pos, 1: 1, 2: -1, 3: 0}[a]
        changed = int(target != self.pos)
        flip = int(abs(target - self.pos) == 2)

        fee = self.fee_rate * abs(target - self.pos) * px if changed else 0.0
        signed = 0 if a in (0, 3) else (1 if a == 1 else -1)
        exec_px = px * (1 + signed * self.slip_bp * 1e-4) if changed else px

        nxt_px = float(self.price.iloc[min(self.t + 1, len(self.price) - 1)])
        pnl = (nxt_px - exec_px) / exec_px * self.pos

        # 타이밍 품질
        k_idx = min(self.t + TIMING_K, len(self.price) - 1)
        fut_px = float(self.price.iloc[k_idx])
        dret = (fut_px - px) / px
        timing = (dret if a == 1 else -dret) if a in (1, 2) else 0.0

        # reward 구성
        r = pnl - fee + TIMING_K_COEF * timing \
            - TURN_PENALTY * changed - FLIP_PENALTY * flip + pen

        # ① 소프트 방향 정렬(포지션 유무 관계없이 약하게)
        align = self.align_eps * dir_ * ((nxt_px - px) / px)
        r += align

        # ② 기회비용: 게이트 통과+방향 있는데 Hold 선택
        if gate_active and dir_ != 0 and a == 0 and self.t > 0:
            gsig_prev = int(self.gate15.iloc[self.t - 1])
            if (dir_ > 0 and gsig_prev > 0) or (dir_ < 0 and gsig_prev < 0):
                r -= self.opp_cost

        # ③ entry bonus: 방향 일치 신규 진입에 소액 보상
        if a in (1, 2) and self.pos == 0:
            if (a == 1 and dir_ > 0) or (a == 2 and dir_ < 0):
                entry_bonus = 0.002 + 0.5 * abs(dret)
                r += entry_bonus

        # transition
        self.pos = target
        self.t += 1
        self.total_steps += 1
        terminated = (self.t >= len(self.X) - 2)
        info = {
            "fee": fee, "mask": mask, "dir": dir_,
            "gate15": int(self.gate15.iloc[self.t - 1]),
            "trade": int(changed), "flip": int(flip)
        }
        return self._obs(), float(r), terminated, False, info

# ===== Manager Env (1h) =====
class ManagerEnv(gym.Env):
    """
    Obs: [X_1h + 4h(ffill) scaled]
    Act: MultiDiscrete([3, 11])  → dir ∈ {-1,0,1}, conf ∈ {0..10}/10
    Rule: 4h 레짐 weak면 dir은 강제로 0 (거래 금지), 해당 시도는 소액 패널티
    Reward: 방향 정확도(1h,3h) - flip 패널티 - 4h 상충 패널티
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str, gb: GoalBridge):
        super().__init__()
        data = build_manager_inputs(split)
        self.XH = data["XH"]
        self.price = data["price"]
        self.regweak = data["reg4h_weak"].astype(bool)
        self.regsign = data["reg4h_sign"].astype(int)
        self.idx = self.XH.index
        self.t = 0
        self.gb = gb
        self.prev_dir = 0

        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(self.XH.shape[1],),
                                            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 11])

    def _obs(self) -> np.ndarray:
        return self.XH.iloc[self.t].to_numpy(dtype=np.float32, copy=False)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0; self.prev_dir = 0
        self.gb.set(Goal(0, 0.0))
        return self._obs(), {}

    def step(self, a: np.ndarray):
        dir_raw = [-1, 0, 1][int(a[0])]
        conf = float(int(a[1]) / 10.0)

        weak = bool(self.regweak.iloc[self.t])
        tried_nonflat = (dir_raw != 0)
        dir_eff = 0 if weak else dir_raw
        pen = -0.02 if (weak and tried_nonflat) else 0.0

        self.gb.set(Goal(dir_eff, conf))

        cur = float(self.price.iloc[self.t])
        nxt1 = float(self.price.iloc[min(self.t + 1, len(self.price) - 1)])
        nxt3 = float(self.price.iloc[min(self.t + 3, len(self.price) - 1)])
        r1 = (nxt1 - cur) / cur
        r3 = (nxt3 - cur) / cur

        R_dir  = M_W1 * dir_eff * r1 + M_W3 * dir_eff * r3
        R_flip = -M_FLIP * int(dir_eff != self.prev_dir)
        R_mis  = -M_MIS  * int(np.sign(dir_eff) != int(self.regsign.iloc[self.t]) and dir_eff != 0)
        R = R_dir + R_flip + R_mis + pen

        self.prev_dir = dir_eff
        self.t += 1
        terminated = (self.t >= len(self.XH) - 2)
        info = {"weak4h": weak, "reg4h": int(self.regsign.iloc[self.t - 1]), "dir": dir_eff}
        return self._obs(), float(R), terminated, False, info

# ===== Entropy Decay Callback (탐색 → 수렴) =====
class EntropyDecay(BaseCallback):
    def __init__(self, start=0.10, end=0.03, decay_steps=600_000, verbose=0):
        super().__init__(verbose)
        self.start, self.end, self.decay = start, end, decay_steps
    def _on_training_start(self):
        self.model.ent_coef = self.start
    def _on_step(self):
        step = self.num_timesteps
        frac = min(1.0, step / self.decay)
        self.model.ent_coef = float(self.start + (self.end - self.start) * frac)
        return True

# ===== Training Orchestration =====
def train_worker(split: str = "train", steps: int = 1_000_000, seed: int = 42, save_path: str | None = None):
    """
    Worker 예열: 휴리스틱 Manager goal 주입.
    ★ 약레짐 무시: 4h 약레짐도 방향(부호)만 따라가게 하여 초기 탐색 확보.
    """
    class _HeuristicGB(GoalBridge):
        def __init__(self, split):
            super().__init__()
            self.m = build_manager_inputs(split)
            self.idx = self.m["XH"].index
            self.k = 0
        def tick(self, ts_5m):
            th = pd.Timestamp(ts_5m).floor("1h")
            while self.k + 1 < len(self.idx) and self.idx[self.k + 1] <= th:
                self.k += 1
            reg = int(self.m["reg4h_sign"].iloc[self.k])  # ★ 약레짐 무시
            self.set(Goal(reg, 0.5))

    gb = _HeuristicGB(split)
    class _Harness(WorkerEnv):
        def step(self, a: int):
            gb.tick(self.idx[self.t])
            return super().step(a)

    # VecNormalize: 보상만 정규화 (가치함수 안정화)
    env = DummyVecEnv([lambda: _Harness(split=split, gb=gb)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=2048,
        batch_size=1024,
        device="cpu",
        learning_rate=3e-4, gamma=0.99,
        ent_coef=0.10, clip_range=0.2, gae_lambda=0.95,
        vf_coef=1.2,            # ← 가치망 비중 강화
        seed=seed, verbose=1
    )
    model.learn(total_timesteps=steps, callback=EntropyDecay())

    sp = save_path or os.path.join(MODEL_DIR, "worker_stage1.zip")
    model.save(sp)
    env.save(os.path.join(MODEL_DIR, "worker_stage1_vecnorm.pkl"))
    return sp

def train_manager(split: str = "train", steps: int = 800_000, seed: int = 42, save_path: str | None = None):
    gb = GoalBridge()

    # VecNormalize: 보상만 정규화
    env = DummyVecEnv([lambda: ManagerEnv(split=split, gb=gb)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=256,
        batch_size=256,
        device="cpu",
        learning_rate=1e-4, gamma=0.95,
        ent_coef=0.005, clip_range=0.2, gae_lambda=0.95,
        vf_coef=1.2,            # ← 가치망 비중 강화
        seed=seed, verbose=1
    )
    model.learn(total_timesteps=steps)

    sp = save_path or os.path.join(MODEL_DIR, "manager_stage1.zip")
    model.save(sp)
    env.save(os.path.join(MODEL_DIR, "manager_stage1_vecnorm.pkl"))
    return sp

# ===== Simple runners (no CLI) =====
def run_worker():
    print("[HRL] Training Worker…")
    wp = train_worker()
    print(f"[OK] Worker saved → {wp}")

def run_manager():
    print("[HRL] Training Manager…")
    mp = train_manager()
    print(f"[OK] Manager saved → {mp}")

def run_all():
    run_worker()
    run_manager()

if __name__ == "__main__":
    # 기본 실행: 워커 → 매니저 순으로 학습
    run_all()
