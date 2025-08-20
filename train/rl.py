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

# ===== Core Config =====
FEE_RATE      = 0.0005       # 왕복 구조 맞춰 조정
TRAIN_SLIP_BP = 0.0          # 훈련 초반 슬리피지 0 (추후 단계 주입)
MIN_HOLD      = 6            # 최소 보유 6틱(=30분; 5m 기준)
COOLDOWN      = 2            # 청산 후 2틱 쿨다운
CONF_ENTER    = 0.70         # 매니저 확신 상향 임계(히스테리시스)
CONF_EXIT     = 0.50         # 매니저 확신 하향 임계

# Manager 보상 가중치(단순화 버전)
M_W1 = 1.0
M_W3 = 0.5
M_FLIP = 0.2

# ==== 관망 유도/억제 미세 보상 ====
M_IDLE   = 5e-5   # dir=0(관망)일 때 아주 작은 세금
M_COMMIT = 1e-4   # |dir|=1(방향 선택) 시 작은 보너스

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

    # OHLC (보상·신호용)
    ohlc5  = df5[["Open","High","Low","Close"]]
    ohlc4h = df4h[["Open","High","Low","Close"]].reindex(df5.index, method="ffill")

    # 4h 방향 참고(간단 rule): Heikin-Ashi
    ha_c_4h, ha_o_4h = _heikin_ashi_ohlc(ohlc4h)
    reg4h_sign = np.sign(ha_c_4h - ha_o_4h).astype(int)

    return dict(
        X=X,
        price=ohlc5["Close"].astype(float),
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
    reg4h_sign = np.sign(ha_c_4h - ha_o_4h).astype(int)

    return dict(
        XH=XH,
        price=df1h["Close"].astype(float),
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

# ===== Worker Env (5m) — 거래단위 보상 & Flip 금지 =====
class WorkerEnv(gym.Env):
    """
    Obs: [X_5m.. + goal_dir, goal_conf]
    Act: 0 Hold, 1 EnterLong, 2 EnterShort, 3 Exit
    Rules:
      - pos==0: {Hold, Enter}만. (매니저 마스크: conf<enter면 Enter 금지, dir에 반하는 Enter 금지)
      - pos!=0: {Hold, Exit}만. (Flip 금지)
      - min_hold, cooldown 적용
    Reward:
      - Entry: 보상 0 (수수료는 나중에 정산용으로 기록)
      - Hold:  보상 0
      - Exit:  (exit/entry - 1)*sign - (entry_fee + exit_fee)  → 그 순간 1회 지급
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str, gb: GoalBridge,
                 fee_rate=FEE_RATE, slip_bp=TRAIN_SLIP_BP,
                 min_hold=MIN_HOLD, cooldown=COOLDOWN,
                 conf_enter=CONF_ENTER, conf_exit=CONF_EXIT):
        super().__init__()
        self.gb = gb
        data = build_worker_inputs(split)
        self.X = data["X"]; self.price = data["price"]
        self.regsign = data["reg4h_sign"].astype(int)

        self.idx = self.X.index
        self.t = 0

        # trading state
        self.pos = 0               # -1,0,1
        self.entry_px = None
        self.entry_fee = 0.0
        self.hold_ticks = 0
        self.cooldown = 0

        # params
        self.fee_rate = float(fee_rate)
        self.slip_bp = float(slip_bp)
        self.min_hold = int(min_hold)
        self.cooldown_reset = int(cooldown)
        self.conf_enter = float(conf_enter)
        self.conf_exit  = float(conf_exit)

        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(self.X.shape[1] + 2,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    # 합법 액션 마스크 생성
    def _legal_mask(self, dir_: int, conf: float) -> np.ndarray:
        m = np.zeros(4, dtype=np.int8)
        if self.pos == 0:
            if self.cooldown > 0 or conf < self.conf_enter or dir_ == 0:
                m[0] = 1  # Hold만
            else:
                m[0] = 1
                if dir_ > 0: m[1] = 1  # EnterLong
                if dir_ < 0: m[2] = 1  # EnterShort
        else:
            # 보유 중: Exit만 (Flip 금지)
            if self.hold_ticks >= self.min_hold:
                m[0] = 1; m[3] = 1
            else:
                m[0] = 1
        return m

    def _obs(self) -> np.ndarray:
        x = self.X.iloc[self.t].to_numpy(dtype=np.float32, copy=False)
        g = self.gb.vec()
        return np.concatenate([x, g], axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = 0
        self.entry_px = None
        self.entry_fee = 0.0
        self.hold_ticks = 0
        self.cooldown = 0
        return self._obs(), {}

    def step(self, a: int):
        px = float(self.price.iloc[self.t])
        dir_, conf = int(self.gb.vec()[0]), float(self.gb.vec()[1])

        # 히스테리시스: conf 낮아지면 dir=0처럼 행동(Enter 금지)
        if conf < self.conf_exit:
            dir_eff = 0
        else:
            dir_eff = dir_

        legal = self._legal_mask(dir_eff, conf)
        if legal[a] == 0:
            a = 0  # 불법 액션은 Hold로 대체 (패널티 없음)

        r = 0.0

        # === 상태 전이 ===
        if self.pos == 0:
            # Enter?
            if a == 1 or a == 2:
                signed = 1 if a == 1 else -1
                exec_px = px * (1 + signed * self.slip_bp * 1e-4)
                self.pos = signed
                self.entry_px = exec_px
                self.entry_fee = self.fee_rate * px
                self.hold_ticks = 0
                # 보상은 0 (정산은 청산 시)
        else:
            # Hold / Exit?
            if a == 3 and self.hold_ticks >= self.min_hold:
                # Exit 실행
                exec_px = px * (1 - self.pos * self.slip_bp * 1e-4)
                pnl = (exec_px - self.entry_px) / self.entry_px * self.pos
                exit_fee = self.fee_rate * px
                r = pnl - (self.entry_fee + exit_fee)

                # reset
                self.pos = 0
                self.entry_px = None
                self.entry_fee = 0.0
                self.hold_ticks = 0
                self.cooldown = self.cooldown_reset
            else:
                # 보유 지속
                self.hold_ticks += 1

        # 쿨다운 감소
        if self.cooldown > 0 and self.pos == 0:
            self.cooldown -= 1

        # time step
        self.t += 1
        terminated = (self.t >= len(self.X) - 1)

        # 에피소드 강제 청산(마지막 스텝)
        if terminated and self.pos != 0:
            exec_px = px  # 마지막 가격으로 청산
            pnl = (exec_px - self.entry_px) / self.entry_px * self.pos
            exit_fee = self.fee_rate * px
            r += pnl - (self.entry_fee + exit_fee)
            # reset
            self.pos = 0
            self.entry_px = None
            self.entry_fee = 0.0
            self.hold_ticks = 0
            self.cooldown = 0

        info = {
            "dir": dir_eff, "conf": conf, "legal": legal,
            "pos": self.pos, "cooldown": self.cooldown, "hold": self.hold_ticks
        }
        return self._obs(), float(r), terminated, False, info

# ===== Manager Env (1h) — 단순 방향 학습 + 관망세 억제 =====
class ManagerEnv(gym.Env):
    """
    Obs: [X_1h + 4h(ffill) scaled]
    Act: MultiDiscrete([3, 11])  → dir ∈ {-1,0,1}, conf ∈ {0..10}/10
    Reward: w1*dir*r_1h + w3*dir*r_3h - flip_penalty + (idle/commit micro)
    (레짐 미스매치/weak 필터는 초기 학습에서는 사용하지 않음)
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str, gb: GoalBridge):
        super().__init__()
        data = build_manager_inputs(split)
        self.XH = data["XH"]
        self.price = data["price"]
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

        # goal set
        self.gb.set(Goal(dir_raw, conf))

        cur = float(self.price.iloc[self.t])
        nxt1 = float(self.price.iloc[min(self.t + 1, len(self.price) - 1)])
        nxt3 = float(self.price.iloc[min(self.t + 3, len(self.price) - 1)])
        r1 = (nxt1 - cur) / cur
        r3 = (nxt3 - cur) / cur

        R_dir  = M_W1 * dir_raw * r1 + M_W3 * dir_raw * r3
        R_flip = -M_FLIP * int(dir_raw != self.prev_dir and dir_raw != 0)

        # 관망 세금 & 방향 커밋 보너스
        R_idle   = -M_IDLE if dir_raw == 0 else 0.0
        R_commit =  M_COMMIT if dir_raw != 0 else 0.0

        R = R_dir + R_flip + R_idle + R_commit

        self.prev_dir = dir_raw if dir_raw != 0 else self.prev_dir
        self.t += 1
        terminated = (self.t >= len(self.XH) - 2)
        info = {"dir": dir_raw, "conf": conf}
        return self._obs(), float(R), terminated, False, info

# ===== Entropy Decay (탐색 → 수렴) =====
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

# ===== Manager Runner (inference for guided worker) =====
class _ManagerRunner:
    def __init__(self, split: str, manager_ckpt: str, conf_enter=CONF_ENTER, conf_exit=CONF_EXIT):
        self.m = build_manager_inputs(split)
        self.idx = self.m["XH"].index
        self.k = 0
        self.prev_dir = 0
        self.conf_enter = conf_enter
        self.conf_exit = conf_exit
        # Load PPO manager (obs normalization 안 씀: norm_obs=False로 학습)
        self.model = PPO.load(manager_ckpt, device="cpu")

    def _obs_at_k(self) -> np.ndarray:
        return self.m["XH"].iloc[self.k].to_numpy(dtype=np.float32, copy=False)

    def tick(self, ts_5m) -> Goal:
        th = pd.Timestamp(ts_5m).floor("1h")
        while self.k + 1 < len(self.idx) and self.idx[self.k + 1] <= th:
            self.k += 1
        obs = self._obs_at_k()
        act, _ = self.model.predict(obs, deterministic=True)
        dir_raw = [-1, 0, 1][int(act[0])]
        conf = float(int(act[1]) / 10.0)

        # 히스테리시스
        if conf >= self.conf_enter:
            dir_eff = dir_raw
        elif conf <= self.conf_exit:
            dir_eff = 0
        else:
            dir_eff = self.prev_dir

        self.prev_dir = dir_eff
        return Goal(dir_eff, conf)

# ===== Training Orchestration =====
def train_manager(split: str = "train", steps: int = 800_000, seed: int = 42,
                  save_path: str | None = None):
    gb = GoalBridge()
    env = DummyVecEnv([lambda: ManagerEnv(split=split, gb=gb)])
    # 보상 클리핑 강화: 1.0
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=1.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=256, batch_size=256, device="cpu",
        learning_rate=3e-4,      # 1e-4 → 3e-4
        gamma=0.95,
        ent_coef=0.03,           # 0.005 → 0.03 (초반 탐색 강화)
        clip_range=0.2, gae_lambda=0.95,
        vf_coef=1.2, clip_range_vf=0.2,
        seed=seed, verbose=1
    )
    # 엔트로피 선형 감소: 0.03 → 0.005
    model.learn(total_timesteps=steps, callback=EntropyDecay(start=0.03, end=0.005, decay_steps=steps))

    sp = save_path or os.path.join(MODEL_DIR, "manager_stage1.zip")
    model.save(sp)
    env.save(os.path.join(MODEL_DIR, "manager_stage1_vecnorm.pkl"))
    return sp

def train_worker_guided(split: str = "train",
                        manager_ckpt: str | None = None,
                        steps: int = 1_000_000, seed: int = 42,
                        save_path: str | None = None,
                        min_hold: int = MIN_HOLD, cooldown: int = COOLDOWN,
                        conf_enter: float = CONF_ENTER, conf_exit: float = CONF_EXIT,
                        slip_bp: float = TRAIN_SLIP_BP):
    """
    매니저 모델이 내는 (dir, conf)로 워커를 가이드하여 학습.
    - Enter는 conf>=conf_enter & dir!=0 일 때만 허용
    - Exit는 min_hold 이후 허용
    - Flip 금지, cooldown 적용
    - 보상은 청산 시 1회
    """
    assert manager_ckpt and os.path.exists(manager_ckpt), f"manager ckpt not found: {manager_ckpt}"

    class _GuidedGB(GoalBridge):
        def __init__(self, runner: _ManagerRunner, idx5):
            super().__init__()
            self.runner = runner
            self.idx5 = idx5
        def tick(self, t):
            ts = self.idx5[t]
            g = self.runner.tick(ts)
            self.set(g)

    # Harness: 매 step마다 매니저 예측으로 goal 갱신
    mr = _ManagerRunner(split, manager_ckpt, conf_enter, conf_exit)
    class _Harness(WorkerEnv):
        def __init__(self, split, gb):
            super().__init__(split=split, gb=gb,
                             min_hold=min_hold, cooldown=cooldown,
                             conf_enter=conf_enter, conf_exit=conf_exit,
                             slip_bp=slip_bp)
        def step(self, a: int):
            gb.tick(self.t)
            return super().step(a)

    # VecNormalize: 보상만 정규화 (가치함수 안정화)
    gb = _GuidedGB(mr, build_worker_inputs(split)["X"].index)
    env = DummyVecEnv([lambda: _Harness(split=split, gb=gb)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=5.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=8192, batch_size=2048, device="cpu",
        learning_rate=3e-4, gamma=0.99,
        ent_coef=0.05, clip_range=0.2, gae_lambda=0.95,
        vf_coef=1.2, clip_range_vf=0.2,
        seed=seed, verbose=1
    )
    model.learn(total_timesteps=steps, callback=EntropyDecay(start=0.05, end=0.01, decay_steps=steps))

    sp = save_path or os.path.join(MODEL_DIR, "worker_stage1.zip")
    model.save(sp)
    env.save(os.path.join(MODEL_DIR, "worker_stage1_vecnorm.pkl"))
    return sp

# ===== Simple runners =====
def run_manager():
    print("[HRL] Training Manager first…")
    mp = train_manager()
    print(f"[OK] Manager saved → {mp}")

def run_worker_guided_with(mp: str):
    print("[HRL] Training Worker (guided by Manager)…")
    wp = train_worker_guided(manager_ckpt=mp)
    print(f"[OK] Worker saved → {wp}")

def run_all():
    print("[HRL] Manager → Worker(guided)")
    mp = train_manager()
    run_worker_guided_with(mp)

if __name__ == "__main__":
    # 기본 실행: 매니저 먼저 → 워커(가이드) 순서
    run_all()
