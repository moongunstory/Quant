# rl_mtf_practitioner.py — PPO for Crypto Futures (Multi-TimeFrame, cold-start safe + anti-collapse)
# - MTF 입력(5m, 15m, 1h, 4h), 기준 타임라인=5m, 다음봉 체결(1봉 레이턴시)
# - 액션: a∈[-1,1] = 목표 포지션(연속)
# - 보상: pos*ret − (fee+slip)*|Δpos| − pos*funding_step − λ·excess_turnover
# - 비용: 포지션 변경 시 1회만, 펀딩=Funding8h/96 per 5m
# - 회전 예산: 1일(≈288스텝) 초과분에 라그랑주 패널티(λ 자동 상향)
# - 근본 해결:
#   (1) 각 TF가 window 길이만큼 유효 피처가 쌓인 뒤의 공통 시점부터만 사용(bfill 금지, 누수 차단)
#   (2) 탐험 유지(SDE/entropy/log_std_init) + 회전 패널티 커리큘럼
#   (3) 데드존 제거 + ACTION_GAIN으로 det(평균정책)에서도 충분한 포지션 출력
from __future__ import annotations

import os, json, warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

# ===== Paths =====
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROC_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
MODEL_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "model"))
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "reports"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ===== Multi-TimeFrame Setup =====
TIMEFRAMES    = ["5m", "15m", "1h", "4h"]
BASE_INTERVAL = "5m"  # execution timeline

# ===== Hyperparams =====
SEED = 42
WINDOWS = {"5m": 48, "15m": 32, "1h": 24, "4h": 12}

# Market/exec costs
FEE_BPS   = 0.0006
SLIP_BPS  = 0.0003
MIN_DPOS  = 0.00        # 데드존 제거
COOLDOWN  = 0
LEVERAGE  = 1.0
SMOOTH_ALPHA = 0.35     # 목표로 더 빠르게 수렴
ACTION_GAIN  = 2.0      # 정책 출력 증폭 (det에서도 포지션 확보)

# Turnover budget & Lagrange
TURN_BUDGET_DAILY = 3.0
LAMBDA_INIT = 0.0
LAMBDA_STEP = 1e-5
LAMBDA_MAX  = 25.0

# Train loop
EVAL_EVERY  = 50_000
TOTAL_STEPS = 1_000_000

# ===== Data loading =====
def _load_fe_mtf(split: str) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Load processed features for all TFs. Uses 5m feature_list as canonical."""
    with open(os.path.join(PROC_DIR, f"fe_feature_list_{BASE_INTERVAL}.json"), "r", encoding="utf-8") as f:
        feat_cols = json.load(f)

    mtf_data: Dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        path = os.path.join(PROC_DIR, f"fe_{split}_{tf}.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed file missing: {path}")
        df = pd.read_parquet(path).sort_index()

        # Required refs
        if "FundingRate" not in df.columns:
            df["FundingRate"] = 0.0
        if "Funding8h" not in df.columns:
            df["Funding8h"] = df["FundingRate"]
        if "Close" not in df.columns:
            if "close_ref" in df.columns:
                df["Close"] = df["close_ref"]
            else:
                raise ValueError(f"{tf}: neither Close nor close_ref present")

        # Optional funding settle marker
        if "FundingSettle" not in df.columns:
            idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index, utc=True)
            df["FundingSettle"] = (((idx.hour % 8 == 0) & (idx.minute == 0))).astype("int8")

        df = df.replace([np.inf, -np.inf], np.nan)
        mtf_data[tf] = df

    return mtf_data, feat_cols

# ===== Cold-start safe alignment =====
def _first_ready_ts(df: pd.DataFrame, feat_cols_tf: List[str], window: int) -> pd.Timestamp:
    """Earliest timestamp where last `window` rows of all used features are finite."""
    cols = [c for c in feat_cols_tf if c in df.columns]
    if not cols:
        raise ValueError("No matching features for timeframe")
    arr = df[cols].to_numpy(dtype=float, copy=False)
    finite_row = np.isfinite(arr).all(axis=1)
    mask = pd.Series(finite_row, index=df.index)
    ready = mask.rolling(window, min_periods=window).apply(lambda x: 1.0 if bool(np.all(x)) else 0.0)
    ts = ready[ready == 1.0].index.min()
    if ts is None:
        raise ValueError("No valid warm-up segment for timeframe")
    return ts

def _align_timeframes(mtf_data: Dict[str, pd.DataFrame], feat_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """Cut all TFs to start only after their own warmups; ffill to 5m grid; no bfill."""
    ready_points: Dict[str, pd.Timestamp] = {}
    for tf in TIMEFRAMES:
        ready_points[tf] = _first_ready_ts(mtf_data[tf], feat_cols, WINDOWS[tf])
    start_ts = max(ready_points.values())  # common start after all warmups

    base_df = mtf_data[BASE_INTERVAL].sort_index()
    base_df = base_df.loc[base_df.index >= start_ts]
    if base_df.empty:
        raise ValueError("No overlapping time after cold-start cut")

    base_times = base_df.index
    aligned: Dict[str, pd.DataFrame] = {}
    for tf in TIMEFRAMES:
        df = mtf_data[tf].sort_index()
        df = df.loc[df.index >= start_ts]
        out = df.reindex(base_times).ffill()         # align to 5m timeline, ffill only
        out = out.replace([np.inf, -np.inf], np.nan)
        aligned[tf] = out

    # strict common index
    common_index = aligned[BASE_INTERVAL].index
    for tf in TIMEFRAMES:
        common_index = common_index.intersection(aligned[tf].dropna(how="any").index)
    if len(common_index) == 0:
        raise ValueError("After alignment/dropna, no common rows remain")
    for tf in TIMEFRAMES:
        aligned[tf] = aligned[tf].reindex(common_index)
    return aligned

def _to_numpy_windows_mtf(mtf_data: Dict[str, pd.DataFrame], feat_cols: List[str]) -> Dict[str, np.ndarray]:
    aligned = _align_timeframes(mtf_data, feat_cols)
    base_df = aligned[BASE_INTERVAL]
    max_window = max(WINDOWS.values())
    N = len(base_df)
    if N <= max_window + 1:
        raise ValueError(f"Not enough rows after alignment: need >{max_window+1}, got {N}")

    # Per-TF windowed obs
    mtf_obs: Dict[str, np.ndarray] = {}
    total_dim = 0
    for tf in TIMEFRAMES:
        df = aligned[tf]
        w = WINDOWS[tf]
        tf_cols = [c for c in feat_cols if c in df.columns]
        if not tf_cols:
            print(f"[MTF][warn] {tf}: no matching features; skipped")
            continue
        X = df[tf_cols].astype("float32").to_numpy(copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        T = N - max_window
        obs = np.empty((T, w * len(tf_cols)), dtype=np.float32)
        for t in range(T):
            s = max_window - w + t
            e = max_window + t
            obs[t] = X[s:e].reshape(-1)

        mtf_obs[tf] = obs
        total_dim += obs.shape[1]
        print(f"[MTF] {tf}: {len(tf_cols)} features × {w} window = {obs.shape[1]} dims")

    if not mtf_obs:
        raise ValueError("No timeframe produced observations")
    T = min(arr.shape[0] for arr in mtf_obs.values())
    combined = np.empty((T, total_dim), dtype=np.float32)
    i = 0
    for tf in TIMEFRAMES:
        if tf in mtf_obs:
            d = mtf_obs[tf][:T]
            combined[:, i:i+d.shape[1]] = d
            i += d.shape[1]

    # Returns & funding from base interval
    base_close = base_df["Close"].astype("float64").to_numpy(copy=False)
    base_fund8h = base_df.get("Funding8h", base_df["FundingRate"]).astype("float64").to_numpy(copy=False)

    ret = (base_close[max_window:max_window+T] - base_close[max_window-1:max_window+T-1]) / \
          np.maximum(base_close[max_window-1:max_window+T-1], 1e-12)
    fund_step = base_fund8h[max_window:max_window+T] / 96.0
    ts = pd.to_datetime(base_df.index[max_window:max_window+T], utc=True)

    combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    assert np.isfinite(combined).all(), "Non-finite in combined obs"

    print(f"[MTF] Combined obs shape: {combined.shape}")
    return dict(obs=combined, ret=ret.astype(np.float64), fund=fund_step.astype(np.float64), ts=ts)

# ===== Env =====
@dataclass
class CostConfig:
    fee_bps: float = FEE_BPS
    slip_bps: float = SLIP_BPS
    min_dpos: float = MIN_DPOS
    cooldown: int = COOLDOWN
    budget_daily: float = TURN_BUDGET_DAILY
    leverage: float = LEVERAGE
    smooth_alpha: float = SMOOTH_ALPHA

class CryptoFuturesMTFEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, data: Dict[str, np.ndarray], cost: CostConfig,
                 lambda_init=LAMBDA_INIT, lambda_step=LAMBDA_STEP, lambda_max=LAMBDA_MAX):
        super().__init__()
        self.obs_mat = data["obs"]
        self.rets    = data["ret"]
        self.fund    = data["fund"]
        self.ts      = data["ts"]
        self.T = len(self.rets)

        self.cost = cost
        self.lambda_ = lambda_init
        self.lambda_step = lambda_step
        self.lambda_max  = lambda_max

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_mat.shape[1],), dtype=np.float32)

        # state
        self.t = 0
        self.pos = 0.0
        self.pos_target = 0.0
        self.pos_last_change = -10**9
        self.turn_hist: List[float] = []
        self.turnover_roll = 0.0

        # 누적 보상 관리
        self.cum_pnl = 0.0
        self.cum_cost = 0.0
        self.prev_pos = 0.0

        # safety
        first = self.obs_mat[0]
        if not np.all(np.isfinite(first)):
            raise ValueError("Non-finite values in first observation")

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = 0.0
        self.pos_target = 0.0
        self.pos_last_change = -10**9
        self.turn_hist.clear()
        self.turnover_roll = 0.0
        self.lambda_ = min(self.lambda_, self.lambda_max)

        # 누적 초기화
        self.cum_pnl = 0.0
        self.cum_cost = 0.0
        self.prev_pos = 0.0

        return self.obs_mat[self.t], {}

    def _apply_action(self, a_raw: float) -> Tuple[float, float]:
        if (self.t - self.pos_last_change) < self.cost.cooldown:
            a = self.pos_target
        else:
            a_raw = float(np.clip(a_raw * ACTION_GAIN, -1.0, 1.0))
            a = (1 - self.cost.smooth_alpha) * self.pos_target + self.cost.smooth_alpha * a_raw
            if abs(a - self.pos_target) > 1e-12:
                self.pos_last_change = self.t
        dpos = a - self.pos
        return a, dpos

    def step(self, action: np.ndarray):
        assert self.t < self.T, "Episode done"
        a_raw = float(action[0])
        self.prev_pos = self.pos
        a_target, dpos = self._apply_action(a_raw)

        # === 비용 계산 ===
        fee_cost  = self.cost.fee_bps  * abs(dpos) * self.cost.leverage
        slip_cost = self.cost.slip_bps * abs(dpos) * self.cost.leverage

        self.pos = a_target
        self.pos_target = a_target

        pnl_ret   = (self.pos * self.cost.leverage) * self.rets[self.t]
        fund_cost = self.pos * self.fund[self.t]

        # === 누적 관리 ===
        self.cum_pnl  += pnl_ret
        self.cum_cost += fee_cost + slip_cost + fund_cost

        # === 턴오버 제약 ===
        self.turn_hist.append(abs(dpos))
        if len(self.turn_hist) > 288:
            self.turn_hist.pop(0)
        self.turnover_roll = float(sum(self.turn_hist))
        excess = max(0.0, self.turnover_roll - self.cost.budget_daily)

        if excess > 0.0:
            # λ 상승 속도 완화 (기존보다 절반)
            self.lambda_ = min(self.lambda_max, self.lambda_ + self.lambda_step * 0.5)

        # === 보상 계산 ===
        reward = 0.0

        # (1) shaping reward: 방향성 힌트 → 강도 0.1배 (이전엔 0.01)
        if self.pos != 0.0:
            reward += 0.1 * pnl_ret

        # (2) 청산 시점 정산
        if self.pos == 0.0 and self.prev_pos != 0.0:
            net = self.cum_pnl - self.cum_cost - self.lambda_ * excess
            reward += net
            self.cum_pnl = 0.0
            self.cum_cost = 0.0

        # (3) 전체 보상 scale normalization (스케일 차이 줄이기)
        reward = reward * 100.0

        info = dict(
            t=int(self.t), ts=str(self.ts[self.t].to_pydatetime()),
            pos=float(self.pos), dpos=float(dpos), pnl_ret=float(pnl_ret),
            fee=float(fee_cost), slip=float(slip_cost), fund=float(fund_cost),
            lambda_=float(self.lambda_), excess_turn=float(excess),
            ret=float(self.rets[self.t]), reward=float(reward)
        )

        self.t += 1
        terminated = self.t >= self.T
        truncated = False
        obs = self.obs_mat[self.t-1] if not terminated else self.obs_mat[self.T-1]
        return obs, float(reward), terminated, truncated, info

# ===== Callbacks =====
class PrintDiagCallback(BaseCallback):
    """주기별(롤링) 통계만 출력. 에피소드 종료/주기 출력 뒤에는 누적값 리셋."""
    def __init__(self, freq=5000, verbose=0):
        super().__init__(verbose)
        self.freq = freq
        self._reset()

    def _reset(self):
        self._eq = 1.0
        self._pnl = 0.0
        self._costs = 0.0
        self._turn = 0.0
        self._n = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)

        for info in infos:
            if not info:
                continue
            self._eq   *= (1.0 + info.get("ret", 0.0) * info.get("pos", 0.0))
            self._pnl  += info.get("pnl_ret", 0.0)
            self._costs+= (info.get("fee", 0.0) + info.get("slip", 0.0) + info.get("fund", 0.0))
            self._turn  = info.get("excess_turn", self._turn)
            self._n    += 1

        # 주기 출력 후 리셋
        if self.num_timesteps % self.freq == 0 and self._n:
            print(f"[diag] steps={self.num_timesteps:,} eq={self._eq:.3f} pnl={self._pnl:.5f} "
                  f"costs={self._costs:.5f} excess={self._turn:.3f}")
            self._reset()

        # 에피소드 종료 시에도 리셋
        if dones is not None and any(dones):
            self._reset()

        return True

class TurnoverCurriculum(BaseCallback):
    """초반 신호 학습을 위해 회전 제약 완화 → 점진 현실화."""
    def __init__(self, warmup_steps=300_000, tighten_steps=300_000):
        super().__init__()
        self.warmup = warmup_steps
        self.tight  = tighten_steps
    def _on_step(self) -> bool:
        t = self.num_timesteps
        vec = self.model.get_env()
        if not hasattr(vec, "envs") or len(vec.envs) == 0:
            return True
        env: CryptoFuturesMTFEnv = vec.envs[0]  # DummyVecEnv
        # 0 ~ warmup: 완화
        if t <= self.warmup:
            env.cost.cooldown      = 0
            env.cost.min_dpos      = 0.00
            env.cost.smooth_alpha  = 0.25
            env.cost.budget_daily  = 10.0
            env.lambda_step        = 0.0
        # warmup ~ warmup+tighten: 선형 보간
        elif t <= self.warmup + self.tight:
            r = (t - self.warmup) / self.tight
            env.cost.budget_daily  = 10.0 - r * (10.0 - 3.0)    # 10 → 3
            env.lambda_step        = r * 1e-5                  # 0 → 1e-5
            env.cost.min_dpos      = 0.00 + r * (0.00 - 0.00)  # 항상 0
            env.cost.smooth_alpha  = 0.25 + r * (0.35 - 0.25)  # 0.25 → 0.35
            env.cost.cooldown      = 0
        # 이후: 타깃 고정
        else:
            env.cost.budget_daily  = 3.0
            env.lambda_step        = 1e-5
            env.cost.min_dpos      = 0.00     # 최종 0 유지
            env.cost.smooth_alpha  = 0.35
            env.cost.cooldown      = 0
        return True

# ===== Train / Eval =====
def make_env_mtf(
    split="train",
    *,
    min_dpos: float = MIN_DPOS,
    cooldown: int = COOLDOWN,
    smooth_alpha: float = SMOOTH_ALPHA,
) -> gym.Env:
    mtf_data, feat_cols = _load_fe_mtf(split)
    data = _to_numpy_windows_mtf(mtf_data, feat_cols)
    env = CryptoFuturesMTFEnv(data, CostConfig(
        fee_bps=FEE_BPS, slip_bps=SLIP_BPS, min_dpos=min_dpos, cooldown=cooldown,
        budget_daily=TURN_BUDGET_DAILY, leverage=LEVERAGE, smooth_alpha=smooth_alpha
    ))
    return env

def evaluate_model_mtf(model: PPO, split="val") -> Dict[str, float]:
    """검증: deadzone=0로 평균정책(det) 체결 확인 + 활동성 지표."""
    env = make_env_mtf(split, min_dpos=0.0, cooldown=COOLDOWN, smooth_alpha=SMOOTH_ALPHA)
    obs, _ = env.reset()
    eq = 1.0; fees = slips = funds = 0.0; pnl = 0.0
    trades = 0; turn = 0.0
    pos_abs: List[float] = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        eq   *= (1.0 + info["pos"] * info["ret"])
        fees += info["fee"]; slips += info["slip"]; funds += info["fund"]; pnl += info["pnl_ret"]
        if abs(info["dpos"]) > 1e-9:
            trades += 1
            turn += abs(info["dpos"])
        pos_abs.append(abs(info["pos"]))
        if terminated or truncated:
            break

    p_mean = float(np.mean(pos_abs)) if pos_abs else 0.0
    p_p95  = float(np.percentile(pos_abs, 95)) if pos_abs else 0.0

    if trades == 0:
        print("[warn] EVAL made zero trades; mean action likely within min_dpos.")

    return dict(
        final_eq=eq, pnl=pnl, fees=fees, slips=slips, funds=funds, lambda_=info["lambda_"],
        trades=trades, turnover=turn, p_mean=p_mean, p_p95=p_p95
    )

def train_mtf():
    set_random_seed(SEED)
    print("[MTF] Creating training and validation environments...")
    # 초기엔 다소 부드럽게 시작(커리큘럼이 최종값으로 끌고감)
    train_env = DummyVecEnv([lambda: make_env_mtf("train", min_dpos=MIN_DPOS, cooldown=COOLDOWN, smooth_alpha=0.25)])
    val_env   = DummyVecEnv([lambda: make_env_mtf("val",   min_dpos=MIN_DPOS, cooldown=COOLDOWN, smooth_alpha=0.25)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=SEED,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,                 # 탐험 유지
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,                  # SDE 탐험
        sde_sample_freq=4,
        target_kl=0.05,                # 과도한 정책 수축 방지
        policy_kwargs=dict(
            net_arch=[512, 512, 256],
            activation_fn=torch.nn.ReLU,
            log_std_init=-0.5,         # 초기 분포폭
            ortho_init=False
        ),
        tensorboard_log=os.path.join(MODEL_DIR, "tb_mtf"),
        device="auto",
    )

    diag = PrintDiagCallback(freq=5000)

    class _EvalAndSaveMTF(BaseCallback):
        def __init__(self, eval_every=EVAL_EVERY):
            super().__init__()
            self.eval_every = eval_every
            self.best = -1e9
        def _on_step(self) -> bool:
            if self.num_timesteps % self.eval_every == 0:
                m = self.model
                metrics = evaluate_model_mtf(m, "val")
                score = metrics["final_eq"] - (metrics["fees"] + metrics["slips"] + abs(metrics["funds"]))
                active = (metrics["trades"] >= 200) or (metrics["turnover"] >= 0.02) or (metrics["p_mean"] >= 0.02)
                print(f"[eval] steps={self.num_timesteps:,} "
                      f"eq={metrics['final_eq']:.3f} pnl={metrics['pnl']:.5f} "
                      f"fees={metrics['fees']:.5f} slip={metrics['slips']:.5f} fund={metrics['funds']:.5f} "
                      f"| trades={metrics['trades']} turn={metrics['turnover']:.4f} p_mean={metrics['p_mean']:.4f}")
                if active and score > self.best:
                    self.best = score
                    path = os.path.join(MODEL_DIR, "best_model.zip")
                    m.save(path)
                    print(f"[save] {path}")
                else:
                    if not active:
                        print(f"[skip] inactive model (trades={metrics['trades']}, turn={metrics['turnover']:.4f}, p_mean={metrics['p_mean']:.4f})")
            return True

    curr = TurnoverCurriculum(warmup_steps=300_000, tighten_steps=300_000)

    print(f"[MTF] Starting training for {TOTAL_STEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_STEPS, callback=[diag, _EvalAndSaveMTF(), curr])

    final_path = os.path.join(MODEL_DIR, "ppo_mtf_practitioner_final.zip")
    model.save(final_path)
    print(f"[save] {final_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_mtf()
