# ai_binance/train/backtest.py
from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# ---- Force CPU ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

from ai_binance.train.reinforce.worker import TradeEnv, _ModelGB, _mask_fn
from ai_binance.train.reinforce.manager import ManagerV2Env

# ===== 사용자 경로 설정 =====
MODEL_ROOT = os.path.join("ai_binance", "data", "model")
MANAGER_MODEL_PATH   = os.path.join(MODEL_ROOT, "manager_v2.zip")
MANAGER_VECNORM_PATH = os.path.join(MODEL_ROOT, "manager_v2_vecnorm.pkl")

# 반드시 TradeEnv로 학습된 unified 워커 모델/벡노멀 pkl 사용!
WORKER_MODEL_PATH    = os.path.join(MODEL_ROOT, "worker_unified_final.zip")
WORKER_VECNORM_PATH  = os.path.join(MODEL_ROOT, "worker_unified_vecnorm.pkl")

SPLIT = "test"


def _load_worker_model(path: str):
    try:
        m = MaskablePPO.load(path, device="cpu")
        return m, "maskable"
    except Exception:
        m = PPO.load(path, device="cpu")
        return m, "ppo"


def _load_manager_bundle():
    # Manager VecNormalize는 ManagerV2Env(obs_dim = feat_dim * SEQ_WINDOW)에 묶여야 함
    dummy = DummyVecEnv([lambda: ManagerV2Env(split=SPLIT)])
    vec = VecNormalize.load(MANAGER_VECNORM_PATH, dummy)
    vec.training = False
    vec.norm_reward = False
    model = PPO.load(MANAGER_MODEL_PATH, device="cpu")
    return model, vec


def _build_goal_bridge():
    m, v = _load_manager_bundle()
    return _ModelGB(SPLIT, m, v)


def _load_worker_vecnorm(gb):
    dummy_worker_env = DummyVecEnv([lambda: TradeEnv(split=SPLIT, gb=gb, randomize_start=False)])
    vn = VecNormalize.load(WORKER_VECNORM_PATH, dummy_worker_env)
    vn.training = False
    vn.norm_reward = False
    # DummyVecEnv 자체가 VecEnv이므로 여기서 바로 shape 조회
    return vn, dummy_worker_env.observation_space.shape


def _check_shapes(model_obs_shape, env_obs_shape):
    mo = int(model_obs_shape[0]) if isinstance(model_obs_shape, tuple) else int(model_obs_shape)
    eo = int(env_obs_shape[0]) if isinstance(env_obs_shape, tuple) else int(env_obs_shape)
    if mo != eo:
        raise ValueError(f"[ShapeMismatch] worker_model_obs={mo} vs env_obs={eo} "
                         f"→ 워커 모델과 TradeEnv/VecNormalize의 관측차원이 다릅니다. "
                         f"TradeEnv로 학습된 unified 워커 모델(.zip/.pkl)을 지정하세요.")


def run_backtest():
    print("Initializing backtester...")

    # === Manager GoalBridge ===
    gb = _build_goal_bridge()

    # === Worker model ===
    worker_model, worker_kind = _load_worker_model(WORKER_MODEL_PATH)
    print(f"[WorkerModel] Loaded as: {worker_kind.upper()} | obs_shape={worker_model.observation_space.shape}")

    # === Worker VecNormalize ===
    worker_vecnorm, env_obs_shape = _load_worker_vecnorm(gb)
    print(f"[VecNorm] Worker VecNormalize loaded | env_obs_shape={env_obs_shape}")

    # === Sanity: shape match ===
    _check_shapes(worker_model.observation_space.shape, env_obs_shape)

    # === 데이터 길이 파악( deterministically ) ===
    # TradeEnv 내부 데이터 인덱스를 쓰기 위해 임시 env 생성
    probe_env = TradeEnv(split=SPLIT, gb=gb, randomize_start=False)
    idx = probe_env.idx
    N = len(idx)

    # === 백테스트 메인 루프: 전체 구간을 순차 스캔 (1 trade = 1 episode) ===
    trades = []
    equity = 1.0  # 초기자본 1.0 기준
    i = 0
    print("Starting backtest simulation...")

    while i < N - 2:
        # Episode 시작 인덱스 고정
        env = TradeEnv(split=SPLIT, gb=gb, randomize_start=False)
        # reset()으로 내부 상태 초기화 후, 시작 커서를 강제로 지정
        _obs0, _ = env.reset()
        env._cursor = i
        env.in_position = False
        env.entry_time = 0
        env.entry_price = 0.0
        env.holding_steps = 0
        env.current_dir = 0

        # 시작 obs 재계산
        obs = env._obs()

        while True:
            norm_obs = worker_vecnorm.normalize_obs(obs)
            if worker_kind == "maskable":
                try:
                    masks = _mask_fn(env)
                except Exception:
                    masks = None
                action, _ = worker_model.predict(norm_obs, deterministic=True, action_masks=masks)
            else:
                action, _ = worker_model.predict(norm_obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(int(action))
            # TradeEnv는 3항 done(terminated)만 의미 있음
            if done:
                # 트레이드가 있었으면 기록
                if "pnl" in info:
                    pnl = float(info["pnl"])
                    holding = int(info.get("holding_steps", 0))
                    forced = bool(info.get("forced_exit", False))
                    equity *= (1.0 + pnl)
                    trades.append({
                        "entry_ts": info.get("entry_ts"),
                        "exit_ts": info.get("exit_ts"),
                        "dir": int(info.get("dir", 0)),
                        "pnl": pnl,
                        "gross": float(info.get("gross", np.nan)),
                        "fee": float(info.get("fee", np.nan)),
                        "fund": float(info.get("fund", np.nan)),
                        "slip": float(info.get("slip", np.nan)),
                        "holding_steps": holding,
                        "forced_exit": forced,
                        "equity": equity
                    })
                    print(f"Trade closed. PnL={pnl:+.6f} | gross={info.get('gross'):+.6f} "
                        f"| fee={info.get('fee'):+.6f} | fund={info.get('fund'):+.6f} "
                        f"| slip={info.get('slip'):+.6f} | hold={holding} | equity={equity:.6f}")

                # 다음 에피소드 시작 인덱스: 현 커서 다음
                i = env._cursor + 1
                break

    print("Backtest simulation finished.")
    return summarize_results(trades)


def summarize_results(trades: list[dict]):
    if not trades:
        print("No trades were made during the backtest.")
        return {}

    df = pd.DataFrame(trades)

    # --- numeric coercion (robust to missing columns) ---
    def _num(s, default=np.nan):
        return pd.to_numeric(df.get(s), errors="coerce").fillna(default)

    pnl    = _num("pnl", 0.0).astype(float)
    equity = _num("equity").astype(float)

    total_return = float((equity.iloc[-1] - 1.0))
    sharpe = float((pnl.mean() / (pnl.std() + 1e-12)) * np.sqrt(252.0))
    mdd = float((equity.cummax() - equity).max())
    win_rate = float((pnl > 0).mean())
    avg_pnl = float(pnl.mean())

    # optional components (will be NaN if not provided)
    fee  = _num("fee")
    fund = _num("fund")
    slip = _num("slip")
    hold = _num("holding_steps")

    avg_fee  = float(fee.mean())
    avg_fund = float(fund.mean())
    avg_slip = float(slip.mean())
    avg_hold = float(hold.mean())
    p95_hold = float(hold.quantile(0.95))

    metrics = {
        "Total Return": f"{total_return:+.6f}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{mdd:.6f}",
        "Win Rate": f"{win_rate:.2%}",
        "Total Trades": int(len(df)),
        "Avg PnL/Trade": f"{avg_pnl:+.6f}",
        "Final Equity": f"{equity.iloc[-1]:.6f}",
        "Avg Fee/Trade": f"{avg_fee:+.6f}",
        "Avg Funding/Trade": f"{avg_fund:+.6f}",
        "Avg Slippage/Trade": f"{avg_slip:+.6f}",
        "Avg Holding Steps": f"{avg_hold:.2f}",
        "P95 Holding Steps": f"{p95_hold:.0f}",
    }

    print("\n--- Backtest Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # Save trades with all columns for deeper analysis
    out_dir = os.path.join("ai_binance", "data", "backtest")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "trades_unified_worker.csv"), index=False)

    return metrics


if __name__ == "__main__":
    # 파일 존재 체크(명시적)
    for p in [MANAGER_MODEL_PATH, MANAGER_VECNORM_PATH, WORKER_MODEL_PATH, WORKER_VECNORM_PATH]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
    results = run_backtest()
