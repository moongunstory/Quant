# -*- coding: utf-8 -*-
"""
HPO 탐색 오케스트레이터 (유전 알고리즘)
- 개체: 피처 조합 (크기 제한: min_k ~ max_k)
- 적합도: Sharpe/IR 가중합 - MDD/제약 벌점
- 연산: 엘리트 보존, 토너먼트 선택, 교차, 돌연변이
- 병렬 평가, 조기 종료, 결과 저장(best_features.json, hpo_history.json)

실행:
    python -m train.reinforce.hpo.opt
또는
    python train/reinforce/hpo/opt.py
"""
from __future__ import annotations
import os, sys, json, random, multiprocessing as mp
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# ----- 안전 임포트 -----
HERE = os.path.dirname(__file__)
TRAIN_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

try:
    from fe import load_processed, build_universe_from_processed
    from reinforce.hpo.eval import evaluate_feature_set
except Exception:
    BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
    if BASE not in sys.path:
        sys.path.append(BASE)
    from fe import load_processed, build_universe_from_processed
    # eval 모듈 경로 보정
    HPO_DIR = os.path.abspath(os.path.join(HERE))
    if HPO_DIR not in sys.path:
        sys.path.append(HPO_DIR)
    from eval import evaluate_feature_set

# ----- 제약/패널티 -----
def _correlation_penalty(df: pd.DataFrame, cols: List[str], thr: float = 0.9) -> float:
    if len(cols) < 2:
        return 0.0
    # 존재하는 컬럼만 사용
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return 0.0
    C = df[cols].astype(float).corr().abs()
    np.fill_diagonal(C.values, 0.0)
    return float((C.values > thr).sum() / 2)

def _diversity_penalty(cols: List[str]) -> float:
    """간단 다양성 패널티: 동일 접미(TF) 또는 동일 prefix 그룹 쏠림 방지."""
    if not cols:
        return 0.0
    from collections import Counter
    tfs = [c.split("_")[-1] for c in cols]
    tf_max = max(Counter(tfs).values()) / len(cols)
    # 대략적 그룹화: 'bin_', 'ratio_', 'ret_', 'ema', 'rsi' 등 접두 기준
    def _grp(c: str) -> str:
        if c.startswith("f_bin_"): return "bin"
        if "_over_atr_" in c or c.startswith("f_ret_"): return "ret"
        if "rsi" in c: return "rsi"
        if "ema" in c: return "ema"
        if "macd" in c: return "macd"
        return "etc"
    gmax = max(Counter([_grp(c) for c in cols]).values()) / len(cols)
    return float(tf_max + gmax)

# ----- GA 연산 -----
def _init_population(universe: List[str], pop_size: int, min_k: int, max_k: int) -> List[List[str]]:
    pop = []
    for _ in range(pop_size):
        k = random.randint(min_k, max_k)
        pop.append(random.sample(universe, k))
    return pop

def _tournament_select(pop: List[List[str]], scores: List[float], k: int = 3) -> List[str]:
    cand = random.sample(list(range(len(pop))), min(k, len(pop)))
    cand.sort(key=lambda i: scores[i], reverse=True)
    return list(pop[cand[0]])

def _crossover(A: List[str], B: List[str], max_k: int) -> List[str]:
    inter = list(set(A) & set(B))
    only  = list(set(A) ^ set(B))
    need  = max(0, max_k - len(inter))
    add   = random.sample(only, min(need, len(only)))
    return inter + add

def _mutate(S: List[str], universe: List[str],
            p_add=0.3, p_del=0.3, p_swap=0.1,
            min_k: int = 50, max_k: int = 100) -> List[str]:
    S = set(S)
    if random.random() < p_add and len(S) < max_k:
        S.add(random.choice(universe))
    if random.random() < p_del and len(S) > min_k:
        S.remove(random.choice(list(S)))
    if random.random() < p_swap and len(S) >= 1 and len(universe) > len(S):
        out = list(S)
        out[random.randrange(len(out))] = random.choice(list(set(universe) - S))
        S = set(out)
    return list(S)

# ----- 후보 평가 -----
def _eval_candidate(args) -> Dict:
    (df_tr, df_va, selected, env_kwargs, seeds, train_steps, val_steps,
     weights, limits, corr_thr, cache_dir) = args

    m = evaluate_feature_set(
        df_train_5m=df_tr, df_val_5m=df_va, selected_feats=selected,
        env_kwargs=env_kwargs, seeds=seeds,
        train_steps=train_steps, val_steps=val_steps,
        cache_dir=cache_dir
    )

    # 기본 점수(가중합). mdd는 낮을수록 좋아서 음수 가중.
    score = (m["sharpe"] * weights.get("sharpe", 1.0)
             + m["ir"]   * weights.get("ir", 0.0)
             - m["mdd"]  * abs(weights.get("mdd", 1.0)))

    # 제약/패널티
    corr_p = _correlation_penalty(df_va, selected, thr=corr_thr)
    div_p  = _diversity_penalty(selected)
    score += -0.2 * corr_p - 0.2 * div_p

    if m["mdd"] > limits.get("max_mdd", 0.3):
        score -= (m["mdd"] - limits["max_mdd"]) * 10.0
    if m["trades_per_day"] > limits.get("max_trades_per_day", 60):
        score -= (m["trades_per_day"] - limits["max_trades_per_day"]) * 0.5

    return {"metrics": m, "score": float(score)}

# ----- 메인 러너 -----
def run(
    pop_size: int = 24,
    generations: int = 20,
    min_k: int = 50, max_k: int = 100,
    seeds: List[int] = [0, 1],
    train_steps: int = 30_000,
    val_steps: int = 20_000,
    env_kwargs: Dict = {"fee_rate": 0.0004, "slip_bp": 2.0, "random_start": False},
    weights: Dict = {"sharpe": 1.0, "ir": 0.3, "mdd": 1.0},
    limits: Dict = {"max_mdd": 0.25, "max_trades_per_day": 40},
    corr_thr: float = 0.9,
    n_jobs: int = 4,
    cache_dir: str = "./.hpo_cache",
    out_dir: str = os.path.join("train", "reinforce", "hpo"),
    save_best_path: str = "best_features.json",
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # 데이터/유니버스 로드 (HPO 확장 프레임이 있으면 fe.load_processed(..., mode="auto") 가 자동 사용)
    df_tr = load_processed("train", "5m", mode="auto")
    df_va = load_processed("val",   "5m", mode="auto")
    universe = build_universe_from_processed("train", "5m", mode="auto")
    if len(universe) < max_k:
        print(f"[warn] universe size ({len(universe)}) < max_k ({max_k}). max_k를 줄이거나 FE 확장을 확인하세요.")
        max_k = max(10, len(universe) // 2)

    print(f"[init] universe={len(universe)} features, pop={pop_size}, gen={generations}, "
          f"k∈[{min_k},{max_k}], seeds={seeds}, steps={train_steps}/{val_steps}")

    # 초기 개체군
    pop = _init_population(universe, pop_size, min_k, max_k)
    best, best_score, stall = None, -1e9, 0
    history = []

    for gen in range(generations):
        args = [
            (df_tr, df_va, ind, env_kwargs, seeds, train_steps, val_steps,
             weights, limits, corr_thr, cache_dir)
            for ind in pop
        ]

        if n_jobs > 1:
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.map(_eval_candidate, args)
        else:
            results = list(map(_eval_candidate, args))

        scores = [r["score"] for r in results]
        g_best_idx = int(np.argmax(scores))
        g_best = pop[g_best_idx]
        g_score = float(scores[g_best_idx])
        g_mean = float(np.mean(scores))

        history.append({"gen": gen, "best_score": g_score, "mean_score": g_mean, "best_size": len(g_best)})
        print(f"[gen {gen:02d}] best={g_score:.3f} mean={g_mean:.3f} | size={len(g_best)}")

        # 글로벌 베스트 갱신 + 저장
        if g_score > best_score:
            best, best_score, stall = g_best, g_score, 0
            with open(os.path.join(out_dir, save_best_path), "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)
        else:
            stall += 1

        # 조기 종료
        if stall >= 5:
            print("[early-stop] score stalled.")
            break

        # 다음 세대 (엘리트 보존 + 토너먼트 + 교차/돌연변이)
        next_pop = [g_best]
        while len(next_pop) < len(pop):
            A = _tournament_select(pop, scores, k=3)
            B = _tournament_select(pop, scores, k=3)
            C = _crossover(A, B, max_k)
            C = _mutate(C, universe, min_k=min_k, max_k=max_k)
            next_pop.append(C)
        pop = next_pop

    # 결과 저장
    out = {"best_feats": best, "best_score": float(best_score), "history": history}
    with open(os.path.join(out_dir, "hpo_history.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] best_score={best_score:.3f}, feats={len(best) if best else 0}")
    return out

if __name__ == "__main__":
    # 기본 설정으로 실행
    run()
