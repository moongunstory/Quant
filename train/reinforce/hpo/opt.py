# -*- coding: utf-8 -*-
"""
HPO 탐색 오케스트레이터 (유전 알고리즘)
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
    from prepare.engine import load_processed, build_universe_from_processed
    from reinforce.hpo.eval import evaluate_feature_set
except Exception:
    BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
    if BASE not in sys.path:
        sys.path.append(BASE)
    from prepare.engine import load_processed, build_universe_from_processed
    HPO_DIR = os.path.abspath(os.path.join(HERE))
    if HPO_DIR not in sys.path:
        sys.path.append(HPO_DIR)
    from eval import evaluate_feature_set

# ----- 제약/패널티 -----
def _correlation_penalty(df: pd.DataFrame, cols: List[str], thr: float = 0.9) -> float:
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns in df for correlation check: {missing}"
    if len(cols) < 2:
        return 0.0
    C = df[cols].astype(float).corr().abs()
    np.fill_diagonal(C.values, 0.0)
    return float((C.values > thr).sum() / 2)

def _diversity_penalty(cols: List[str]) -> float:
    if not cols:
        return 0.0
    from collections import Counter
    tfs = [c.split("_")[-1] for c in cols]
    tf_max = max(Counter(tfs).values()) / len(cols)
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
    return [random.sample(universe, random.randint(min_k, max_k)) for _ in range(pop_size)]

def _tournament_select(pop: List[List[str]], scores: List[float], k: int = 3) -> List[str]:
    cand = random.sample(range(len(pop)), min(k, len(pop)))
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
    try:
        (df_tr, df_va, selected, env_kwargs, seeds, train_steps, val_steps,
         weights, limits, corr_thr, cache_dir) = args

        m = evaluate_feature_set(
            df_train_5m=df_tr, df_val_5m=df_va, selected_feats=selected,
            env_kwargs=env_kwargs, seeds=seeds,
            train_steps=train_steps, val_steps=val_steps,
            cache_dir=cache_dir
        )

        score = (m["sharpe"] * weights.get("sharpe", 1.0)
                 + m["ir"]   * weights.get("ir", 0.0)
                 - m["mdd"]  * abs(weights.get("mdd", 1.0)))

        corr_p = _correlation_penalty(df_va, selected, thr=corr_thr)
        div_p  = _diversity_penalty(selected)
        score += -0.2 * corr_p - 0.2 * div_p

        if m["mdd"] > limits.get("max_mdd", 0.3):
            score -= (m["mdd"] - limits["max_mdd"]) * 10.0
        if m["trades_per_day"] > limits.get("max_trades_per_day", 60):
            score -= (m["trades_per_day"] - limits["max_trades_per_day"]) * 0.5

        return {"metrics": m, "score": float(score)}

    except Exception as e:
        print(f"[eval error] {e}")
        return {"metrics": None, "score": float("-inf")}

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
    out_dir: str = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data", "hpo")),
    save_best_path: str = "best_features.json",
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    df_tr = load_processed("train", "5m", mode="auto")
    df_va = load_processed("val",   "5m", mode="auto")
    universe = build_universe_from_processed("train", "5m", mode="auto")
    if len(universe) < max_k:
        print(f"[warn] universe size ({len(universe)}) < max_k ({max_k}).")
        max_k = max(10, len(universe) // 2)

    print(f"[init] universe={len(universe)}, pop={pop_size}, gen={generations}, "
          f"k∈[{min_k},{max_k}], seeds={seeds}, steps={train_steps}/{val_steps}")

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

        if g_score > best_score:
            best, best_score, stall = g_best, g_score, 0
            with open(os.path.join(out_dir, save_best_path), "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2,
                          default=lambda x: int(x) if isinstance(x, (np.integer, np.int_)) else x)
        else:
            stall += 1

        if stall >= 5:
            print("[early-stop] score stalled.")
            break

        next_pop = [g_best]
        while len(next_pop) < len(pop):
            A = _tournament_select(pop, scores, k=3)
            B = _tournament_select(pop, scores, k=3)
            C = _crossover(A, B, max_k)
            C = _mutate(C, universe, min_k=min_k, max_k=max_k)
            next_pop.append(C)
        pop = next_pop

    out = {"best_feats": best, "best_score": float(best_score), "history": history}
    with open(os.path.join(out_dir, "hpo_history.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2,
                  default=lambda x: int(x) if isinstance(x, (np.integer, np.int_)) else x)
    print(f"[done] best_score={best_score:.3f}, feats={len(best) if best else 0}")
    return out

if __name__ == "__main__":
    run()
