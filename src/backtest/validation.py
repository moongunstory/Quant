"""검증 — 진짜 엣지와 '운 좋아 보이는 노이즈'를 구분(과최적화 방어).

알파를 많이 시도하면 그중 몇 개는 순전히 우연으로 좋아 보인다. 이 도구들은
알파 하나의 순손익 시리즈를 스트레스 테스트해서 살아남는 것만 남긴다:

  1. oos_split      — 시간을 IS(과거)/OOS(미래)로 나눔. 진짜 엣지는 '본 적 없는'
                      데이터에서도 비슷한 샤프를 유지.
  2. walk_forward   — 여러 구간으로 나눠 반복. 진짜 엣지는 한 구간이 아니라 대부분에서 양수.
  3. block_permutation_test — 정직성 검사. 며칠씩 묶은 블록의 부호를 무작위로 수천 번
                      뒤집어 '엣지 없음' 분포를 만들고, 노이즈가 실제 샤프를 이기는 빈도(=p값)를 잰다.
  4. bonferroni     — 다중검정 보정. N개 알파를 시도했으면 유의 기준을 N배 엄격(alpha/N)하게.

입력은 모두 엔진의 일별 순손익 시리즈(이미 비용 차감됨). 시드로 재현 가능.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest import metrics
from src.config.backtest_settings import SETTINGS


def _clean(net_pnl):
    return pd.Series(net_pnl).dropna()


def oos_split(net_pnl, oos_fraction=0.3):
    """시간순 분할: 앞 (1-frac)=IS, 뒤 frac=OOS. 각 샤프 + 비율(OOS/IS). 1 근처면 유지된 것."""
    r = _clean(net_pnl)
    if len(r) < 4:
        raise ValueError("IS/OOS 분할에 최소 4개 관측 필요")
    cut = min(max(int(len(r) * (1.0 - oos_fraction)), 1), len(r) - 1)
    is_s = metrics.sharpe(r.iloc[:cut])
    oos_s = metrics.sharpe(r.iloc[cut:])
    ratio = (oos_s / is_s) if is_s != 0.0 else float("nan")
    return {"is_sharpe": is_s, "oos_sharpe": oos_s, "oos_is_ratio": ratio,
            "is_days": int(cut), "oos_days": int(len(r) - cut)}


def walk_forward(net_pnl, n_folds=4):
    """시계열을 n_folds 등분, 각 샤프. 견고한 알파는 대부분 구간에서 양수."""
    r = _clean(net_pnl)
    if len(r) < n_folds * 2:
        raise ValueError(f"{n_folds} 폴드에 최소 {n_folds*2} 관측 필요")
    folds = np.array_split(r.values, n_folds)
    sharpes = [metrics.sharpe(pd.Series(f)) for f in folds]
    pos = sum(1 for s in sharpes if s > 0)
    return {"fold_sharpes": [float(s) for s in sharpes], "n_folds": n_folds,
            "pct_positive": pos / n_folds, "min_sharpe": float(min(sharpes)),
            "mean_sharpe": float(np.mean(sharpes))}


def block_permutation_test(net_pnl, n_perm=1000, block=5, seed=None):
    """블록 부호뒤집기 순열검정. 귀무가설: 일별 수익 부호가 무작위(진짜 드리프트 없음).
    노이즈 샤프가 실제 샤프 이상인 빈도 = p값. 낮으면(<0.05) 운으로 보기 어려움."""
    r = _clean(net_pnl)
    if len(r) < block * 2:
        raise ValueError("블록 크기에 비해 시계열이 너무 짧음")
    rng = np.random.default_rng(seed if seed is not None else SETTINGS.random_seed)
    vals = r.values
    observed = metrics.sharpe(r)
    n_blocks = int(np.ceil(len(vals) / block))
    ge = 0
    for _ in range(n_perm):
        signs = rng.choice((-1.0, 1.0), size=n_blocks)
        signs_full = np.repeat(signs, block)[: len(vals)]
        if metrics.sharpe(pd.Series(vals * signs_full)) >= observed:
            ge += 1
    p_value = (ge + 1) / (n_perm + 1)  # +1 평활: p 가 정확히 0 이 되지 않게
    return {"observed_sharpe": float(observed), "p_value": float(p_value),
            "n_perm": n_perm, "block": block}


def bonferroni(p_values, alpha=0.05):
    """다중검정 보정. N개 알파의 p값 -> 유의 기준 alpha/N. 각 통과여부 반환."""
    p = list(p_values)
    n = len(p)
    if n == 0:
        return {"threshold": alpha, "n_tests": 0, "passed": []}
    threshold = alpha / n
    return {"threshold": threshold, "n_tests": n,
            "passed": [bool(pv <= threshold) for pv in p]}


def validate(net_pnl, n_perm=1000, block=5, n_folds=4, oos_fraction=0.3, seed=None):
    """단일 알파 검사를 한 번에. (bonferroni 는 여러 알파 p값에 나중에 적용.)"""
    return {
        "oos": oos_split(net_pnl, oos_fraction=oos_fraction),
        "walk_forward": walk_forward(net_pnl, n_folds=n_folds),
        "permutation": block_permutation_test(net_pnl, n_perm=n_perm, block=block, seed=seed),
    }
