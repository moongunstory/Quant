"""combine — 알파별 가중치 패널을 하나의 포트폴리오 가중치로 결합.

입력: {alpha_name -> date×coin 가중치 패널} (각 알파는 이미 일별 L1=1).
출력: 결합 date×coin 가중치 패널 (다시 일별 L1=1 로 재정규화).

가중방식(WEIGHTING_REGISTRY, config 진입점):
  equal        모든 알파 동일 비중.
  inverse_vol  변동성 낮은(=꾸준한) 알파에 더 큰 비중. 알파별 순손익 필요.

coin portfolio/combine.py 이식(Phase 1). hedge(시변 EG 가중)·상관기반 선택은 Phase 3.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.backtest import operators as ops
from src.backtest import metrics as M
from src.config.backtest_settings import SETTINGS

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 상관행렬 (coin alpha_db.correlation_matrix 이식 — DB 없이 net_pnls 로부터)
# --------------------------------------------------------------------------- #

def correlation_matrix(net_pnls):
    """알파별 net-PnL 시리즈 dict -> 쌍별 상관행렬 DataFrame.

    coin 은 SQLite alpha_db 에 PnL 을 저장하고 correlation_matrix(conn, names)
    로 계산하지만, Quant 파이프라인은 이미 net_pnls(알파별 순손익)을 메모리로
    들고 있으므로 DB 없이 곧바로 계산한다(PLAN 2026-07-11 이식 판단)."""
    frame = pd.DataFrame({n: pd.Series(p) for n, p in net_pnls.items()})
    return frame.corr()


# --------------------------------------------------------------------------- #
# selection (coin research/portfolio/combine.py 이식)
# --------------------------------------------------------------------------- #

def select_low_corr(ranked_names, corr, threshold=None, max_n=None,
                    fitness=None, min_fitness=None,
                    families=None, max_per_family=None,
                    metrics=None, min_recent_sharpe=None,
                    recent_key="sharpe_hl"):
    """fitness 순(랭킹 순)으로 훑으며, 이미 뽑은 알파와 |상관|이 threshold 미만일
    때만 채택하는 greedy dedup.

    corr          : 쌍별 net-PnL 상관 정사각 DataFrame(index/cols = 이름).
    fitness/min_fitness : 품질 게이트. 둘 다 주어지면 fitness<min_fitness 인 알파를
                    greedy 상관 패스 전에 제거. "상관 낮은 노이즈"가 다양성으로
                    둔갑해 뽑히는 것을 막는다. fitness 는 {name->float}, 여기 없는
                    이름도 제거(점수 없음 = 품질 증거 없음).
    metrics/min_recent_sharpe : 최근성(쇠퇴) 게이트. 둘 다 주어지면 최근 샤프
                    (metrics[name][recent_key], 기본 sharpe_hl=반감기가중 샤프)가
                    min_recent_sharpe 미만인 알파를 제거. 전체기간 fitness 는 좋아도
                    최근에 죽은 알파(예: 2020~21 에만 수익)를 편입하는 것을 막는다.
    families/max_per_family : 패밀리 캡. 상관이 threshold 미만이어도 같은 매크로
                    팩터(예: funding 계열)를 여러 번 쌓는 것을 막는다. {name->family}
                    와 상한이 주어지면 패밀리당 최대 그만큼만, fitness 좋은 순으로.
    반환: 선택된 이름들(선택 순서)."""
    threshold = SETTINGS.combine_corr_threshold if threshold is None else threshold
    all_ranked = list(ranked_names)
    pool = list(ranked_names)
    if fitness is not None and min_fitness is not None:
        pool = [n for n in pool
                if fitness.get(n) is not None and fitness[n] >= min_fitness]
    dropped_gate = [n for n in all_ranked if n not in pool]
    if metrics is not None and min_recent_sharpe is not None:
        def _recent(n):
            m = metrics.get(n) or {}
            v = m.get(recent_key)
            return v if v is not None else float("-inf")
        before = pool
        pool = [n for n in before if _recent(n) >= min_recent_sharpe]
        dropped_gate += [n for n in before if n not in pool]
    fam_count = {}
    selected = []
    dropped_corr = {}    # name -> (가장가까운선택알파, 상관)
    dropped_family = []  # 패밀리 캡으로 탈락
    for name in pool:
        if families is not None and max_per_family is not None:
            f = families.get(name, "other")
            if fam_count.get(f, 0) >= int(max_per_family):
                dropped_family.append(name)
                continue
        ok = True
        worst = (None, 0.0)
        for chosen in selected:
            try:
                c = abs(corr.loc[name, chosen])
            except KeyError:
                c = 0.0
            if pd.notna(c) and c > worst[1]:
                worst = (chosen, float(c))
            if pd.notna(c) and c >= threshold:
                ok = False
        if ok:
            selected.append(name)
            if families is not None and max_per_family is not None:
                f = families.get(name, "other")
                fam_count[f] = fam_count.get(f, 0) + 1
        else:
            dropped_corr[name] = worst
        if max_n and len(selected) >= max_n:
            break

    # 상세 선택 로그는 debug(=기본 미출력). walkforward 는 시점마다 이 함수를 수십 번
    # 부르므로 info 로 두면 콘솔이 스팸으로 뒤덮인다. 시점별 선택 내역이 필요하면
    # 로깅 레벨을 DEBUG 로 올리거나 walkforward 리포트(선택빈도·회전로그)를 보면 된다.
    log.debug("selection: 후보 %d → 게이트통과 %d → 선택 %d %s",
              len(all_ranked), len(pool), len(selected), selected)
    if dropped_gate:
        log.debug("selection: 게이트 탈락(fitness/recency) %d: %s",
                  len(dropped_gate), dropped_gate)
    if dropped_corr:
        detail = ", ".join(f"{n}(vs {c[0]} r={c[1]:.2f})"
                           for n, c in dropped_corr.items())
        log.debug("selection: 상관 탈락(>=%.2f) %d: %s",
                  threshold, len(dropped_corr), detail)
    if dropped_family:
        log.debug("selection: 패밀리캡 탈락 %d: %s",
                  len(dropped_family), dropped_family)
    return selected


def _sel_low_correlation(ranked_names, corr, max_corr_threshold=None, top_n=None,
                         min_fitness=0.0, fitness=None,
                         max_per_family=None, families=None,
                         min_recent_sharpe=None, recent_key="sharpe_hl",
                         metrics=None, **kwargs):
    """Registry 어댑터: config 파라미터명 -> select_low_corr 인자.

    min_fitness 기본 0.0: 기본적으로 fitness 음수 알파는 절대 편입 불가.
    config 에서 "min_fitness": null 로 게이트 해제 가능. 게이트는 호출자가
    fitness dict 를 줄 때만 작동(파이프라인은 준다).
    min_recent_sharpe(선택): 주면 최근성 게이트 활성 — 최근 샤프가 낮은(쇠퇴한)
    알파를 배제(metrics dict 필요, 파이프라인이 항상 넘김). **kwargs 는 공용 진입점
    select_alphas 가 모든 method 에 균일하게 넘기는 여분 인자를 흡수."""
    return select_low_corr(ranked_names, corr,
                           threshold=max_corr_threshold, max_n=top_n,
                           fitness=fitness, min_fitness=min_fitness,
                           families=families, max_per_family=max_per_family,
                           metrics=metrics, min_recent_sharpe=min_recent_sharpe,
                           recent_key=recent_key)


def _sel_manual(ranked_names, corr, names=None, **kwargs):
    """수동 지정: 주어진 리스트를 그 순서 그대로 반환(이름이 없으면 fail loud).
    상관/패밀리 필터 없음 — 의도적으로 손으로 고른 구성을 그대로 쓰기 위한 것."""
    if not names:
        raise ValueError("selection method 'manual' 은 params.names(알파 이름 리스트)가 필요")
    available = set(ranked_names)
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"selection method 'manual': 이번 run 에서 못 찾은 알파: {missing}")
    return list(names)


SELECTION_REGISTRY = {
    "low_correlation": _sel_low_correlation,
    "manual": _sel_manual,
    # Phase 3: "low_correlation_recency", "min_fitness_gate"
}


def select_alphas(ranked_names, corr, method="low_correlation", params=None,
                  fitness=None, families=None, metrics=None, pnl_by_alpha=None):
    """config 진입점: 등록된 selection method 실행.

    모든 registry 함수는 fitness/metrics/pnl_by_alpha 를 받아야(그리고 **kwargs 로
    무시해도) 한다 — 공용 진입점이 항상 셋 다 넘긴다.
    contract: f(ranked_names, corr, fitness=None, families=None, metrics=None,
    pnl_by_alpha=None, **params)."""
    if method not in SELECTION_REGISTRY:
        raise ValueError(f"unknown selection method {method!r} "
                         f"(available: {sorted(SELECTION_REGISTRY)})")
    return SELECTION_REGISTRY[method](ranked_names, corr, fitness=fitness,
                                      families=families, metrics=metrics,
                                      pnl_by_alpha=pnl_by_alpha,
                                      **(params or {}))


def _alpha_vol(net_pnl):
    r = pd.Series(net_pnl).dropna()
    sd = r.std(ddof=1)
    return float(sd) if sd and not np.isnan(sd) else np.nan


def _cap_weights(w, max_weight):
    """어떤 알파도 max_weight 를 못 넘게 하고, 초과분은 여유 있는 알파에 비례 재분배.
    합=1 유지. inverse_vol 이 저변동 알파 하나에 북을 몰빵하는 것을 막는다
    (저변동≠저위험 — 느리게 큰 드로다운을 내는 알파가 최대 비중을 받는 문제)."""
    w = dict(w)
    n = len(w)
    if n == 0:
        return w
    if max_weight * n <= 1.0 + 1e-9:
        # 캡이 너무 낮아 합=1 이 불가능 → 균등가중으로 폴백.
        return {k: 1.0 / n for k in w}
    for _ in range(1000):
        over = [k for k, v in w.items() if v > max_weight + 1e-12]
        if not over:
            break
        excess = sum(w[k] - max_weight for k in over)
        for k in over:
            w[k] = max_weight
        under = [k for k in w if w[k] < max_weight - 1e-12]
        pool = sum(w[k] for k in under) or 1.0
        for k in under:
            w[k] += excess * (w[k] / pool)
    return w


def _wgt_equal(names, pnl_by_alpha=None):
    return {n: 1.0 for n in names}


def _wgt_inverse_vol(names, pnl_by_alpha=None, max_weight=None):
    """PnL 변동성 역수 가중. max_weight 를 주면 어떤 알파도 그 비중을 못 넘게 캡
    (초과분은 나머지에 비례 재분배). 몰빵 방지용."""
    if pnl_by_alpha is None:
        raise ValueError("inverse_vol 은 pnl_by_alpha 가 필요")
    inv = {n: 1.0 / _alpha_vol(pnl_by_alpha[n]) for n in names}
    inv = {n: (0.0 if np.isnan(v) or np.isinf(v) else v) for n, v in inv.items()}
    total = sum(inv.values()) or 1.0
    w = {n: inv[n] / total for n in names}
    if max_weight is not None:
        w = _cap_weights(w, float(max_weight))
    return w


def _wgt_skill(names, pnl_by_alpha=None, half_life=90, floor=0.0,
              power=1.0, max_weight=None):
    """최근 실력(반감기가중 샤프)에 '비례'해 비중을 준다. 뜨거운 알파는 더 받고,
    식은 알파는 비중이 부드럽게 0 으로 수렴한다 — 이진(넣거나 빼거나) 게이트와
    달리 뚝 끊기지 않고 서서히 벤치/복귀한다.

    half_life : 반감기(일, 기본 90). 작을수록 최근 성적에 민감(더 빨리 갈아탐).
    floor     : 이 (반감기가중)샤프 미만이면 기여 0 — 죽은 알파를 부드럽게 뺀다.
                기본 0.0 = 최근 손실 알파는 비중 0.
    power     : 실력 격차 강조. 1=선형, >1 이면 잘하는 알파에 더 몰아줌.
    max_weight: 한 알파 비중 상한(초과분 나머지에 재분배). 몰빵 방지.
    causal: pnl_by_alpha 가 asof 까지만 담기면 점수도 asof 까지 계산 → 미래참조 0
    (walkforward 가 causal_pnls 를 넘기므로 자동 보장; 프로덕션과 동일 척도)."""
    if pnl_by_alpha is None:
        raise ValueError("skill 가중은 pnl_by_alpha(알파별 순손익)가 필요")
    score = {}
    for n in names:
        p = pd.Series(pnl_by_alpha[n]).dropna()
        s = M.halflife_weighted_sharpe(p, half_life=int(half_life)) if len(p) >= 2 else 0.0
        s = 0.0 if (s is None or np.isnan(s)) else float(s)
        score[n] = max(s - float(floor), 0.0) ** float(power)
    total = sum(score.values())
    if total <= 0:
        # 전부 floor 밑(다 식음) → 균등 폴백. 전량 청산 대신 중립 보유가 안전.
        return {n: 1.0 / len(names) for n in names}
    w = {n: score[n] / total for n in names}
    if max_weight is not None:
        w = _cap_weights(w, float(max_weight))
    return w


WEIGHTING_REGISTRY = {
    "equal": _wgt_equal,
    "inverse_vol": _wgt_inverse_vol,
    "skill": _wgt_skill,
    # Phase 3: "hedge", "risk_parity", ...
}


def _blend_panels(weight_panels, alpha_w, return_contributions=False):
    """알파별 가중치 패널을 합집합 그리드 위에서 가중합 후 일별 L1=1 재정규화.
    alpha_w = {name: scalar} (정적 가중).

    return_contributions=True 면 (결합북, {alpha -> 기여분패널}) 을 반환한다.
    기여분_m = alpha_w[m]·pos_m / N_d (N_d=결합북 정규화상수) 로, 그 합이 정확히
    결합북과 같다(Σ 기여분 = weights). family 리스크 모듈은 이 '결합북 내 실제
    기여분'을 받아야 스케일이 맞는다 — 원본 L1=1 패널을 그대로 쓰면 L1≈멤버수 라
    L1=1 결합북에서 빼는 순간 gross 가 폭발한다."""
    names = list(weight_panels)
    all_index = None
    all_cols = None
    for w in weight_panels.values():
        all_index = w.index if all_index is None else all_index.union(w.index)
        all_cols = w.columns if all_cols is None else all_cols.union(w.columns)

    combined = pd.DataFrame(0.0, index=all_index, columns=all_cols).sort_index()
    for n in names:
        w = weight_panels[n].reindex(index=all_index, columns=all_cols)
        combined = combined.add(w.fillna(0.0) * alpha_w[n], fill_value=0.0)

    # 포지션 없는 날은 0 유지. scale() 이 전부 0 인 행을 NaN 으로 만드니 0 으로 복원.
    active = combined.abs().sum(axis=1) > 0
    denom = combined.abs().sum(axis=1).replace(0.0, np.nan)
    scaled = combined.div(denom, axis=0).where(active, 0.0)
    if not return_contributions:
        return scaled

    contributions = {}
    for n in names:
        c = weight_panels[n].reindex(index=all_index, columns=all_cols).fillna(0.0) * alpha_w[n]
        contributions[n] = c.div(denom, axis=0).where(active, 0.0)
    return scaled, contributions


def synthesize(weight_panels, method="equal", pnl_by_alpha=None, params=None,
               return_contributions=False):
    """알파별 가중치 패널(dict) -> 결합 가중치 패널(일별 L1=1).
    method 는 WEIGHTING_REGISTRY 에서 조회(config 진입점).

    return_contributions=True 면 (결합북, alpha_w, {alpha->기여분}) 3-튜플 반환.
    family 리스크 모듈에 넘길 스케일 정합 기여분이 필요할 때 사용."""
    names = list(weight_panels)
    if not names:
        raise ValueError("결합할 알파가 없음")
    if method not in WEIGHTING_REGISTRY:
        raise ValueError(f"unknown method {method!r} "
                         f"(available: {sorted(WEIGHTING_REGISTRY)})")
    alpha_w = WEIGHTING_REGISTRY[method](names, pnl_by_alpha=pnl_by_alpha,
                                         **(params or {}))
    if return_contributions:
        combined, contributions = _blend_panels(weight_panels, alpha_w,
                                                return_contributions=True)
        return combined, alpha_w, contributions
    return _blend_panels(weight_panels, alpha_w), alpha_w


def combined_pnl(net_pnls, weights=None):
    """이미 계산된 알파별 순손익을 포트폴리오 순손익으로 결합(엔진 재실행 없이
    빠른 추정). weights 생략 시 동일가중. 날짜 합집합으로 정렬."""
    frame = pd.DataFrame(net_pnls)
    if weights is None:
        return frame.mean(axis=1)
    w = pd.Series(weights)
    w = w / w.sum()
    return (frame * w).sum(axis=1, min_count=1)
