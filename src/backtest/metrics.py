"""메트릭 — 알파 하나의 순손익 시리즈를 점수화(비교/랭킹용).

입력은 engine.py 의 EngineResult. 출력은 메트릭 dict:
    {sharpe, fitness, ic, turnover, net_exposure, mdd, ann_return, days}

쉬운 설명:
  sharpe       위험대비수익. 높을수록 변동 대비 꾸준한 이익.
  ann_return   1단위 북의 연 환산 평균수익(비용 차감 후).
  turnover     하루에 북의 몇 %를 갈아엎는지(높을수록 수수료 많음).
  mdd          자본곡선 최대낙폭(작을수록 안전).
  ic           스피어만 IC 평균 (1·5·10·20일 멀티 호라이즌 동일가중 평균).
  net_exposure 평균 순 롱숏 불균형(~0 = 시장중립).
  fitness      단일 선택점수. 낮은 회전율로 높은 샤프를 벌수록 우대.

모두 순손익(NET)에서 계산 — 수수료/슬리피지/펀딩 이미 차감됨.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.backtest_settings import SETTINGS

_TURNOVER_FLOOR = 0.125  # 거의 거래 안 하는 알파가 무한 fitness 받는 것 방지


def _ann():
    return int(SETTINGS.trading_days_per_year)


def sharpe(net_pnl):
    r = pd.Series(net_pnl).dropna()
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(r.mean() / sd * np.sqrt(_ann()))


def recent_sharpe(net_pnl, window=90):
    """최근 window 일만의 Sharpe. 전 기간 Sharpe 와 비교해 '지금도 유효한지' 확인.
    표본이 window 보다 짧으면 있는 만큼만 사용."""
    r = pd.Series(net_pnl).dropna()
    if len(r) < 2:
        return 0.0
    return sharpe(r.iloc[-int(window):])


def halflife_weighted_sharpe(net_pnl, half_life=90):
    """반감기 지수가중 Sharpe — 최근 수익에 더 큰 가중(half_life 일마다 가중치 절반).
    전 기간을 쓰되 최근을 강조하므로, 창 경계에서 뚝 끊기는 recent_sharpe 보다
    부드럽게 '최근 성향'을 반영. 가중평균/가중표준편차로 계산."""
    r = pd.Series(net_pnl).dropna()
    if len(r) < 2:
        return 0.0
    vals = r.values
    age = np.arange(len(vals))[::-1]          # 0 = 가장 최근
    w = 0.5 ** (age / float(half_life))
    w = w / w.sum()
    mean = float((w * vals).sum())
    sd = float(np.sqrt((w * (vals - mean) ** 2).sum()))
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(mean / sd * np.sqrt(_ann()))


def ann_return(net_pnl):
    r = pd.Series(net_pnl).dropna()
    return float(r.mean() * _ann()) if not r.empty else 0.0


def max_drawdown(net_pnl):
    r = pd.Series(net_pnl).fillna(0.0)
    if r.empty:
        return 0.0
    equity = r.cumsum()
    dd = equity - equity.cummax()
    return float(-dd.min())


def realized_vol(net_pnl):
    """자산변동률 — 순손익(1단위 북)의 연환산 실현변동성.
    equity=net_pnl.cumsum() 의 일변동 크기. 샤프의 분모와 동일 척도.
    리스크 모듈 전/후로 이 값이 얼마나 줄었는지 추적하는 데 쓴다."""
    r = pd.Series(net_pnl).dropna()
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(sd * np.sqrt(_ann()))


def avg_turnover(turnover):
    r = pd.Series(turnover).dropna()
    return float(r.mean()) if not r.empty else 0.0


def avg_net_exposure(net_exposure):
    r = pd.Series(net_exposure).dropna()
    return float(r.abs().mean()) if not r.empty else 0.0


_IC_HORIZONS = (1, 5, 10, 20)


def _ic_for_horizon(positions, returns, horizon: int) -> float:
    """단일 호라이즌 스피어만 IC. forward_return[t] = t부터 horizon일 누적 수익률."""
    fwd_ret = returns.rolling(horizon, min_periods=horizon).sum().shift(-horizon + 1)
    pos = positions.reindex_like(fwd_ret)
    ics = []
    for day in fwd_ret.index:
        p = pos.loc[day]
        r = fwd_ret.loc[day]
        mask = p.notna() & r.notna()
        if mask.sum() < 3:
            continue
        pr = p[mask].rank()
        rr = r[mask].rank()
        if pr.std(ddof=0) == 0 or rr.std(ddof=0) == 0:
            continue
        ics.append(np.corrcoef(pr, rr)[0, 1])
    return float(np.nanmean(ics)) if ics else 0.0


def information_coefficient(positions, returns) -> float:
    """멀티 호라이즌 IC: 1·5·10·20일 스피어만 IC의 동일가중 평균."""
    vals = [_ic_for_horizon(positions, returns, h) for h in _IC_HORIZONS]
    valid = [v for v in vals if not np.isnan(v)]
    return float(np.nanmean(valid)) if valid else 0.0


def fitness(sharpe_val, ann_ret, turnover_val):
    """fitness = sharpe * sqrt(|ann_return| / max(turnover, floor))"""
    denom = max(abs(turnover_val), _TURNOVER_FLOOR)
    return float(sharpe_val * np.sqrt(abs(ann_ret) / denom))


def compute(result):
    s = sharpe(result.net_pnl)
    ar = ann_return(result.net_pnl)
    to = avg_turnover(result.turnover)
    pos, ret = result.positions, result.returns
    horizon_ics = {h: _ic_for_horizon(pos, ret, h) for h in _IC_HORIZONS}
    ic_mean = float(np.nanmean(list(horizon_ics.values())))
    return {
        "sharpe": s,
        "sharpe_recent": recent_sharpe(result.net_pnl),      # 최근 90일
        "sharpe_hl": halflife_weighted_sharpe(result.net_pnl),  # 반감기 90일 가중
        "fitness": fitness(s, ar, to),
        "ic":     ic_mean,
        "ic_1d":  horizon_ics[1],
        "ic_5d":  horizon_ics[5],
        "ic_10d": horizon_ics[10],
        "ic_20d": horizon_ics[20],
        "turnover": to,
        "net_exposure": avg_net_exposure(result.net_exposure),
        "mdd": max_drawdown(result.net_pnl),
        "ann_return": ar,
        "vol": realized_vol(result.net_pnl),
        "days": int(pd.Series(result.net_pnl).notna().sum()),
    }
