"""알파 연산자 — 알파 수식에서 쓸 수 있는 어휘.

모든 연산자는 "date×coin" 패널(index=날짜, columns=코인)을 받아 같은 모양을 돌려준다.
두 종류:
  * 횡단면(cross-sectional, 하루의 코인들 사이): rank, zscore, scale, demean ...
  * 시계열(time-series, 한 코인의 시간축): ts_delta, ts_mean, returns, delay ...

반드시 지켜야 하는 2가지 불변식:
  1. 인과성(CAUSAL) — t일 값은 t 이하 날짜 데이터만 사용. shift(음수)/미래창 금지.
     이것이 백테스트가 가짜로 좋아 보이는(미래참조 누수) 것을 막는 1순위 방어.
  2. NaN 안전 — 아직 상장 안 됐거나 상폐된 코인은 NaN. NaN in -> NaN out 이고
     다른 코인/날짜를 오염시키지 않는다. 롤링은 min_periods=window 로 완전한 창만.

새 연산자 추가: 여기에 함수 작성(인과+NaN안전) -> OPERATORS 에 등록.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_int(n) -> int:
    i = int(n)
    if i != n or i <= 0:
        raise ValueError(f"창/지연은 양의 정수여야 함, got {n!r}")
    return i


# --------------------------- 횡단면 --------------------------- #

def rank(x):
    """그날 코인들 사이 순위를 (0,1] 로. NaN 은 NaN 유지."""
    return x.rank(axis=1, pct=True)


def zscore(x):
    """(x - 그날 평균) / 그날 표준편차. NaN 안전."""
    mu = x.mean(axis=1)
    sd = x.std(axis=1, ddof=0)
    return x.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)


def scale(x):
    """그날 절대값 합이 1이 되도록 정규화 (L1=1)."""
    denom = x.abs().sum(axis=1).replace(0.0, np.nan)
    return x.div(denom, axis=0)


def demean(x):
    """그날 횡단면 평균을 빼기(시장중립). NaN 안전."""
    return x.sub(x.mean(axis=1), axis=0)


def demean_partial(x, beta=1.0):
    """부분 시장중립: 그날 횡단면 평균의 beta 배만 빼기. beta=1 이면 완전중립
    (=demean), beta=0 이면 중립 없음(=raw). 0<beta<1 이면 순노출(방향성)을 일부
    허용 — 불장/베어장 방향성 수익을 포기하지 않으려는 목적(PLAN 2026-07-11 ①)."""
    return x.sub(beta * x.mean(axis=1), axis=0)


# --------------------------- 원소별 --------------------------- #

def neg(x):
    return -x


def abs_(x):
    return x.abs() if isinstance(x, pd.DataFrame) else abs(x)


def sign(x):
    return np.sign(x)


def log(x):
    """자연로그. 0 이하 -> NaN (에러 안 냄)."""
    return np.log(x.where(x > 0))


def power(x, p):
    """부호 유지 거듭제곱."""
    return np.sign(x) * (x.abs() ** float(p))


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    """안전 나눗셈: 0으로 나누면 inf 대신 NaN."""
    if isinstance(b, pd.DataFrame):
        b = b.replace(0.0, np.nan)
    elif b == 0:
        return a * np.nan
    return a / b


# --------------------------- 시계열(인과) --------------------------- #

def delay(x, n):
    """n일 전 값(양의 shift만 — 절대 미래를 보지 않음)."""
    return x.shift(_as_int(n))


def ts_delta(x, n):
    """n일 변화량: x_t - x_{t-n}."""
    n = _as_int(n)
    return x - x.shift(n)


def returns(x, n=1):
    """n일 단순수익률: x_t / x_{t-n} - 1."""
    n = _as_int(n)
    return x / x.shift(n) - 1.0


def ts_sum(x, n):
    n = _as_int(n)
    return x.rolling(n, min_periods=n).sum()


def ts_mean(x, n):
    n = _as_int(n)
    return x.rolling(n, min_periods=n).mean()


def ts_std(x, n):
    n = _as_int(n)
    return x.rolling(n, min_periods=n).std(ddof=0)


def ts_min(x, n):
    n = _as_int(n)
    return x.rolling(n, min_periods=n).min()


def ts_max(x, n):
    n = _as_int(n)
    return x.rolling(n, min_periods=n).max()


def ts_rank(x, n):
    """오늘 값이 최근 n일 창 안에서 몇 등인지, (0,1] 로."""
    n = _as_int(n)

    def _last_rank(a):
        last = a[-1]
        if np.isnan(last):
            return np.nan
        valid = a[~np.isnan(a)]
        return (valid <= last).sum() / valid.size

    return x.rolling(n, min_periods=n).apply(_last_rank, raw=True)


def decay_linear(x, n):
    """선형 가중 이동평균: 오늘 n, ..., t-n+1 일 1. 가중치 합 1. 창 안 NaN -> NaN."""
    n = _as_int(n)
    if n == 1:
        return x.copy()
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()

    def _wmean(a):
        if np.isnan(a).any():
            return np.nan
        return float(np.dot(a, w))

    return x.rolling(n, min_periods=n).apply(_wmean, raw=True)


def ts_corr(x, y, n):
    """최근 n일 피어슨 상관, 코인별. 인과."""
    n = _as_int(n)
    return x.rolling(n, min_periods=n).corr(y)


def csmean(x):
    """횡단면 평균을 구한 뒤 모든 코인 컬럼에 동일하게 broadcast한다."""
    mu = x.mean(axis=1)
    return pd.DataFrame(
        np.broadcast_to(mu.to_numpy()[:, None], x.shape), index=x.index, columns=x.columns
    )


# --------------------------- 레지스트리 --------------------------- #

OPERATORS = {
    # 횡단면
    "rank": rank, "zscore": zscore, "scale": scale, "demean": demean, "csmean": csmean,
    # 원소별
    "neg": neg, "abs": abs_, "sign": sign, "log": log, "power": power,
    "add": add, "sub": sub, "mul": mul, "div": div,
    # 시계열
    "delay": delay, "ts_delta": ts_delta, "delta": ts_delta, "returns": returns,
    "ts_sum": ts_sum, "ts_mean": ts_mean, "ts_std": ts_std,
    "ts_min": ts_min, "ts_max": ts_max, "ts_rank": ts_rank,
    "decay_linear": decay_linear, "ts_corr": ts_corr,
}
