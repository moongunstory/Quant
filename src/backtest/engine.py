"""알파 엔진 — AlphaSpec 하나를 포지션과 순손익으로 바꾼다.

파이프라인:
    expression -> 원점수
               -> neutralize   (none / market)         시장 방향성 제거
               -> scale(L1=1)   매일 sum(|w|)=1          북 크기 정규화
               -> decay         (선택) 가중치 선형 감쇠   회전율 완화
               -> delay(>=1)    오늘 신호로 내일 거래     미래참조 방지
               -> 순손익         gross - 거래비용 + 펀딩

손익 정렬(틀리기 쉬운 부분):
  ret_t        = close_t / close_{t-1} - 1      (t일 동안 실현된 수익률)
  positions_t  = weights_{t-delay}              (delay 일 전에 정한 가중치)
  gross_pnl_t  = sum_i positions_t,i * ret_t,i
  turnover_t   = sum_i |positions_t,i - positions_{t-1},i|
  trade_cost_t = turnover_t * (taker_fee + slippage) / 100
  funding_pnl_t= -sum_i positions_t,i * funding_t,i   (롱은 펀딩 양수면 지불)
  net_pnl_t    = gross_pnl_t - trade_cost_t + funding_pnl_t

포지션이 '지연된 가중치'이므로 t일의 모든 항은 t일 이전 정보만 사용 → 미래참조 없음.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest import operators as ops
from src.config.backtest_settings import SETTINGS
from src.backtest.evaluate import evaluate
from src.backtest.spec import AlphaSpec


@dataclass
class EngineResult:
    spec: AlphaSpec
    weights: pd.DataFrame      # 신호(지연 전), 매일 L1=1
    positions: pd.DataFrame    # 실제 보유 = 지연된 가중치
    returns: pd.DataFrame      # 손익에 쓴 코인별 일수익률
    gross_pnl: pd.Series
    trade_cost: pd.Series
    funding_pnl: pd.Series
    net_pnl: pd.Series
    turnover: pd.Series
    net_exposure: pd.Series

    @property
    def equity(self):
        """누적 순손익(1단위 북 기준 가법)."""
        return self.net_pnl.cumsum()


def compute_weights(spec, panels, universe=None):
    """expression -> neutralize -> scale(L1=1) -> decay. (아직 delay 안 함)"""
    raw = evaluate(spec.expression, panels)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(
            f"expression 은 date×coin 패널로 환원돼야 함, got {type(raw).__name__}"
        )

    if universe is not None:
        # 유니버스 마스크는 float(1.0=포함, 0.0/NaN=제외). reindex 후 >0.5 로
        # bool 화하면 NaN(미포함)은 자동으로 False → 다운캐스팅 경고 없음.
        mask = universe.reindex(index=raw.index, columns=raw.columns) > 0.5
        raw = raw.where(mask)

    neutral = _neutralize(raw, spec.neutralization, panels=panels,
                          universe=universe, beta=spec.neut_beta)
    weights = ops.scale(neutral)

    if spec.decay and spec.decay > 1:
        weights = ops.scale(ops.decay_linear(weights, spec.decay))
    return weights


def run(spec, panels, universe=None, funding_events=None):
    """전체 파이프라인 -> EngineResult (수수료+슬리피지+펀딩 반영 순손익).

    funding_events: 선택. panel.load_funding_events() 의 8h 네이티브 펀딩패널.
    주어지면 펀딩을 8h 정산 이벤트마다 부과(정확). 없으면 panels['funding_rate']
    (일단위)로 폴백."""
    if "close" not in panels:
        raise KeyError("panels 에 'close' 가 있어야 수익률 계산 가능")

    weights = compute_weights(spec, panels, universe=universe)
    return result_from_weights(weights, panels, delay=spec.delay,
                               funding_events=funding_events, spec=spec)


def result_from_weights(weights, panels, delay=1, funding_events=None, spec=None,
                        execution=None, rebalance_band=0.0):
    """이미 만들어진 가중치 패널(신호, 지연 전) -> EngineResult.

    포트폴리오 결합 후 combine/risk 를 거친 최종 가중치에 대해서도 동일한 손익
    회계(gross - 거래비용 + 8h펀딩)를 적용하려고 run() 에서 분리한 것.
    spec 없이(포트폴리오) 호출 가능 — 메트릭 계산은 spec 을 안 쓴다.

    execution: 체결가 가정. None 이면 SETTINGS.execution("close" 기본).
      "close"     positions_t 가 close_{t-1}→close_t 를 번다(결정=체결 종가, 약간 낙관적).
      "next_open" positions_t 가 open_t→open_{t+1} 을 번다(종가 결정 후 '다음 봉 시가'
                  체결 — 실전과 정합, 신뢰도↑). open 패널 필요. 미래참조 없음:
                  positions_t 는 close_{t-1}까지 정보로 정해졌고 수익은 그 이후 실현.
    """
    if "close" not in panels:
        raise KeyError("panels 에 'close' 가 있어야 수익률 계산 가능")

    execution = execution or SETTINGS.execution
    if execution == "next_open":
        if "open" not in panels:
            raise KeyError(
                "execution='next_open' 은 'open' 패널이 필요 — 필드 로드에 'open' 을 "
                "포함하라(QUANT_EXECUTION=next_open 시 자동 포함되도록 파이프라인 처리됨)"
            )
        op = panels["open"].reindex_like(weights)
        returns = op.shift(-1) / op - 1.0   # open_t -> open_{t+1} (다음 봉 시가 체결)
    elif execution == "close":
        close = panels["close"].reindex_like(weights)
        returns = close / close.shift(1) - 1.0
    else:
        raise ValueError(f"알 수 없는 execution {execution!r} (close|next_open)")

    positions = weights.shift(delay)  # 당일 체결 없음

    if rebalance_band > 0.0:
        sim_pos = positions.copy()
        last_w = None
        for idx, row in positions.iterrows():
            if last_w is None:
                if row.notna().any():
                    last_w = row.fillna(0.0)
                continue
            row_filled = row.fillna(0.0)
            drift = (row_filled - last_w).abs().sum()
            if drift < rebalance_band:
                sim_pos.loc[idx] = last_w
            else:
                last_w = row_filled
        positions = sim_pos

    contrib = positions * returns
    gross_pnl = contrib.sum(axis=1, min_count=1)

    prev = positions.shift(1)
    turnover = (positions.fillna(0.0) - prev.fillna(0.0)).abs().sum(axis=1)

    cost_rate = (SETTINGS.taker_fee_pct + SETTINGS.slippage_pct) / 100.0
    trade_cost = turnover * cost_rate

    funding_pnl = _funding_pnl(positions, panels, funding_events=funding_events)

    net_pnl = gross_pnl.fillna(0.0) - trade_cost + funding_pnl
    net_exposure = positions.sum(axis=1, min_count=1)

    return EngineResult(
        spec=spec, weights=weights, positions=positions, returns=returns,
        gross_pnl=gross_pnl, trade_cost=trade_cost, funding_pnl=funding_pnl,
        net_pnl=net_pnl, turnover=turnover, net_exposure=net_exposure,
    )


def _neutralize(raw, mode, panels=None, universe=None, beta=1.0):
    if mode == "none":
        return raw
    if mode == "market":
        return ops.demean(raw)
    if mode == "partial":
        # 부분 중립: 순노출을 beta 만큼만 제거(방향성 일부 허용). beta=1 이면 market 과 동일.
        return ops.demean_partial(raw, beta)
    if mode == "turnover_rank":
        if panels is None or "quote_volume" not in panels:
            raise ValueError("neutralization='turnover_rank' 를 위해서는 quote_volume 패널이 필요합니다.")
        quote_volume = panels["quote_volume"]
        n_q = 5
        lookback = 30
        
        turnover = quote_volume.rolling(lookback, min_periods=1).mean()
        if universe is not None:
            # universe 는 float 마스크(1.0/0.0/NaN) — where() 는 boolean 조건을 요구하므로
            # >0.5 로 bool 화(NaN=미포함→False). compute_weights 의 마스킹과 동일 패턴.
            turnover = turnover.where(universe.reindex_like(turnover) > 0.5)
            
        out = raw.copy()
        for day in raw.index:
            row = raw.loc[day]
            t_row = turnover.loc[day]
            valid = row.notna() & t_row.notna()
            n_valid = int(valid.sum())
            if n_valid == 0:
                continue
            bins = min(n_q, n_valid)
            try:
                bucket = pd.qcut(t_row[valid], q=bins, labels=False, duplicates="drop")
                means = row[valid].groupby(bucket).transform("mean")
                out.loc[day, means.index] = row[valid] - means
            except ValueError:
                continue
        return out
    raise ValueError(f"알 수 없는 neutralization {mode!r}")


def _funding_pnl(positions, panels, funding_events=None):
    """펀딩 비용: 롱(w>0)은 funding>0 일 때 지불.

    우선순위:
      1) funding_events(8h 네이티브)가 있으면 → 각 8h 정산시각에 '그 순간 보유
         포지션'에만 부과 후 일단위로 집계 (정확, 일별 sum 이 아님).
      2) 없으면 panels['funding_rate'](일단위)로 폴백.
    둘 다 없으면 0."""
    if funding_events is not None and not funding_events.empty:
        return _funding_pnl_events(positions, funding_events)

    funding = panels.get("funding_rate")
    if funding is None:
        return pd.Series(0.0, index=positions.index)
    funding = funding.reindex_like(positions)
    paid = (positions * funding).sum(axis=1, min_count=1).fillna(0.0)
    return -paid


def _funding_pnl_events(positions, funding_events):
    """8h 정산 이벤트 기반 펀딩 손익 -> positions.index(임의 grid)로 집계.

    각 정산시각(00/08/16, 일부 코인 4h)에 실제로 보유한 포지션 = 그 시각 직전(포함)
    가장 최근 positions 행이므로, 두 인덱스를 합쳐 ffill 한다. 그리고 각 정산을 그것이
    속한 '포지션 버킷'(직전 positions 타임스탬프)으로 묶어 합산한다.

    일 그리드(1d)에선 버킷=자정 → 하루 3정산이 그날 포지션에 걸려 정확한 일별 펀딩이
    되고, Phase 3 인트라데이 그리드(1h/8h 등)에선 각 bar 버킷으로 정확히 배분된다.
    포지션이 bar 안에서 상수든 아니든 같은 코드로 정확."""
    ev = funding_events.reindex(columns=positions.columns)
    ev = ev.loc[(ev.index >= positions.index.min())]  # 포지션 시작 전 정산 무시
    if ev.empty:
        return pd.Series(0.0, index=positions.index)
    union = positions.index.union(ev.index)
    pos_on_ev = positions.reindex(union).ffill().reindex(ev.index)
    paid_ev = (pos_on_ev * ev).sum(axis=1, min_count=1)          # 정산별 지불액
    # 각 정산 -> 직전(포함) positions 버킷으로 라벨링(임의 grid 일반화)
    bucket = pd.Series(positions.index, index=positions.index) \
        .reindex(union).ffill().reindex(ev.index)
    grouped = (-paid_ev).groupby(bucket).sum()
    return grouped.reindex(positions.index).fillna(0.0)
