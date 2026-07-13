"""risk — 결합된 포트폴리오 가중치에 씌우는 안전 오버레이.

coin research/risk/risk.py 를 Quant 로 이식(Phase 1 스코프). config 구동
파이프라인(RISK_REGISTRY): 어떤 모듈이/어떤 순서로/어떤 임계값으로 도는지는
portfolio.json 의 "risk_pipeline" 리스트가 전부 정한다 — 코드 수정 없이
추가/제거/재정렬/튜닝.

Phase 1 모듈(현재 이식됨):
  position_cap      단일 코인 최대 가중치 클립(한 코인 폭발 방지).
  participation_cap 코인 자기 ADV(평균일거래대금)의 N%까지만 — 유동성 낮은 코인 축소.
  market_neutrality |Σw| 를 한계 내로 — 방향이 아니라 상대 움직임에 베팅.
  vol_target        북 전체를 목표 변동성에 맞춰 스케일(과거 변동성만, causal).
  mdd_killswitch    낙폭 한계 넘으면 노출 0, 회복하면 점진 재진입(안전벨트).
  killswitch_v2     낙폭 '속도' 기반 2세대 킬스위치(느린 all-time-peak 락아웃 해결).
  gross_cap         총 레버리지 하드 상한(어떤 config로도 몰래 레버 금지).
  lot_rounding      거래 최소 단위로 반올림.

Phase 3 추가: downside_vol_target(하방변동성만 반응), alpha_decay_gate(추세샤프 게이트).
미이식(components/families 필요): family_corr_scale, family_gross_cap, top_n_execution_universe.
(Quant 는 top-100 유니버스 마스크가 실행스코프를 이미 담당 → top_n_execution_universe 불필요.)

모듈 계약(공통 인터페이스): f(weights, ctx, **params) -> weights
ctx 는 RiskContext(수익률 패널 + causal PnL 헬퍼 + 선택적 quote_volume).
모든 계산은 causal: 오늘 스케일은 어제까지 정보만 사용, 미래참조 없음.
기본 임계값은 config/backtest_settings.py(매직넘버 금지).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from src.config.backtest_settings import SETTINGS
from src.backtest import metrics as _metrics


def _ann_days():
    return int(SETTINGS.trading_days_per_year)


# --------------------------------------------------------------------------- #
# primitives (순수 함수)
# --------------------------------------------------------------------------- #

def cap_positions(weights, cap=None):
    """모든 가중치를 ±cap 으로 클립. NaN(미보유)은 NaN 유지."""
    cap = SETTINGS.position_cap if cap is None else cap
    return weights.clip(lower=-cap, upper=cap)


def participation_cap_positions(weights, quote_volume, participation_rate=None,
                                lookback=20, aum_usd=None):
    """각 코인의 포지션이 '그 코인 자기 ADV(평균일거래대금)의 participation_rate'
    를 넘지 않도록 코인별·일별 동적 클립. causal(ADV 는 오늘까지의 데이터).

    aum_usd: 가정 총 북 규모(USD). 이 엔진은 상대가중(L1=1)만 다루므로, 실제
    달러 ADV 와 비교하려면 북 전체의 달러 스케일이 필요하다."""
    rate = SETTINGS.participation_rate if participation_rate is None else participation_rate
    aum = SETTINGS.book_aum_usd if aum_usd is None else aum_usd
    lookback = int(lookback)
    adv = quote_volume.reindex_like(weights).rolling(lookback, min_periods=1).mean()
    cap = (rate * adv / aum).clip(upper=1.0)   # 북의 100% 이상은 불필요
    return weights.clip(lower=-cap, upper=cap)


def limit_net_exposure(weights, limit=None):
    """그날의 순 롱숏 불균형을 ±limit 내로 — 초과분을 그날 실제 보유한 코인들에
    균등하게 덜어낸다."""
    limit = SETTINGS.net_exposure_limit if limit is None else limit
    net = weights.sum(axis=1)
    excess = net - net.clip(lower=-limit, upper=limit)   # 한계 내면 0
    held = weights.notna() & (weights != 0)
    count = held.sum(axis=1).replace(0, np.nan)
    per_coin = (excess / count)
    adjustment = held.mul(per_coin, axis=0).fillna(0.0)
    return weights - adjustment


def realized_vol(pnl, lookback=30):
    """일 순손익의 추세 연환산 변동성(causal: 오늘 제외 shift(1))."""
    r = pd.Series(pnl).fillna(0.0)
    daily = r.rolling(lookback, min_periods=5).std(ddof=0).shift(1)
    return daily * np.sqrt(_ann_days())


def vol_target_scale(pnl, target_annual=None, lookback=30, max_scale=5.0,
                     min_scale=0.0):
    """실현변동성이 목표를 추종하도록 하는 일별 배수. 과거 PnL만 사용 → 미래참조
    없음. 히스토리 부족 시 1.0. max_scale 로 과도 레버 방지. min_scale 은
    디레버리징 하한(vol_target 은 사이즈 정규화지 크래시 방어가 아님 — 그건
    killswitch 몫). 0.0 = 무제한 디레버리징."""
    target_annual = SETTINGS.target_annual_vol if target_annual is None else target_annual
    if not (0.0 <= min_scale <= max_scale):
        raise ValueError(f"need 0 <= min_scale <= max_scale, got {min_scale}/{max_scale}")
    vol = realized_vol(pnl, lookback=lookback)
    scale = (target_annual / vol).clip(lower=min_scale, upper=max_scale)
    return scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def mdd_killswitch_scale(pnl, mdd_kill=None, mdd_reentry=None, rampup_days=None,
                         peak_lookback=None):
    """[0,1] 일별 노출배수. 낙폭이 -mdd_kill 을 깨면 OFF, SHADOW equity(안 자른
    전략이 갈 궤적)가 -mdd_reentry 내로 회복하면 램프업 재진입.

    shadow equity 로 회복을 감지하는 게 핵심: 0으로 자르면 실현 equity 가 안 움직여
    스스로 회복 못 함 → 원 전략을 대신 관찰. (scale, managed_pnl) 반환.

    peak_lookback: None=전고점 기준(구버전, 대형 구조적 고점 후 몇 년간 OFF 위험).
    int=추세창 최대값 기준 → 참조 낙폭이 시간이 지나면 소멸해 스위치 재무장. causal."""
    mdd_kill = SETTINGS.mdd_kill if mdd_kill is None else mdd_kill
    mdd_reentry = SETTINGS.mdd_reentry if mdd_reentry is None else mdd_reentry
    rampup_days = SETTINGS.rampup_days if rampup_days is None else rampup_days
    step = 1.0 / max(int(rampup_days), 1)

    r = pd.Series(pnl).fillna(0.0)
    idx = r.index
    scale = pd.Series(1.0, index=idx)
    managed = pd.Series(0.0, index=idx)

    if peak_lookback is not None:
        peak_lookback = int(peak_lookback)
        if peak_lookback < 1:
            raise ValueError(f"peak_lookback must be >= 1 or None, got {peak_lookback}")
    hist = deque(maxlen=peak_lookback)

    shadow_equity = 0.0
    shadow_peak = 0.0
    state = "on"
    cur = 1.0
    for i in range(len(r)):
        scale.iloc[i] = cur
        managed.iloc[i] = r.iloc[i] * cur

        shadow_equity += r.iloc[i]
        if peak_lookback is None:
            shadow_peak = max(shadow_peak, shadow_equity)
        else:
            hist.append(shadow_equity)
            shadow_peak = max(hist)
        sdd = shadow_equity - shadow_peak

        if state == "on":
            if sdd <= -mdd_kill:
                state, cur = "off", 0.0
        elif state == "off":
            if sdd >= -mdd_reentry:
                state, cur = "ramp", step
        elif state == "ramp":
            cur = min(1.0, cur + step)
            if cur >= 1.0:
                state, cur = "on", 1.0

    return scale, managed


def killswitch_v2_scale(pnl, velocity_kill=0.10, velocity_window=10,
                        mdd_kill=None, stabilize_days=20,
                        stabilize_tolerance=0.05, rampup_days=None):
    """2세대 킬스위치 — mdd_killswitch 의 '전고점 대비 회복' 재진입이 유발하는
    영구 락아웃을 구조적으로 대체.

    트리거(둘 중 하나 발화 → OFF):
      velocity : shadow equity 가 최근 velocity_window 일 동안 velocity_kill 이상
                 하락 — 급락 포착(vol_target 이 늦게 반응하는 구간). 주 트리거.
      magnitude: shadow equity 가 WATERMARK 대비 mdd_kill 이상 아래 — 느린 출혈
                 백스톱. watermark 는 전고점이 아니라 재진입마다 리셋(영구기억 없음).
                 None 이면 비활성.
    재진입(OFF 중): 최근 velocity_window 일 변화가 -stabilize_tolerance 위인 CALM
      일수를 센다. stabilize_days 연속 CALM 이면 watermark 를 현재 equity 로 리셋하고
      rampup_days 에 걸쳐 램프업(램프 중 재트리거 시 다시 컷). causal."""
    mdd_kill = SETTINGS.mdd_kill if mdd_kill is None else mdd_kill
    rampup_days = SETTINGS.rampup_days if rampup_days is None else rampup_days
    velocity_window = int(velocity_window)
    stabilize_days = int(stabilize_days)
    if velocity_kill <= 0:
        raise ValueError(f"velocity_kill must be > 0, got {velocity_kill}")
    if stabilize_tolerance <= 0:
        raise ValueError(f"stabilize_tolerance must be > 0, got {stabilize_tolerance}")
    if velocity_window < 1:
        raise ValueError(f"velocity_window must be >= 1, got {velocity_window}")
    if stabilize_days < 1:
        raise ValueError(f"stabilize_days must be >= 1, got {stabilize_days}")
    step = 1.0 / max(int(rampup_days), 1)

    r = pd.Series(pnl).fillna(0.0)
    idx = r.index
    scale = pd.Series(1.0, index=idx)
    managed = pd.Series(0.0, index=idx)

    equity_path = []
    equity = 0.0
    watermark = 0.0
    state = "on"
    cur = 1.0
    calm = 0
    for i in range(len(r)):
        scale.iloc[i] = cur
        managed.iloc[i] = r.iloc[i] * cur

        equity += r.iloc[i]
        equity_path.append(equity)
        j = i - velocity_window
        vel = equity - (equity_path[j] if j >= 0 else 0.0)
        fast_crash = vel <= -velocity_kill

        if state in ("on", "ramp"):
            watermark = max(watermark, equity)
            deep = mdd_kill is not None and (equity - watermark) <= -mdd_kill
            if fast_crash or deep:
                state, cur, calm = "off", 0.0, 0
            elif state == "ramp":
                cur = min(1.0, cur + step)
                if cur >= 1.0:
                    state, cur = "on", 1.0
        else:  # off
            calm = calm + 1 if vel > -stabilize_tolerance else 0
            if calm >= stabilize_days:
                watermark = equity
                state, cur = "ramp", step

    return scale, managed


def round_lots(weights, step=None):
    """각 가중치를 거래 가능한 lot step 으로 반올림. NaN 유지. step<=0 이면 그대로."""
    step = SETTINGS.lot_step if step is None else step
    if step <= 0:
        return weights
    return (weights / step).round() * step


def downside_realized_vol(pnl, lookback=30):
    """추세 연환산 '하방' 변동성(semi-deviation). 음(-)의 날만 기여, sqrt(2) 보정으로
    대칭분포에선 일반 변동성과 맞춰 target_annual 비교 가능. causal(shift 1)."""
    r = pd.Series(pnl).fillna(0.0)
    neg = r.where(r < 0.0, 0.0)
    daily = neg.rolling(lookback, min_periods=5).std(ddof=0).shift(1)
    return daily * np.sqrt(_ann_days()) * np.sqrt(2.0)


def downside_vol_target_scale(pnl, target_annual=None, lookback=30,
                              max_scale=1.0, min_scale=0.5):
    """하방 변동성에만 반응하는 사이징(대칭 vol_target 이 '이익발 변동성'까지 벌주는
    문제 회피). 손실이 출렁일 때만 노출 축소, 이익발 변동성은 유지. 히스토리 부족 시 1.0."""
    target_annual = SETTINGS.target_annual_vol if target_annual is None else target_annual
    if not (0.0 <= min_scale <= max_scale):
        raise ValueError(f"need 0 <= min_scale <= max_scale, got {min_scale}/{max_scale}")
    vol = downside_realized_vol(pnl, lookback=lookback)
    scale = (target_annual / vol).clip(lower=min_scale, upper=max_scale)
    return scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)


def alpha_decay_gate_scale(pnl, window=126, sharpe_off=0.0, sharpe_on=0.5,
                           rampup_days=None):
    """추세 샤프(causal, window일 롤링)가 sharpe_off 아래로 떨어지면 노출 0,
    sharpe_on 위로 회복하면 램프업 재진입. 히스테리시스(off<on)로 플립플롭 방지.
    가격 경로가 아니라 '전략 품질'(위험조정수익)을 봐서, 느린 구조적 출혈(알파 소멸)을
    잡는다 — killswitch(급락/깊은낙폭)의 사각지대 보완."""
    if not (sharpe_off < sharpe_on):
        raise ValueError(f"need sharpe_off < sharpe_on, got {sharpe_off}/{sharpe_on}")
    window = int(window)
    if window < 20:
        raise ValueError(f"window must be >= 20, got {window}")
    rampup_days = SETTINGS.rampup_days if rampup_days is None else rampup_days
    step = 1.0 / max(int(rampup_days), 1)

    r = pd.Series(pnl).fillna(0.0)
    mu = r.rolling(window, min_periods=window).mean()
    sd = r.rolling(window, min_periods=window).std(ddof=0)
    tsharpe = (mu / sd.replace(0.0, np.nan) * np.sqrt(_ann_days())).shift(1)  # causal

    scale = pd.Series(1.0, index=r.index)
    state, cur = "on", 1.0
    for i in range(len(r)):
        scale.iloc[i] = cur
        ts = tsharpe.iloc[i]
        if np.isnan(ts):
            continue
        if state in ("on", "ramp"):
            if ts <= sharpe_off:
                state, cur = "off", 0.0
            elif state == "ramp":
                cur = min(1.0, cur + step)
                if cur >= 1.0:
                    state, cur = "on", 1.0
        else:  # off
            if ts >= sharpe_on:
                state, cur = "ramp", step
    return scale


# --------------------------------------------------------------------------- #
# family 상관 (coin research/risk/risk.py 이식, D14)
# --------------------------------------------------------------------------- #

def rolling_family_corr(component_pnls, lookback=60):
    """한 패밀리 멤버들의 일 PnL 사이 '트레일링 평균 쌍별 상관'을 하루 한 값으로.
    정적 선택시점 패밀리 캡이 못 보는 '백테스트 중간에 급등하는 상관'을 잡는다.
    causal: 창이 t-1 에서 끝남(shift(1)). 멤버 2 미만이면 0."""
    df = pd.DataFrame(component_pnls).fillna(0.0)
    if df.shape[1] < 2:
        return pd.Series(0.0, index=df.index)
    corr = df.rolling(lookback).corr()          # long-format rolling corr(MultiIndex)
    n = df.shape[1]
    # 하루 n×n 상관행렬 합 = n(대각) + 비대각 합; 평균 쌍별 = (합-n)/(n(n-1)).
    daily_sum = corr.groupby(level=0).sum().sum(axis=1)
    avg_pairwise = (daily_sum - n) / (n * (n - 1))
    avg_pairwise = avg_pairwise.reindex(df.index)
    return avg_pairwise.shift(1)                 # causal


def effective_n(n_members, avg_corr):
    """평균 쌍별 상관이 avg_corr 일 때 n_members 의 '유효 독립 베팅 수'(분산비율
    공식: 완전상관=1, 완전독립=n_members). 정적 카운트 캡이 못 보는, 멤버 수는
    그대로인데 상관만 급등해 유효 n 이 1로 붕괴하는 상황을 감지."""
    avg_corr = pd.Series(avg_corr).clip(lower=-1.0 / max(n_members - 1, 1))
    return n_members / (1.0 + (n_members - 1) * avg_corr)


# --------------------------------------------------------------------------- #
# pluggable pipeline
# --------------------------------------------------------------------------- #

class RiskContext:
    """리스크 파이프라인을 통해 공유되는 상태.

    returns      : date x coin 수익률 패널.
    quote_volume : date x coin 거래대금 패널(participation_cap 에만 필요, 없으면 None).
    pnl(w)       : 가중치 패널의 causal 일 PnL. panels 가 주어지면 엔진과 '동일한'
                   회계(거래비용 + 8h 펀딩, delay=0 — 파이프라인 가중치는 이미 delay
                   반영됨)로 계산 → stage 리포트가 최종 포트폴리오 지표와 정합.
                   panels 가 없으면 가격수익률만의 근사(어제 가중치×오늘 수익률).
    note()       : 모듈이 stage 리포트용 진단을 남긴다."""

    def __init__(self, returns, quote_volume=None, components=None, families=None,
                 panels=None, funding_events=None):
        self.returns = returns
        self.quote_volume = quote_volume
        # panels/funding_events: 주어지면 pnl() 이 엔진과 동일한 net 회계를 쓴다.
        # 없으면 가격수익률만의 gross 근사로 폴백(하위호환).
        self.panels = panels
        self.funding_events = funding_events
        # components : {alpha_name -> weight_panel} 블렌드 전 알파별 가중치.
        #              family-scoped 모듈(family_corr_scale/gross_cap)에만 필요.
        # families   : {alpha_name -> family_name} (data/alpha_families.json).
        # family_state: family-scoped 모듈이 방금 축소한 패밀리 슬라이스를 기록 →
        #              다음 family-scoped 모듈이 원본이 아니라 '이미 축소된' 슬라이스
        #              위에서 계산하게 함(coin D14 fix).
        self.components = components
        self.families = families
        self.family_state = {}
        self._notes = {}

    def pnl(self, weights):
        if self.panels is not None:
            # 엔진과 동일 회계: 비용+8h펀딩 반영 net_pnl. 파이프라인 가중치는 이미
            # delay 반영이므로 delay=0(추가 지연 없음, 미래참조 없음).
            from src.backtest import engine
            return engine.result_from_weights(
                weights, self.panels, delay=0,
                funding_events=self.funding_events).net_pnl
        return (weights.shift(1) * self.returns.reindex_like(weights)) \
            .sum(axis=1, min_count=1).fillna(0.0)

    def note(self, **kv):
        self._notes.update(kv)

    def pop_notes(self):
        n, self._notes = self._notes, {}
        return n


def _rm_position_cap(weights, ctx, max_weight=None):
    cap_val = SETTINGS.position_cap if max_weight is None else max_weight
    out = cap_positions(weights, cap=cap_val)
    total = weights.notna().sum().sum()
    hits = int((weights.abs() > (cap_val + 1e-7)).sum().sum())
    ctx.note(limit=cap_val, cap_triggered_count=hits,
             cap_triggered_pct=(hits / total * 100) if total else 0.0)
    return out


def _rm_participation_cap(weights, ctx, participation_rate=None, lookback=20,
                          aum_usd=None):
    if ctx.quote_volume is None:
        raise ValueError(
            "risk type 'participation_cap' 는 quote_volume 패널이 필요 — "
            "run_risk_pipeline(..., quote_volume=...) 로 넘겨라"
        )
    rate = SETTINGS.participation_rate if participation_rate is None else participation_rate
    aum = SETTINGS.book_aum_usd if aum_usd is None else aum_usd
    out = participation_cap_positions(weights, ctx.quote_volume,
                                      participation_rate=rate, lookback=lookback,
                                      aum_usd=aum)
    total = weights.notna().sum().sum()
    hits = int((out.sub(weights).abs() > 1e-9).sum().sum())
    ctx.note(limit=rate, aum_usd=aum, lookback=int(lookback),
             cap_triggered_count=hits,
             cap_triggered_pct=(hits / total * 100) if total else 0.0)
    return out


def _rm_market_neutrality(weights, ctx, tolerance=None):
    tol = SETTINGS.net_exposure_limit if tolerance is None else tolerance
    out = limit_net_exposure(weights, limit=tol)
    days = len(weights)
    hits = int((weights.sum(axis=1).abs() > (tol + 1e-7)).sum())
    ctx.note(limit=tol, net_triggered_days=hits,
             net_triggered_pct=(hits / days * 100) if days else 0.0)
    return out


def _rm_vol_target(weights, ctx, target_annual_vol=None, lookback=30, max_scale=5.0,
                   min_scale=0.0):
    pnl = ctx.pnl(weights)
    vscale = vol_target_scale(pnl, target_annual=target_annual_vol,
                              lookback=lookback, max_scale=max_scale,
                              min_scale=min_scale)
    ctx.note(limit=SETTINGS.target_annual_vol if target_annual_vol is None else target_annual_vol,
             avg_scale=float(vscale.mean()), min_scale=float(vscale.min()),
             max_scale=float(vscale.max()))
    return weights.mul(vscale, axis=0)


def _rm_mdd_killswitch(weights, ctx, threshold=None, reentry=None, rampup_days=None,
                       peak_lookback=None):
    pnl = ctx.pnl(weights)
    kscale, _ = mdd_killswitch_scale(pnl, mdd_kill=threshold,
                                     mdd_reentry=reentry, rampup_days=rampup_days,
                                     peak_lookback=peak_lookback)
    days = len(kscale)
    kill = int((kscale == 0.0).sum())
    ramp = int(((kscale > 0.0) & (kscale < 1.0)).sum())
    ctx.note(limit=SETTINGS.mdd_kill if threshold is None else threshold,
             kill_days=kill, kill_pct=(kill / days * 100) if days else 0.0,
             ramp_days=ramp, ramp_pct=(ramp / days * 100) if days else 0.0)
    return weights.mul(kscale, axis=0)


def _rm_killswitch_v2(weights, ctx, velocity_kill=0.10, velocity_window=10,
                      threshold=None, stabilize_days=20, stabilize_tolerance=0.05,
                      rampup_days=None):
    """threshold 는 크기 백스톱(mdd_kill)에 매핑; JSON null 이면 비활성 →
    velocity 트리거만 남는다."""
    pnl = ctx.pnl(weights)
    kscale, _ = killswitch_v2_scale(pnl, velocity_kill=velocity_kill,
                                    velocity_window=velocity_window,
                                    mdd_kill=threshold,
                                    stabilize_days=stabilize_days,
                                    stabilize_tolerance=stabilize_tolerance,
                                    rampup_days=rampup_days)
    days = len(kscale)
    kill = int((kscale == 0.0).sum())
    ramp = int(((kscale > 0.0) & (kscale < 1.0)).sum())
    reentries = int(((kscale.shift(1) == 0.0) & (kscale > 0.0)).sum())
    ctx.note(limit=velocity_kill, velocity_window=int(velocity_window),
             kill_days=kill, kill_pct=(kill / days * 100) if days else 0.0,
             ramp_days=ramp, ramp_pct=(ramp / days * 100) if days else 0.0,
             reentries=reentries)
    return weights.mul(kscale, axis=0)


def _rm_gross_cap(weights, ctx, max_gross=1.0):
    """총 레버리지 하드 불변식: gross=Σ|w| 가 max_gross 초과 시 북 전체 비례
    축소. L1=1 블렌드 + 상류 스케일러 <=1 이면 보통 no-op — 어떤 미래 config
    조합으로도 몰래 레버가 걸리지 않게 하는 안전장치."""
    gross = weights.abs().sum(axis=1, min_count=1)
    scale = (max_gross / gross).clip(upper=1.0).fillna(1.0)
    days = int((scale < 1.0).sum())
    ctx.note(limit=max_gross, days_capped=days,
             days_capped_pct=(days / len(scale) * 100) if len(scale) else 0.0,
             avg_gross_before=float(gross.mean()) if len(gross) else 0.0)
    return weights.mul(scale, axis=0)


def _rm_lot_rounding(weights, ctx, step=None):
    ctx.note(limit=SETTINGS.lot_step if step is None else step)
    return round_lots(weights, step=step)


def _rm_downside_vol_target(weights, ctx, target_annual_vol=None, lookback=30,
                            max_scale=1.0, min_scale=0.5):
    pnl = ctx.pnl(weights)
    vscale = downside_vol_target_scale(pnl, target_annual=target_annual_vol,
                                       lookback=lookback, max_scale=max_scale,
                                       min_scale=min_scale)
    ctx.note(limit=SETTINGS.target_annual_vol if target_annual_vol is None else target_annual_vol,
             avg_scale=float(vscale.mean()), min_scale=float(vscale.min()),
             max_scale=float(vscale.max()))
    return weights.mul(vscale, axis=0)


def _rm_alpha_decay_gate(weights, ctx, window=126, sharpe_off=0.0, sharpe_on=0.5,
                         rampup_days=None):
    pnl = ctx.pnl(weights)
    scale = alpha_decay_gate_scale(pnl, window=window, sharpe_off=sharpe_off,
                                   sharpe_on=sharpe_on, rampup_days=rampup_days)
    days = len(scale)
    off_days = int((scale == 0.0).sum())
    ctx.note(limit=sharpe_off, sharpe_on=sharpe_on, window=int(window),
             gate_off_days=off_days,
             gate_off_pct=(off_days / days * 100) if days else 0.0)
    return weights.mul(scale, axis=0)


def _family_slices(weights, ctx):
    """{family -> (members, family_weight_panel)} — 각 패밀리의 현재 북 슬라이스.
    멤버 2 미만 또는 ctx.components 에 없는 패밀리는 건너뜀. family_corr_scale 과
    family_gross_cap 이 공유.

    D14 fix: 앞선 family-scoped 모듈이 이미 이 패밀리를 축소했다면(ctx.family_state
    에 기록됨) 원본 components 가 아니라 '그 축소된 슬라이스'를 현재 가중으로 재사용.
    안 그러면 두 번째 family 모듈이 축소 안 된 노출 기준으로 또 클립해 첫 모듈 효과를
    상쇄한다(관측: +0.47 sharpe -> 직후 -0.68). 가장 먼저 도는 모듈은 원본 fallback."""
    if ctx.components is None or ctx.families is None:
        raise ValueError(
            "이 리스크 모듈은 ctx.components + ctx.families 가 필요 — "
            "run_risk_pipeline(..., components={알파:가중치패널}, "
            "families={알파:패밀리}) 로 넘겨라 (coin D14)"
        )
    fam_members = {}
    for name, fam in ctx.families.items():
        if name in ctx.components:
            fam_members.setdefault(fam, []).append(name)

    out = {}
    for fam, members in fam_members.items():
        if len(members) < 2:
            continue
        cached = ctx.family_state.get(fam)
        if cached is not None:
            fam_weight = cached.reindex_like(weights).fillna(0.0)
        else:
            fam_weight = sum(ctx.components[m].reindex_like(weights).fillna(0.0)
                             for m in members)
        out[fam] = (members, fam_weight)
    return out


def _rm_family_corr_scale(weights, ctx, lookback=60, target_effective_n=None,
                          min_scale=0.3, max_scale=1.0):
    """패밀리 멤버들의 트레일링 PnL 상관이 급등하면 그 패밀리 슬라이스를 축소
    (선택시점에 한 번 정해지는 정적 카운트 캡 대신). causal.

    PROVISIONAL: lookback/min_scale/max_scale 는 아직 Quant 데이터로 검증 안 됨 —
    portfolio.json 에서 기본 비활성(enabled:false)로 두고 stage 리포트로 확인 후 켠다.

    target_effective_n: 원하는 패밀리 내 다양성. None 이면 n_members(측정된 상관이
    조금이라도 있으면 노출 축소). scale=clip(effective_n/target, min,max) 를 그날
    가중치 중 그 패밀리 기여분에만 적용 → 앞선 stage 와 정합적으로 합성."""
    slices = _family_slices(weights, ctx)
    out = weights.copy()
    notes = {}
    for fam, (members, fam_weight) in slices.items():
        pnls = {m: ctx.pnl(ctx.components[m]) for m in members}
        avg_corr = rolling_family_corr(pnls, lookback=lookback).reindex(weights.index)
        n = len(members)
        eff_n = effective_n(n, avg_corr)
        target = n if target_effective_n is None else target_effective_n
        scale = (eff_n / target).clip(lower=min_scale, upper=max_scale).fillna(1.0)
        out = out - fam_weight.mul(1.0 - scale, axis=0)
        ctx.family_state[fam] = fam_weight.mul(scale, axis=0)  # D14
        notes[fam] = {
            "n_members": n,
            "avg_corr_last": float(avg_corr.iloc[-1]) if len(avg_corr) and pd.notna(avg_corr.iloc[-1]) else None,
            "scale_last": float(scale.iloc[-1]) if len(scale) else None,
            "scale_min": float(scale.min()) if len(scale) else None,
        }
    ctx.note(lookback=lookback, min_scale=min_scale, max_scale=max_scale,
             families=notes)
    return out


def _rm_family_gross_cap(weights, ctx, max_gross_per_family=0.5):
    """한 패밀리에 귀속되는 sum(|weight|) 의 하드 상한 — family_corr_scale 의 백스톱
    (killswitch_v2 가 vol_target 에 대해 갖는 관계와 동일). 롤링 창 지연 없이 즉시
    반응하지만 부드러운 스케일이 아니라 뭉툭한 클립.

    PROVISIONAL 기본값(0.5): 아직 Quant 데이터로 미검증."""
    slices = _family_slices(weights, ctx)
    out = weights.copy()
    notes = {}
    for fam, (members, fam_weight) in slices.items():
        gross = fam_weight.abs().sum(axis=1, min_count=1)
        scale = (max_gross_per_family / gross).clip(upper=1.0).fillna(1.0)
        out = out - fam_weight.mul(1.0 - scale, axis=0)
        ctx.family_state[fam] = fam_weight.mul(scale, axis=0)  # D14
        days_capped = int((scale < 1.0).sum())
        notes[fam] = {"n_members": len(members), "days_capped": days_capped,
                      "avg_gross_before": float(gross.mean()) if len(gross) else 0.0}
    ctx.note(limit=max_gross_per_family, families=notes)
    return out


RISK_REGISTRY = {
    "position_cap": _rm_position_cap,
    "participation_cap": _rm_participation_cap,
    "market_neutrality": _rm_market_neutrality,
    "vol_target": _rm_vol_target,
    "mdd_killswitch": _rm_mdd_killswitch,
    "killswitch_v2": _rm_killswitch_v2,
    "gross_cap": _rm_gross_cap,
    "lot_rounding": _rm_lot_rounding,
    "downside_vol_target": _rm_downside_vol_target,
    "alpha_decay_gate": _rm_alpha_decay_gate,
    "family_corr_scale": _rm_family_corr_scale,
    "family_gross_cap": _rm_family_gross_cap,
}


def run_risk_pipeline(weights, returns, pipeline_cfg, quote_volume=None,
                      components=None, families=None,
                      panels=None, funding_events=None):
    """설정된 리스크 모듈을 순서대로 적용하고 {weights, pnl, stages} 반환.

    pipeline_cfg : [{"type","enabled","params"}, ...] (실행순서=리스트순서).
                   비활성 항목은 건너뛰되 stages 에는 남겨 파이프라인 모양을 보인다.
    quote_volume : participation_cap 포함 시에만 필요.
    components   : {alpha_name -> weight_panel} 블렌드 전 알파별 가중치 —
                   family_corr_scale/family_gross_cap 포함 시에만 필요.
    families     : {alpha_name -> family_name} — 위 family 모듈과 함께 필요.

    stages: 각 모듈 적용 후 북의 sharpe/mdd/ann_return/vol(자산변동률) —
    한 번 실행으로 어느 모듈이 도움/손해인지 단계별로 보인다."""
    ctx = RiskContext(returns, quote_volume=quote_volume,
                      components=components, families=families,
                      panels=panels, funding_events=funding_events)

    def _row(name, w, enabled=True, notes=None):
        if not enabled:
            return {"stage": name, "enabled": False, "sharpe": None,
                    "mdd": None, "ann_return": None, "vol": None,
                    "notes": notes or {}}
        pnl = ctx.pnl(w)
        return {"stage": name, "enabled": True,
                "sharpe": _metrics.sharpe(pnl),
                "mdd": _metrics.max_drawdown(pnl),
                "ann_return": _metrics.ann_return(pnl),
                "vol": _metrics.realized_vol(pnl),
                "notes": notes or {}}

    stages = [_row("input", weights)]
    w = weights
    for item in pipeline_cfg:
        name = item["type"]
        if name not in RISK_REGISTRY:
            raise ValueError(f"unknown risk type {name!r} "
                             f"(available: {sorted(RISK_REGISTRY)})")
        if not item.get("enabled", True):
            stages.append(_row(name, w, enabled=False))
            continue
        w = RISK_REGISTRY[name](w, ctx, **(item.get("params") or {}))
        stages.append(_row(name, w, notes=ctx.pop_notes()))

    return {"weights": w, "pnl": ctx.pnl(w), "stages": stages}
