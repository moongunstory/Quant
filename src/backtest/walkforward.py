"""walkforward — '선택 과정 자체'를 미래참조 0으로 검증(과최적화 방어).

동기(coin 이식, PLAN 2026-07-11 ③-2): 전 표본 fitness 랭킹은 selector 에게
'나중에 평가받을 바로 그 미래'를 보여주기 때문에, 2020~21 에만 수익이 몰린
funding 북도 전 기간 백테스트에선 멀쩡해 보인다. 이 도구는 선택을 '시점마다'
다시 실행한다:

  각 리밸런싱 시점 t 에서:
    1. t 까지의 데이터만으로 모든 알파를 점수화(expanding, 또는 --window trailing)
    2. 품질 게이트(min_fitness) -> greedy low-correlation dedup(+패밀리 캡)
    3. 생존 알파를 inverse-vol 로 가중(같은 과거만 사용)
    4. 그 북을 다음 rebalance 일 동안 그대로 보유하며 pnl 기록

  보유 구간들을 이어붙이면 미래를 한 번도 안 쓴 OOS pnl 시리즈가 된다. 전 표본
북은 멀쩡한데 OOS 가 무너지면, 그건 알파가 아니라 '선택 시점 미래참조'였다.

이건 검증 도구지 프로덕션 파이프라인이 아니다. 조합은 pnl-space(알파별 net pnl
가중합, 가중치 합=1)로 하며 — 프로덕션 패널 블렌드의 빠르고 가까운 근사다. 알파별
pnl 은 이미 비용 차감됨. 아무것도 게이트를 통과 못 한 구간은 FLAT(0) 로 보유 —
그 자체가 발견이다.

coin research/walkforward.py 이식. 단, coin 의 SQLite/alpha_db·recency explain 은
빼고, Quant 의 engine.run(알파별)에서 net_pnl+turnover 를 곧바로 만들어 쓴다.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.backtest import engine, metrics, panel as P
from src.backtest.evaluate import required_fields, spec_required_fields
from src.backtest.spec import load_all
from src.config.backtest_settings import SETTINGS
from src.portfolio import combine


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def collect_alpha_series(alphas_dir=None, rebuild=False):
    """alphas/*.json 을 각각 한 번씩 돌려 {name: {"pnl": Series, "turnover": Series}}.

    엔진이 causal(delay>=1)이므로, 이 전 기간 시리즈를 t 에서 잘라 써도 t 시점에
    가용한 정보만 쓴다."""
    alphas_dir = alphas_dir or SETTINGS.alphas_dir
    specs = load_all(alphas_dir)
    if not specs:
        raise FileNotFoundError(f"{alphas_dir} 에 알파가 없음")
    # 백테스트 불가 알파(미가용 필드: 미수집 bookDepth·미정의 basis 등)는 풀에서 제외.
    available = set(P.FIELD_SPECS)
    _kept = []
    for spec in specs:
        missing = spec_required_fields(spec.expression, spec.neutralization) - available
        if missing:
            logging.getLogger(__name__).warning(
                "알파 스킵 %s — 미가용 필드 %s", spec.name, sorted(missing))
        else:
            _kept.append(spec)
    specs = _kept
    funding_events = P.load_funding_events(rebuild=rebuild)
    out = {}
    for spec in specs:
        fields = spec_required_fields(spec.expression, spec.neutralization)
        panels = P.load_panels_for_bar(fields, bar=spec.bar, rebuild=rebuild)
        close = panels["close"]
        universe = P.build_universe_mask(close.index, close.columns)
        res = engine.run(spec, panels, universe=universe,
                         funding_events=funding_events)
        out[spec.name] = {"pnl": res.net_pnl, "turnover": res.turnover}
    return out


# --------------------------------------------------------------------------- #
# point-in-time scoring + selection
# --------------------------------------------------------------------------- #

def point_in_time_selection(series, asof, window=None, min_history=252,
                            min_fitness=0.0, max_corr=0.5, top_n=None,
                            families=None, max_per_family=None,
                            method="low_correlation", method_params=None,
                            min_recent_sharpe=None, recent_key="sharpe_hl"):
    """asof 까지의 데이터만으로 게이트 + selection dedup + inverse-vol 가중.

    method/method_params 는 combine.select_alphas(프로덕션과 동일 registry)로 라우팅.
    min_history 일 미만 관측 알파는 아직 점수화 불가.
    min_recent_sharpe: 주면 최근성(쇠퇴) 게이트 활성 — snap[recent_key](기본
      sharpe_hl=반감기가중 샤프, causal 로 asof 까지만 계산)가 이 값 미만인 알파 배제.
      improved config 의 selection.params.min_recent_sharpe 를 OOS 에서도 동일 적용.
    반환: (selected, {name: weight}, {name: fitness})."""
    fit, hist, snap = {}, {}, {}
    for name, d in series.items():
        p = d["pnl"].loc[:asof].dropna()
        if window is not None:
            p = p.iloc[-int(window):]
        if len(p) < min_history:
            continue
        to = d["turnover"].reindex(p.index)
        s = metrics.sharpe(p)
        ar = metrics.ann_return(p)
        fit[name] = metrics.fitness(s, ar, metrics.avg_turnover(to))
        hist[name] = p
        # sharpe_hl 도 asof 까지의 p 로만 계산 → recency 게이트도 미래참조 0.
        snap[name] = {"sharpe": s, "mdd": metrics.max_drawdown(p), "ann_return": ar,
                      "sharpe_hl": metrics.halflife_weighted_sharpe(p)}
    ranked = sorted(fit, key=fit.get, reverse=True)
    if min_fitness is not None:
        ranked = [n for n in ranked if fit[n] >= min_fitness]
    if not ranked:
        return [], {}, fit
    corr = pd.DataFrame({n: hist[n] for n in ranked}).corr()
    params = dict(method_params or {})
    params.setdefault("max_corr_threshold", max_corr)
    params.setdefault("top_n", top_n)
    params.setdefault("max_per_family", max_per_family)
    if min_recent_sharpe is not None:
        params.setdefault("min_recent_sharpe", min_recent_sharpe)
        params.setdefault("recent_key", recent_key)
    # 위에서 이미 min_fitness 게이트를 적용(ranked 사전필터)했으므로, 하류 registry
    # method 의 자체 기본값(예: _sel_low_correlation 의 min_fitness=0.0)이 다른
    # 임계값을 재적용하지 않도록 None 으로 끈다(coin 과 동일 규약).
    params.setdefault("min_fitness", None)
    selected = combine.select_alphas(ranked, corr, method=method, params=params,
                                     fitness=fit, families=families,
                                     metrics=snap,
                                     pnl_by_alpha={n: hist[n] for n in ranked})
    inv = {}
    for n in selected:
        sd = hist[n].std(ddof=1)
        inv[n] = (1.0 / sd) if sd and not np.isnan(sd) else 0.0
    total = sum(inv.values()) or 1.0
    return selected, {n: v / total for n, v in inv.items()}, fit


# --------------------------------------------------------------------------- #
# the walk
# --------------------------------------------------------------------------- #

def run_walkforward(series, rebalance=63, window=None, min_history=252,
                    min_fitness=0.0, max_corr=0.5, top_n=None,
                    families=None, max_per_family=None,
                    method="low_correlation", method_params=None):
    """선택을 히스토리 전체에 굴려 {"oos_pnl", "periods"} 반환."""
    idx = None
    for d in series.values():
        idx = d["pnl"].index if idx is None else idx.union(d["pnl"].index)
    idx = idx.sort_values()

    oos = []
    periods = []
    for i in range(int(min_history), len(idx), int(rebalance)):
        asof = idx[i - 1]                        # 결정은 어제까지의 데이터만
        period_idx = idx[i:i + int(rebalance)]
        if len(period_idx) == 0:
            break
        sel, w, fit = point_in_time_selection(
            series, asof, window=window, min_history=min_history,
            min_fitness=min_fitness, max_corr=max_corr, top_n=top_n,
            families=families, max_per_family=max_per_family,
            method=method, method_params=method_params)
        if sel:
            frame = pd.DataFrame({n: series[n]["pnl"].reindex(period_idx) for n in sel})
            pnl = (frame * pd.Series(w)).sum(axis=1, min_count=1).fillna(0.0)
        else:
            pnl = pd.Series(0.0, index=period_idx)   # 자격자 없음 -> flat
        oos.append(pnl)
        top_pool = sorted(fit, key=fit.get, reverse=True)[:5]
        periods.append({
            "asof": str(asof.date()),
            "start": str(period_idx[0].date()),
            "end": str(period_idx[-1].date()),
            "n_scoreable": len(fit),
            "selected": sel,
            "weights": {k: round(v, 4) for k, v in w.items()},
            "fitness_top5": {n: round(fit[n], 3) for n in top_pool},
        })
    return {"oos_pnl": pd.concat(oos) if oos else pd.Series(dtype=float),
            "periods": periods}


# --------------------------------------------------------------------------- #
# weight-space walkforward (리스크 스택까지 OOS 검증)
# --------------------------------------------------------------------------- #

def collect_alpha_books(alphas_dir=None, rebuild=False, families=None):
    """build_portfolio 와 동일한 방식으로 '백테스트가능 알파 전체'의 마스터그리드
    포지션 패널 + net_pnl/turnover 를 한 번에 생성(리스크 워크포워드 재료).

    반환 (series, pos_panels, master_panels, funding_events):
      series     : {name: {"pnl","turnover"}}  마스터그리드 net pnl(선택 점수화용).
      pos_panels : {name: date×coin 포지션}     delay 반영·마스터그리드·causal.
    directional 정책이 켜져 있으면 build_portfolio 와 동일하게 signed 알파를 partial
    로 승격(프로덕션 포트폴리오와 같은 북을 재현)."""
    from src.backtest.timegrid import finest_bar, to_master
    from src.backtest import directional as D
    from src.backtest.evaluate import evaluate

    alphas_dir = alphas_dir or SETTINGS.alphas_dir
    all_specs = load_all(alphas_dir)
    if not all_specs:
        raise FileNotFoundError(f"{alphas_dir} 에 알파가 없음")
    available = set(P.FIELD_SPECS)
    specs = []
    for s in all_specs:
        missing = spec_required_fields(s.expression, s.neutralization) - available
        if missing:
            logging.getLogger(__name__).warning(
                "알파 스킵 %s — 미가용 필드 %s", s.name, sorted(missing))
        else:
            specs.append(s)
    if not specs:
        raise ValueError("백테스트가능 알파가 없음")

    master_bar = finest_bar([s.bar for s in specs])
    all_fields = set()
    for s in specs:
        all_fields |= spec_required_fields(s.expression, s.neutralization)
    master_panels = P.load_panels_for_bar(sorted(all_fields), bar=master_bar,
                                          rebuild=rebuild)
    mclose = master_panels["close"]
    master_index = mclose.index
    funding_events = P.load_funding_events(rebuild=rebuild)

    policy = D.load_policy()
    directional_on = D.is_enabled(policy)

    # to_master ffill 이 유니버스 이탈 코인의 stale 가중치를 끌고 가 마스크를 무효화하므로
    # (build_portfolio 와 동일 버그), to_master 직후 마스터 그리드 마스크를 다시 씌운다.
    master_universe = P.build_universe_mask(master_index, mclose.columns) > 0.5

    pos_panels, series = {}, {}
    for s in specs:
        fields = spec_required_fields(s.expression, s.neutralization)
        pnl_panels = P.load_panels_for_bar(sorted(fields), bar=s.bar, rebuild=rebuild)
        pclose = pnl_panels["close"]
        uni = P.build_universe_mask(pclose.index, pclose.columns)
        if directional_on:
            raw = evaluate(s.expression, pnl_panels)
            signed = D.is_signed(raw)
            s = D.resolve_spec(s, (families or {}).get(s.name), policy, signed)
        w_bar = engine.compute_weights(s, pnl_panels, universe=uni)
        pos_bar = w_bar.shift(s.delay)                       # delay = bar 단위(causal)
        pos_m = to_master(pos_bar, master_index).reindex(columns=mclose.columns)
        pos_m = pos_m.where(master_universe.reindex_like(pos_m))   # ffill 누수 차단
        res_a = engine.result_from_weights(pos_m, master_panels, delay=0,
                                           funding_events=funding_events)
        pos_panels[s.name] = pos_m
        series[s.name] = {"pnl": res_a.net_pnl, "turnover": res_a.turnover}
    return series, pos_panels, master_panels, funding_events


def run_walkforward_portfolio(series, pos_panels, master_panels, funding_events,
                              cfg, families=None, rebalance=63, window=None,
                              min_history=252):
    """config(selection+weighting+risk_pipeline)를 시점마다 재적용하는 OOS 검증.

    run_walkforward 가 pnl-space 근사(알파 net pnl 가중합)로 '선택'만 검증하는
    반면, 이쪽은 프로덕션 build_portfolio 와 동일한 weight-space 회계로 '리스크
    스택까지' OOS 검증한다. 각 리밸런싱 시점 asof(어제까지):
      1. asof 까지로 causal 선택 + causal inverse-vol 가중(config params)
      2. 선택 알파의 마스터그리드 포지션을 [처음~폴드끝]까지 합쳐 결합북 + family
         모듈용 기여분(scale 정합)
      3. config.risk_pipeline 을 그 히스토리에 적용 — 리스크 모듈은 이미 causal
         (오늘 스케일은 어제까지 정보만) 이라 미래참조 0. warm-up 을 위해 폴드만이
         아니라 전체 히스토리에서 돌리고 폴드 구간 net_pnl 만 슬라이스.
      4. 폴드 net_pnl 을 이어붙임.

    주의: 원본 run_walkforward 와 마찬가지로 폴드 경계의 로스터 교체 '전환비용'은
    부과하지 않는다(각 폴드를 독립적으로 warm-up). 폴드 내부 거래비용/펀딩은 엔진이
    정확히 반영. 반환 {oos_pnl, periods}(summarize/save_report 호환)."""
    from src.portfolio import combine as C
    from src.portfolio import risk as RK

    sel_cfg = cfg.selection or {}
    sp = dict(sel_cfg.get("params", {}) or {})
    sel_method = sel_cfg.get("method", "low_correlation")
    max_corr = sp.get("max_corr_threshold", 0.5)
    top_n = sp.get("top_n")
    max_per_family = sp.get("max_per_family")
    min_fitness = sp.get("min_fitness", 0.0)
    min_recent_sharpe = sp.get("min_recent_sharpe")
    wmethod = cfg.weighting_method
    wparams = cfg.weighting_params
    risk_cfg = cfg.risk_pipeline

    mclose = master_panels["close"]
    returns = mclose / mclose.shift(1) - 1.0
    quote_volume = master_panels.get("quote_volume")

    idx = mclose.index.sort_values()
    oos, periods = [], []
    for i in range(int(min_history), len(idx), int(rebalance)):
        asof = idx[i - 1]                        # 결정은 어제까지의 데이터만
        period_idx = idx[i:i + int(rebalance)]
        if len(period_idx) == 0:
            break
        sel, _w_iv, fit = point_in_time_selection(
            series, asof, window=window, min_history=min_history,
            min_fitness=min_fitness, max_corr=max_corr, top_n=top_n,
            families=families, max_per_family=max_per_family,
            method=sel_method, min_recent_sharpe=min_recent_sharpe)
        top_pool = sorted(fit, key=fit.get, reverse=True)[:5]
        rec = {
            "asof": str(asof.date()),
            "start": str(period_idx[0].date()),
            "end": str(period_idx[-1].date()),
            "n_scoreable": len(fit),
            "selected": sel,
            "fitness_top5": {n: round(fit[n], 3) for n in top_pool},
        }
        if not sel:
            oos.append(pd.Series(0.0, index=period_idx))   # 자격자 없음 -> flat
            rec["weights"] = {}
            periods.append(rec)
            continue

        fold_end = period_idx[-1]
        # config 가중(causal): asof 까지의 net pnl 로만 inverse-vol → 미래참조 0.
        causal_pnls = {n: series[n]["pnl"].loc[:asof] for n in sel}
        # 결합북/기여분은 [처음~폴드끝] 히스토리(리스크 warm-up). 포지션 자체는 causal.
        sel_pos = {n: pos_panels[n].loc[:fold_end] for n in sel}
        combined, alpha_w, contributions = C.synthesize(
            sel_pos, method=wmethod, pnl_by_alpha=causal_pnls,
            params=wparams, return_contributions=True)

        panels_slc = {k: v.loc[:fold_end] for k, v in master_panels.items()}
        risk_out = RK.run_risk_pipeline(
            combined, returns.loc[:fold_end], risk_cfg,
            quote_volume=None if quote_volume is None else quote_volume.loc[:fold_end],
            components=contributions, families=families,
            panels=panels_slc, funding_events=funding_events)
        final = engine.result_from_weights(risk_out["weights"], panels_slc,
                                           delay=0, funding_events=funding_events)
        oos.append(final.net_pnl.reindex(period_idx).fillna(0.0))
        rec["weights"] = {k: round(float(v), 4) for k, v in alpha_w.items()}
        periods.append(rec)

    return {"oos_pnl": pd.concat(oos) if oos else pd.Series(dtype=float),
            "periods": periods}


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def yearly_sharpe(pnl):
    p = pd.Series(pnl).dropna()
    return {int(y): round(metrics.sharpe(g), 2) for y, g in p.groupby(p.index.year)}


def yearly_sharpe_table(series):
    """알파별×연도 Sharpe 표 — 새 배치마다 필수 리포트."""
    return pd.DataFrame({n: yearly_sharpe(d["pnl"]) for n, d in series.items()}).T.sort_index()


def rotation_events(periods):
    """구간별 선택 diff — 각 리밸런싱에서 무엇이 편입/탈락했는지(+그 순간 fitness)."""
    events = []
    prev = None
    for p in periods:
        cur = set(p["selected"])
        if prev is not None:
            added = sorted(cur - prev)
            dropped = sorted(prev - cur)
            if added or dropped:
                events.append({
                    "asof": p["asof"], "start": p["start"], "end": p["end"],
                    "added": added, "dropped": dropped,
                    "fitness_at_change": {n: p["fitness_top5"].get(n)
                                          for n in (added + dropped)
                                          if n in p["fitness_top5"]},
                })
        prev = cur
    return events


def summarize(wf):
    oos = wf["oos_pnl"]
    flat_days = int((oos == 0.0).sum())
    lines = []
    lines.append(f"periods: {len(wf['periods'])}   OOS days: {len(oos)}   "
                 f"flat(자격자 없음): {flat_days} "
                 f"({flat_days / max(len(oos), 1) * 100:.0f}%)")
    lines.append(f"OOS  sharpe={metrics.sharpe(oos):+.2f}  "
                 f"mdd={metrics.max_drawdown(oos):.3f}  "
                 f"ann_return={metrics.ann_return(oos):+.3f}")
    lines.append(f"OOS yearly sharpe: {yearly_sharpe(oos)}")
    sel_counts = {}
    for p in wf["periods"]:
        for n in p["selected"]:
            sel_counts[n] = sel_counts.get(n, 0) + 1
    lines.append(f"selection frequency: "
                 f"{dict(sorted(sel_counts.items(), key=lambda kv: -kv[1]))}")
    events = rotation_events(wf["periods"])
    lines.append(f"\n=== rotation log: {len(events)} 개 구간에서 로스터 변경 "
                 f"(총 {max(len(wf['periods']) - 1, 0)} 전이 중) ===")
    for e in events:
        lines.append(f"  {e['asof']}: +{e['added']} -{e['dropped']} "
                     f"{e['fitness_at_change']}")
    return "\n".join(lines)


def save_report(wf, series, tag="walkforward"):
    """OOS 요약 JSON + 알파별×연도 Sharpe CSV 저장."""
    out_dir = SETTINGS.data_dir
    jp = out_dir / f"{tag}_report.json"
    jp.write_text(json.dumps({
        "oos_sharpe": metrics.sharpe(wf["oos_pnl"]),
        "oos_mdd": metrics.max_drawdown(wf["oos_pnl"]),
        "oos_yearly_sharpe": yearly_sharpe(wf["oos_pnl"]),
        "periods": wf["periods"],
        "rotation_events": rotation_events(wf["periods"]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    cp = out_dir / f"{tag}_yearly_sharpe.csv"
    yearly_sharpe_table(series).to_csv(cp)
    return jp, cp
