"""pipeline — 승인 포트폴리오 config 를 실제 백테스트로.

흐름:
  1. config 의 알파들을 각각 백테스트(engine.run) → 알파별 (신호 가중치, 순손익)
  2. combine.synthesize → 결합 가중치 패널(신호, 일별 L1=1) + 알파별 비중
  3. risk.run_risk_pipeline → 리스크 오버레이 적용 + stage별 성과 기록
  4. engine.result_from_weights(최종가중치) → 비용/펀딩 반영 최종 순손익
  5. metrics.compute + risk report

stage 리포트(ctx.pnl, gross 기준)는 '어느 리스크 모듈이 도움/손해'인지 상대비교용,
최종 metrics 는 거래비용+8h펀딩까지 반영한 실제 포트폴리오 성과다.

coin research/pipeline.py::build_portfolio 이식(Phase 1).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.backtest import engine, metrics as M, panel as P, directional as D
from src.backtest.evaluate import required_fields, spec_required_fields, evaluate
from src.backtest.spec import load_all
from src.config.backtest_settings import SETTINGS
from src.portfolio import combine as C
from src.portfolio.config import PortfolioConfig
from src.risk import risk as RK, report as RPT

log = logging.getLogger(__name__)

# 필터 스위치. True(기본/프로덕션): config 의 selection(상관·recency 게이트)을 적용
# — 실력 떨어진 알파는 벤치, 회복하면 복귀, 중복 알파는 컷. False 면 게이트를 건너뛰고
# config.alphas 를 '전부' 결합(정예 라인업을 그대로 보는 실험용). 실전은 반드시 True.
SELECTION_ENABLED = True


def _backtestable_specs(specs):
    """required_fields 가 전부 패널 시스템(FIELD_SPECS)에 있는 알파만 남긴다.

    아직 수집 안 된 데이터(bookDepth 등)나 미정의 필드를 쓰는 알파는 백테스트가
    불가능하므로 자동선택 풀에서 제외한다 — 필드가 생기면 자동으로 다시 합류.
    포트폴리오/워크포워드가 필드 union 을 한꺼번에 로드하다 하나 때문에 전체가
    죽는 것을 막는다(backtest-rank 의 per-alpha 스킵과 동일한 취지)."""
    available = set(P.FIELD_SPECS)
    ok, skipped = [], []
    for s in specs:
        missing = spec_required_fields(s.expression, s.neutralization) - available
        (skipped if missing else ok).append((s, missing))
    for s, missing in skipped:
        log.warning("알파 스킵 %s — 미가용 필드 %s (데이터 미수집/미정의)",
                    s.name, sorted(missing))
    return [s for s, _ in ok]


def _load_families(path=None):
    """{alpha_name -> family} 를 data/alpha_families.json({family:[names]})에서 로드.
    파일 없으면 None → family 캡/모듈은 조용히 비활성(coin _load_families 이식)."""
    p = Path(path) if path else SETTINGS.data_dir / "alpha_families.json"
    if not p.exists():
        return None
    fam = json.loads(p.read_text(encoding="utf-8"))
    return {name: family for family, names in fam.items() for name in names}


def _select_specs(cfg: PortfolioConfig, alphas_dir="data/alphas"):
    specs = load_all(alphas_dir)
    if cfg.alphas:
        # 명시 라인업: 이름 못 찾으면 에러(오타 방지). 필드 미가용은 엔진이 명확히 raise.
        by_name = {s.name: s for s in specs}
        missing = [n for n in cfg.alphas if n not in by_name]
        if missing:
            raise ValueError(f"config 알파를 data/alphas 에서 못 찾음: {missing}")
        specs = [by_name[n] for n in cfg.alphas]
    else:
        # 자동선택 풀: 백테스트 불가 알파(필드 미가용)는 조용히 제외.
        specs = _backtestable_specs(specs)
    if not specs:
        raise ValueError("포트폴리오에 알파가 없음")
    return specs


def build_portfolio(cfg: PortfolioConfig, rebuild=False, alphas_dir="data/alphas"):
    """PortfolioConfig -> {weights, net_pnl, metrics, stages, report, alpha_weights}.

    Phase 3: 알파마다 bar(판단주기)가 다를 수 있다. 각 알파를 자기 bar 그리드에서
    계산하고, delay 를 그 bar 단위로 적용한 뒤(pos_bar = weights.shift(delay)), 결과를
    '마스터 그리드'(포트폴리오 알파들 중 가장 촘촘한 bar)로 ffill 해 올린다('다음
    리밸런싱 전까지 보유'). 그 위에서 결합/리스크/최종손익을 계산 -> 주기가 달라도 한
    포트폴리오로 합쳐진다. 모든 알파가 1d 면 마스터=1d 로 기존 동작과 사실상 동일.

    per-alpha 손익(inverse_vol 가중용)도 마스터 그리드에서 계산해 서로 다른 주기의
    변동성이 같은 척도로 비교되게 한다. 포지션이 이미 delay 반영이므로 최종
    result_from_weights 는 delay=0(추가 지연 없음, 미래참조 없음: pos 는 이미 과거 정보)."""
    from src.backtest.timegrid import finest_bar, to_master
    specs = _select_specs(cfg, alphas_dir=alphas_dir)

    families = _load_families()

    # ---- directional 규칙(②): 자격 패밀리 + signed 신호면 neutralization 을 partial 로
    # 자동 승격 → 롱숏 비율을 신호가 정하게 함. 순노출은 market_neutrality 밴드가 통제.
    policy = D.load_policy()
    directional_on = D.is_enabled(policy, cfg_override=cfg.directional)
    if directional_on and not D.has_market_neutrality(cfg.risk_pipeline):
        raise ValueError(
            "directional 규칙이 켜져 있는데 risk_pipeline 에 활성화된 market_neutrality "
            "단계가 없음 — 순노출 밴드(net_exposure_limit)가 유일한 통제기이므로 반드시 "
            "필요. portfolio config 에 market_neutrality 를 추가하거나 directional 을 끄라."
        )

    master_bar = finest_bar([s.bar for s in specs])
    all_fields = set()
    for s in specs:
        all_fields |= spec_required_fields(s.expression, s.neutralization)
    master_panels = P.load_panels_for_bar(sorted(all_fields), bar=master_bar, rebuild=rebuild)
    mclose = master_panels["close"]
    master_index = mclose.index
    funding_events = P.load_funding_events(rebuild=rebuild)

    # 마스터 그리드 유니버스 마스크(시점정확 top-100). to_master 의 무제한 ffill 이
    # 유니버스에서 이탈한 코인의 stale 가중치를 이후 날짜로 끌고 가 마스크를 무효화하므로
    # (523 union 전체로 번짐), to_master 직후 이 마스크를 다시 씌워 이탈 코인을 0 으로 눌러준다.
    # 월별 멤버십이라 유니버스 안에 남아있는 코인의 '다음 리밸런싱까지 보유'는 안 깨진다.
    master_universe = P.build_universe_mask(master_index, mclose.columns) > 0.5

    pos_panels = {}   # 알파별 '마스터 그리드 보유 포지션'(delay 반영)
    net_pnls = {}     # 알파별 마스터 그리드 순손익(가중 계산용)
    metrics_by_name = {}  # 알파별 메트릭(selection fitness 게이트용)
    directional_alphas = []  # partial 로 승격된 알파(리포트용)
    for s in specs:
        fields = spec_required_fields(s.expression, s.neutralization)
        pnl_panels = P.load_panels_for_bar(sorted(fields), bar=s.bar, rebuild=rebuild)
        pclose = pnl_panels["close"]
        uni = P.build_universe_mask(pclose.index, pclose.columns)

        if directional_on:
            raw = evaluate(s.expression, pnl_panels)   # neutralization 전 원신호
            signed = D.is_signed(raw)
            s_res = D.resolve_spec(s, (families or {}).get(s.name), policy, signed)
            if s_res.neutralization == "partial" and s.neutralization != "partial":
                directional_alphas.append(s.name)
            s = s_res

        w_bar = engine.compute_weights(s, pnl_panels, universe=uni)   # 신호(지연 전, bar 그리드)
        pos_bar = w_bar.shift(s.delay)                               # delay = bar 단위
        pos_m = to_master(pos_bar, master_index).reindex(columns=mclose.columns)
        pos_m = pos_m.where(master_universe.reindex_like(pos_m))   # ffill 누수 차단: 이탈 코인 제거
        pos_panels[s.name] = pos_m
        res_a = engine.result_from_weights(pos_m, master_panels, delay=0,
                                           funding_events=funding_events)
        net_pnls[s.name] = res_a.net_pnl
        metrics_by_name[s.name] = M.compute(res_a)

    # ---- selection (coin 이식): config 에 selection 이 있으면 상관/패밀리 dedup 적용.
    # 없으면 종전대로 specs 전부 사용(하위호환). fitness 내림차순 랭킹을 후보로.
    sel_cfg = cfg.selection
    if not SELECTION_ENABLED and sel_cfg and sel_cfg.get("method"):
        log.info("selection: [임시] 필터 비활성(SELECTION_ENABLED=False) — "
                 "config 알파 %d개 전부 결합 %s", len(pos_panels), list(pos_panels))
        sel_cfg = None
    if sel_cfg and sel_cfg.get("method"):
        corr = C.correlation_matrix(net_pnls)
        fitness_by_name = {n: m["fitness"] for n, m in metrics_by_name.items()}
        ranked = sorted(fitness_by_name, key=lambda n: fitness_by_name[n], reverse=True)
        selected = C.select_alphas(
            ranked, corr, method=sel_cfg["method"],
            params=sel_cfg.get("params", {}),
            fitness=fitness_by_name, families=families,
            metrics=metrics_by_name, pnl_by_alpha=net_pnls,
        )
        if not selected:
            raise ValueError("selection 결과가 비어있음 — 게이트/임계값을 완화하라")
        pos_panels = {n: pos_panels[n] for n in selected}
        net_pnls = {n: net_pnls[n] for n in selected}

    method = cfg.weighting_method
    combined, alpha_w, contributions = C.synthesize(
        pos_panels, method=method,
        pnl_by_alpha=(net_pnls if method in ("inverse_vol", "skill") else None),
        params=cfg.weighting_params,
        return_contributions=True,
    )

    returns = mclose / mclose.shift(1) - 1.0

    quote_volume = master_panels.get("quote_volume")
    # components/families (D14): 결합북 내 '알파별 기여분'(Σ=결합북, 스케일 정합) +
    # 패밀리 맵을 넘겨, family_corr_scale/family_gross_cap 이 선택 이후 중간에 생기는
    # 상관·집중을 잡게 한다. 원본 L1=1 패널이 아니라 기여분을 넘겨야 gross 가 안 튄다.
    # family 모듈이 없으면 무해하게 무시됨.
    risk_out = RK.run_risk_pipeline(combined, returns, cfg.risk_pipeline,
                                    quote_volume=quote_volume,
                                    components=contributions, families=families,
                                    panels=master_panels,
                                    funding_events=funding_events)
    final_weights = risk_out["weights"]

    # 포지션이 이미 delay 반영 -> 최종은 delay=0. 비용+8h펀딩 반영 순손익.
    final = engine.result_from_weights(final_weights, master_panels, delay=0,
                                       funding_events=funding_events)
    m = M.compute(final)
    report = RPT.report_from_pipeline_stages(risk_out["stages"])

    return {
        "config": cfg,
        "master_bar": master_bar,
        "weights": final_weights,
        "result": final,
        "net_pnl": final.net_pnl,
        "metrics": m,
        "alpha_weights": alpha_w,
        "directional_on": directional_on,
        "directional_alphas": [n for n in directional_alphas if n in pos_panels],
        "stages": risk_out["stages"],
        "report": report,
    }
