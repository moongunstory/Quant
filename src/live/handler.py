"""handler — 라이브 사이클의 실제 엔트리포인트 (매일 스케줄되는 부분).

run_cycle: (선택)데이터 최신화 -> 목표 가중치 계산 -> 주문 생성/실행 -> 원장 기록.
로컬/클라우드 동일 경로. mode="paper"(기본) 또는 "real".

    python -m src.live.handler --config data/strategy/portfolio/config.json --mode paper

fail-safe: target_weights 가 all_alphas_stale 이면 orders 가 자동 SKIP(포지션 유지).
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from src.portfolio.spec import load_portfolio_spec
from src.live import target_weights as TW, orders as OR, ledger as LG
from src.live.state import load_live_state

log = logging.getLogger("quant.live.handler")


def run_cycle(config_path, mode=None, today=None, refresh=False,
              max_staleness_days=None, rebuild=False, collector_deadline=None):
    """한 사이클 실행. -> {target, orders} 요약 dict.

    collector_deadline: time.monotonic() 기준 절대 마감 시각(초). refresh=True일 때
    live_refresh(및 콜드스타트 시 그 아래 universe_maintenance/full_collector)로 전달됨.
    Lambda처럼 실행시간이 제한된 환경에서 콜드스타트 백필이 타임아웃으로 강제종료되는 걸
    막기 위한 안전장치. None(기본값)이면 무제한.
    """
    today = today or datetime.now(timezone.utc).date()

    # [동적 설정 반영] data/runtime/live/config.json 설정이 존재하면 이를 우선하여 적용함
    live_cfg = load_live_state()
    if not live_cfg.get("enabled", True):
        log.info("라이브 봇이 비활성화(disabled) 상태이므로 사이클을 스킵합니다.")
        try:
            from src.live import telegram_bot as TB
            TB.send_telegram_message("ℹ️ <b>라이브 봇이 비활성화(disabled) 상태이므로 사이클을 스킵합니다.</b>")
        except Exception:
            pass
        return {
            "target": {"date": today.isoformat(), "weights": {}, "held_alphas": [], "diagnostics": {"all_alphas_stale": True}},
            "orders": {"date": today.isoformat(), "mode": mode, "skipped": True, "skip_reason": "봇 비활성화 상태"}
        }

    # [동적 설정 반영] CLI에서 --mode를 명시한 경우 CLI 우선, 생략 시 config.json 따름.
    # config.json 기본값도 없으면 최종 fallback으로 "paper" 사용.
    if mode is None:
        mode = live_cfg.get("mode", "paper")
    cfg = load_portfolio_spec(config_path)

    if refresh:
        rebuild = True  # 데이터 최신화 시 패널 캐시 재빌드(rebuild)도 강제로 활성화합니다.
        try:
            from src.collector import live_refresh as LR
            LR.run(cfg, deadline=collector_deadline)
        except Exception as e:
            log.warning("live_refresh 실패(기존 캐시로 진행): %s", e)

    kw = {} if max_staleness_days is None else {"max_staleness_days": max_staleness_days}
    target = TW.compute_target_weights(cfg, today=today, rebuild=rebuild, **kw)
    LG.record("target", {"date": target["date"], "n_coins": len(target["weights"]),
                         "held_alphas": target["held_alphas"],
                         "all_alphas_stale": target["diagnostics"].get("all_alphas_stale"),
                         "stale_alphas": target["diagnostics"].get("stale_alphas")},
              today=today)

    # [가상 매매 상시 트래킹] 모드와 무관하게 로컬 시뮬레이션(Paper) PnL 을 트래킹합니다.
    # paper_current(주문 전 보유)는 generate_orders 가 positions.json 을 덮어쓰기 전에 캡처해야 한다.
    # 실제 손익 반영(mark_to_market)은 이번 사이클 회전율(order_record.drift)을 알아야
    # 매매비용을 정직하게 차감할 수 있으므로 generate_orders 이후로 미룬다.
    paper_current = OR.load_positions()
    day_returns = target.get("day_returns", {})

    order_record = OR.generate_orders(target, today=today, mode=mode,
                                      rebalance_band=getattr(cfg, "rebalance_band", 0.0))

    if day_returns and paper_current:
        try:
            from src.live import paper as PA
            # 리밸런싱이 실제로 일어났을 때만(SKIP 아님) 회전율 비용 차감. 밴드로 보류/스킵이면 비용 0.
            turnover = 0.0 if order_record.get("skipped") else float(order_record.get("drift", 0.0))
            # returns_rows: '완결된 봉'만 담긴 일자별 수익률(오늘 부분봉 제외) —
            # 페이퍼 곡선이 하루 수익을 통째로 평가하고, 스케줄 누락일도 따라잡는다.
            day_pnl, equity = PA.mark_to_market(paper_current, day_returns, today=today,
                                                turnover=turnover,
                                                returns_rows=target.get("day_returns_rows"))
            log.info("[Paper PnL] 일일 가상 손익: %+.6f | 누적 가상 자산: %.6f (회전율=%.4f)",
                     day_pnl, equity, turnover)
        except Exception as e:
            log.warning("로컬 가상 매매 PnL 업데이트 실패: %s", e)
    LG.record("orders", {"date": order_record["date"], "mode": mode,
                         "skipped": order_record.get("skipped", False),
                         "n_orders": order_record.get("n_orders", 0),
                         "skip_reason": order_record.get("skip_reason")},
              today=today)

    # mode == "real" 일 때도 로컬 가상 포지션을 target 가중치로 갱신하여 다음 사이클의 PnL 계산에 반영되게 합니다.
    # mode == "paper" 일 때는 OR.generate_orders 내부에서 자동으로 save_positions를 수행하므로 생략합니다.
    # 단, 실전송에 성공한 주문이 하나도 없으면(프리플라이트 실패 등) 갱신하지 않는다 —
    # 실제로는 아무것도 안 움직였는데 그림자 장부만 '다 체결됨'이 되어 PnL 추적이 어긋나는 것 방지.
    # (주문이 아예 0건 = 이미 목표와 일치 상태라면 갱신해도 무방하므로 갱신.)
    if mode == "real" and not order_record.get("skipped"):
        sent_orders = order_record.get("orders", [])
        any_sent = any(o.get("exchange_result") for o in sent_orders)
        if any_sent or not sent_orders:
            OR.save_positions(target.get("weights", {}))
        else:
            log.warning("실전송 성공 주문 0건 -- 그림자 포지션(positions.json) 갱신 보류")

    # [실시간 손익용 진입가 스냅샷] 이번 사이클에 실제로 리밸런싱이 일어났으면(주문 SKIP 이 아니면)
    # 그 시점의 마크가격을 코인별로 저장한다. 텔레그램 /잔고 가 나중에 '현재가 vs 진입가'로
    # 실시간 가상 손익을 계산하는 근거다. 마크가격 조회 실패해도 사이클은 계속(fail-open).
    if not order_record.get("skipped"):
        try:
            from src.live import live_pnl as LP
            LP.snapshot_entry_prices(target.get("weights", {}), mode=mode, today=today)
        except Exception as e:
            log.warning("진입가 스냅샷 실패(사이클은 계속): %s", e)

    # [플라이트 레코더] 이 사이클의 상세 스냅샷을 telemetry-<date>.json 으로 남긴다.
    # prev_positions = paper_current(주문 전 보유) — 당일 day_returns 를 실제로 번 포지션.
    # 실패해도 사이클 자체는 계속(감사용 보조 기록이므로 fail-open).
    try:
        LG.record_telemetry(target, order_record, prev_positions=paper_current,
                            today=today, mode=mode)
    except Exception as e:
        log.warning("텔레메트리 스냅샷 기록 실패(사이클은 계속): %s", e)

    res = {"target": target, "orders": order_record}

    # [보존기간] 90일 지난 날짜별 기록(이벤트로그/주문기록/텔레메트리/번들zip)을 정리해
    # 로그가 무한히 쌓이지 않게 한다. 실패해도 사이클은 계속(fail-open).
    try:
        n = LG.prune_old(days=90, today=today)
        if n:
            log.info("오래된 기록 %d개 정리(90일 초과)", n)
    except Exception as e:
        log.warning("오래된 기록 정리 실패(사이클은 계속): %s", e)

    # [텔레그램 보고서 자동 발송] 사이클 결과를 유저에게 전달합니다.
    try:
        from src.live import telegram_bot as TB
        TB.send_cycle_report(res)
    except Exception as e:
        log.warning("텔레그램 사이클 실행 보고 전송 실패: %s", e)

    return res


def _print_summary(res):
    t, o = res["target"], res["orders"]
    print(f"\n=== 라이브 사이클 {t['date']} (mode={o['mode']}) ===")
    if t["diagnostics"].get("all_alphas_stale"):
        print("  [경고] 모든 알파 STALE -> 목표 미생성, 포지션 유지(SKIP)")
        print("  stale:", t["diagnostics"].get("stale_alphas"))
        return
    if t["diagnostics"].get("stale_alphas"):
        print("  일부 stale(제외):", t["diagnostics"]["stale_alphas"])
    print(f"  보유 알파: {t['held_alphas']}")
    print(f"  목표 코인수: {len(t['weights'])}")
    if o.get("skipped"):
        print(f"  주문 SKIP: {o.get('skip_reason')}")
    else:
        print(f"  주문 {o['n_orders']}건 (큰 이동 5개):")
        for od in o["orders"][:5]:
            print(f"    {od['side']:4} {od['coin']:12} "
                  f"{od['current_weight']:+.4f} -> {od['target_weight']:+.4f} "
                  f"(Δ{od['delta']:+.4f})")
