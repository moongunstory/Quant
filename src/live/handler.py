"""handler — 라이브 사이클의 실제 엔트리포인트 (매일 스케줄되는 부분).

run_cycle: (선택)데이터 최신화 -> 목표 가중치 계산 -> 주문 생성/실행 -> 원장 기록.
로컬/클라우드 동일 경로. mode="paper"(기본) 또는 "real".

    python -m src.live.handler --config data/portfolio.json --mode paper

fail-safe: target_weights 가 all_alphas_stale 이면 orders 가 자동 SKIP(포지션 유지).
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from src.portfolio.config import load_portfolio_config
from src.live import target_weights as TW, orders as OR, ledger as LG

log = logging.getLogger("quant.live.handler")


def run_cycle(config_path, mode="paper", today=None, refresh=False,
              max_staleness_days=None, rebuild=False):
    """한 사이클 실행. -> {target, orders} 요약 dict."""
    today = today or datetime.now(timezone.utc).date()
    cfg = load_portfolio_config(config_path)

    if refresh:
        try:
            from src.collector import live_refresh as LR
            LR.run(cfg)
        except Exception as e:
            log.warning("live_refresh 실패(기존 캐시로 진행): %s", e)

    kw = {} if max_staleness_days is None else {"max_staleness_days": max_staleness_days}
    target = TW.compute_target_weights(cfg, today=today, rebuild=rebuild, **kw)
    LG.record("target", {"date": target["date"], "n_coins": len(target["weights"]),
                         "held_alphas": target["held_alphas"],
                         "all_alphas_stale": target["diagnostics"].get("all_alphas_stale"),
                         "stale_alphas": target["diagnostics"].get("stale_alphas")},
              today=today)

    order_record = OR.generate_orders(target, today=today, mode=mode,
                                      rebalance_band=getattr(cfg, "rebalance_band", 0.0))
    LG.record("orders", {"date": order_record["date"], "mode": mode,
                         "skipped": order_record.get("skipped", False),
                         "n_orders": order_record.get("n_orders", 0),
                         "skip_reason": order_record.get("skip_reason")},
              today=today)

    return {"target": target, "orders": order_record}


def _print_summary(res):
    t, o = res["target"], res["orders"]
    print(f"\n=== 라이브 사이클 {t['date']} (mode={o['mode']}) ===")
    if t["diagnostics"].get("all_alphas_stale"):
        print("  ⚠ 모든 알파 STALE -> 목표 미생성, 포지션 유지(SKIP)")
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


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Quant 라이브 사이클")
    ap.add_argument("--config", default="data/portfolio.json")
    ap.add_argument("--mode", choices=["paper", "real"], default="paper")
    ap.add_argument("--refresh", action="store_true", help="사이클 전 데이터 최신화")
    ap.add_argument("--rebuild", action="store_true", help="패널 캐시 재빌드")
    ap.add_argument("--max-staleness-days", type=int, default=None)
    args = ap.parse_args()
    res = run_cycle(args.config, mode=args.mode, refresh=args.refresh,
                    rebuild=args.rebuild, max_staleness_days=args.max_staleness_days)
    _print_summary(res)


if __name__ == "__main__":
    main()
