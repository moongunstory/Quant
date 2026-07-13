"""orders — 목표 가중치 vs 현재 포지션 -> 주문 리스트 (coin live/orders.py 이식).

target_weights.compute_target_weights() 가 오늘의 목표 가중치를 내면, 이 모듈이
현재 포지션과 diff 해서 주문 리스트를 만든다. 두 모드:

  paper : 실거래 연결 없음. '현재 포지션' = 지난 주문이 옮긴 대상, data/runtime/live/
          positions.json 에 {coin: weight} 로 저장. 즉시/완전 체결 가정(수수료·
          슬리피지는 이미 모든 백테스트 숫자에 반영돼 있으므로 페이퍼에선 생략).
  real  : 현재 포지션을 실제 거래소에서 읽고(trade/exchange.get_open_positions),
          목표 가중치를 mark price + book_aum_usd 로 목표 수량으로 환산, LOT 반올림
          후 시장가 전송. 같은 날 중복 전송을 막는 daily guard 포함.

FAIL-SAFE: target diagnostics.all_alphas_stale 이면 '청산 주문'을 만들지 않고
명시적으로 SKIP (오늘 목표를 신뢰할 수 없으므로 현재 포지션을 그대로 둔다).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from src.config.backtest_settings import SETTINGS

POSITIONS_PATH_DEFAULT = SETTINGS.data_dir / "runtime" / "live" / "positions.json"
DUST_THRESHOLD_DEFAULT = 0.001   # 이보다 작은 가중치 변화는 무시(부동소수/서브-lot 잡음)


def load_positions(path=None):
    """{coin: weight} 현재 보유(페이퍼 모델). 파일 없으면 {} (첫 실행 = 전부 현금)."""
    p = Path(path or POSITIONS_PATH_DEFAULT)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_positions(weights, path=None):
    p = Path(path or POSITIONS_PATH_DEFAULT)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(weights, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_orders(target_weights, current_weights, dust_threshold=DUST_THRESHOLD_DEFAULT):
    """target/current: {coin: weight}. -> [{coin,current_weight,target_weight,delta,side}]
    (|delta|>dust 인 코인만, |delta| 내림차순 = 큰 이동 먼저)."""
    coins = set(target_weights) | set(current_weights)
    orders = []
    for c in sorted(coins):
        cur = float(current_weights.get(c, 0.0))
        tgt = float(target_weights.get(c, 0.0))
        delta = tgt - cur
        if abs(delta) < dust_threshold:
            continue
        orders.append({"coin": c, "current_weight": cur, "target_weight": tgt,
                       "delta": delta, "side": "buy" if delta > 0 else "sell"})
    orders.sort(key=lambda o: -abs(o["delta"]))
    return orders


def _real_current_weights():
    """실제 거래소 포지션 -> {coin: weight} (수량*마크가격 / book_aum_usd)."""
    from src.live.exchange import exchange
    positions = exchange.get_open_positions()
    if not positions:
        return {}
    prices = exchange.get_mark_prices()
    out = {}
    for symbol, qty in positions.items():
        price = prices.get(symbol)
        if price:
            out[symbol] = (qty * price) / SETTINGS.book_aum_usd
    return out


def _already_sent_today(today) -> bool:
    """daily guard: 오늘자 orders 기록이 real & not skipped 면 이미 전송된 것."""
    p = Path(SETTINGS.data_dir) / "runtime" / "live" / f"orders_{today.isoformat()}.json"
    if not p.exists():
        return False
    try:
        record = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    return record.get("mode") == "real" and not record.get("skipped")


def _send_real_orders(orders):
    """각 주문 dict 에 exchange_result 또는 error 를 채운다."""
    from src.live.exchange import exchange
    prices = exchange.get_mark_prices()
    for o in orders:
        symbol = o["coin"]
        price = prices.get(symbol)
        if not price:
            o["error"] = f"{symbol} mark price 없음 -- 미전송"
            continue
        raw_qty = (o["delta"] * SETTINGS.book_aum_usd) / price
        try:
            exchange.set_leverage(symbol)
            qty = exchange.round_quantity(symbol, raw_qty)
            if qty == 0.0:
                o["error"] = "반올림 수량 0 (minQty 미만) -- 미전송"
                continue
            side = "BUY" if raw_qty > 0 else "SELL"
            result = exchange.place_market_order(symbol, side, abs(qty))
            o["exchange_result"] = {"status": getattr(result, "status", None),
                                    "sent_quantity": abs(qty)}
        except Exception as exc:
            o["error"] = str(exc)


def portfolio_drift(target_weights, current_weights):
    """현재 포트폴리오와 목표의 총 L1 드리프트 = Σ_coin |target - current|.
    0 = 동일, 2 = 완전 반대(롱숏 전부 뒤집힘)."""
    coins = set(target_weights) | set(current_weights)
    return float(sum(abs(float(target_weights.get(c, 0.0)) - float(current_weights.get(c, 0.0)))
                     for c in coins))


def generate_orders(target_record, positions_path=None, today=None,
                    dust_threshold=DUST_THRESHOLD_DEFAULT, mode="paper",
                    rebalance_band=0.0):
    """target_record(compute_target_weights 결과)를 현재 포지션과 diff, 주문 리스트
    기록을 data/runtime/live/orders_<date>.json 에 저장(페이퍼는 positions.json 도 갱신).
    real 모드는 주문도 전송. 주문 기록 dict 반환.

    rebalance_band (Phase 3-B): 현재 포트폴리오 대비 총 드리프트(Σ|Δ|)가 이 값보다
    작으면 '아직 충분히 달라지지 않음'으로 보고 리밸런싱을 건너뛴다(현재 포지션 유지,
    회전율/비용 절약). 0 = 매 사이클 리밸런싱(밴드 없음)."""
    today = today or datetime.now(timezone.utc).date()
    diagnostics = target_record.get("diagnostics", {}) or {}

    if diagnostics.get("all_alphas_stale"):
        record = {"date": today.isoformat(), "mode": mode, "skipped": True,
                  "skip_reason": "all_alphas_stale -- 오늘 목표 신뢰 불가, 주문 미생성/포지션 유지",
                  "orders": [], "n_orders": 0}
    elif mode == "real" and _already_sent_today(today):
        record = {"date": today.isoformat(), "mode": mode, "skipped": True,
                  "skip_reason": "실매매 주문 오늘 이미 전송됨(daily guard)",
                  "orders": [], "n_orders": 0}
    else:
        current = (_real_current_weights() if mode == "real"
                   else load_positions(positions_path))
        target = target_record.get("weights", {})
        drift = portfolio_drift(target, current)
        if rebalance_band > 0.0 and drift < rebalance_band:
            # 드리프트가 밴드 미만 -> 리밸런싱 보류, 현재 포지션 그대로.
            record = {"date": today.isoformat(), "mode": mode, "skipped": True,
                      "skip_reason": (f"드리프트 {drift:.4f} < rebalance_band {rebalance_band:.4f} "
                                      "-- 아직 충분히 다르지 않음, 리밸런싱 보류(포지션 유지)"),
                      "orders": [], "n_orders": 0, "drift": drift,
                      "rebalance_band": rebalance_band}
            out_path = Path(SETTINGS.data_dir) / "runtime" / "live" / f"orders_{today.isoformat()}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            return record
        orders = compute_orders(target, current, dust_threshold=dust_threshold)
        if mode == "real" and orders:
            _send_real_orders(orders)
        record = {"date": today.isoformat(), "mode": mode, "skipped": False,
                  "orders": orders, "n_orders": len(orders), "drift": drift,
                  "rebalance_band": rebalance_band,
                  "target_weights": target, "previous_weights": current}
        if mode == "paper":
            save_positions(target, positions_path)  # 즉시·완전 체결 가정

    out_path = Path(SETTINGS.data_dir) / "runtime" / "live" / f"orders_{today.isoformat()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record
