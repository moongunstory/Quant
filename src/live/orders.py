"""orders — 목표 가중치 vs 현재 포지션 -> 주문 리스트 (coin live/orders.py 이식).

target_weights.compute_target_weights() 가 오늘의 목표 가중치를 내면, 이 모듈이
현재 포지션과 diff 해서 주문 리스트를 만든다. 두 모드:

  paper : 실거래 연결 없음. '현재 포지션' = 지난 주문이 옮긴 대상, data/runtime/live/
          positions.json 에 {coin: weight} 로 저장. 즉시/완전 체결 가정(수수료·
          슬리피지는 이미 모든 백테스트 숫자에 반영돼 있으므로 페이퍼에선 생략).
  real  : 현재 포지션을 실제 거래소에서 읽고(trade/exchange.get_open_positions),
          목표 가중치를 mark price + 그 사이클에 조회한 실제 계좌 총자산(aum_usd,
          exchange.get_account_equity) 으로 목표 수량으로 환산, LOT 반올림 후 시장가
          전송. 같은 날 중복 전송을 막는 daily guard 포함. (SETTINGS.book_aum_usd 는
          risk.py 의 백테스트/참여율 캡 계산에만 쓰이는 고정 가정치이고, 실거래 주문
          수량 계산에는 더 이상 안 쓰임 -- 실계좌/테스트넷 잔고가 그 고정값과 안 맞아
          증거금 부족(-2019)이 나던 문제 때문에 동적 조회로 바꿈.)

FAIL-SAFE: target diagnostics.all_alphas_stale 이면 '청산 주문'을 만들지 않고
명시적으로 SKIP (오늘 목표를 신뢰할 수 없으므로 현재 포지션을 그대로 둔다).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from src.config.backtest_settings import SETTINGS

log = logging.getLogger("quant.live.orders")

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
    """target/current: {coin: weight}. -> [{coin,current_weight,target_weight,delta,side,close}]

    - |delta|>dust 인 코인만. 단 '목표 0 인데 보유 중'(close=True)인 코인은 dust 미만이어도
      포함해 완전 청산한다(예전엔 dust 미만 잔여 포지션이 영원히 안 닫혀 실계좌에 쌓임).
    - 정렬: 노출을 '줄이는' 주문(청산/축소) 먼저 -> 증거금을 확보한 뒤 신규/증액 주문.
      (증거금이 빠듯할 때 매수가 먼저 나가면 -2019 증거금 부족이 날 수 있음.)
      같은 그룹 안에서는 |delta| 큰 순."""
    coins = set(target_weights) | set(current_weights)
    orders = []
    for c in sorted(coins):
        cur = float(current_weights.get(c, 0.0))
        tgt = float(target_weights.get(c, 0.0))
        delta = tgt - cur
        is_close = tgt == 0.0 and cur != 0.0
        if abs(delta) < dust_threshold and not is_close:
            continue
        orders.append({"coin": c, "current_weight": cur, "target_weight": tgt,
                       "delta": delta, "side": "buy" if delta > 0 else "sell",
                       "close": is_close})
    # 노출 축소(|target| < |current|) 주문 먼저(False < True), 그 안에서 |delta| 큰 순.
    orders.sort(key=lambda o: (abs(o["target_weight"]) >= abs(o["current_weight"]),
                               -abs(o["delta"])))
    return orders


def _real_current_weights(aum_usd, positions=None):
    """실제 거래소 포지션 -> {coin: weight} (수량*마크가격 / aum_usd).
    aum_usd 는 그 사이클에서 조회한 실제 계좌 총자산(고정 book_aum_usd 아님).
    positions: 이미 조회한 {symbol: signed_qty} 를 주면 재조회하지 않는다(중복 API 방지).

    주의: 마크가격이 없는 심볼(상폐/거래정지 등)은 weight 를 못 구해 여기서 빠진다.
    그런 포지션이 목표에 없으면 weight-diff 만으로는 청산되지 않으므로, 실보유 기준
    청산(_reconcile_orphan_closes)이 별도로 담당한다."""
    from src.live.exchange import exchange
    if positions is None:
        positions = exchange.get_open_positions()
    if not positions:
        return {}
    prices = exchange.get_mark_prices()
    out = {}
    for symbol, qty in positions.items():
        price = prices.get(symbol)
        if price:
            out[symbol] = (qty * price) / aum_usd
    return out


def _reconcile_orphan_closes(orders, target_weights, open_positions,
                             dust_threshold=DUST_THRESHOLD_DEFAULT):
    """[원칙2: 진실은 거래소에 있다] 실제 거래소 보유 포지션 중 '오늘 목표에 없는데
    아직 청산 주문이 안 잡힌' 종목을 찾아 강제 청산 주문을 추가한다.

    왜 필요한가: 청산 여부를 '가중치 계산 결과(모델)'가 아니라 '실제 보유(현실)'로
    판단해야 한다. 마크가격이 없는 상폐/거래정지 코인은 _real_current_weights 에서
    빠져 current=0 으로 취급되고, 그러면 compute_orders 가 close(=목표0 & 보유중)를
    못 만들어 포지션이 영원히 안 닫히고 물린다. 여기서 실보유를 직접 훑어 메꾼다.

    close 주문은 마크가격이 필요 없다(_send_real_orders 가 reduce_only 로 실보유 수량을
    그대로 닫음)."""
    covered = {o["coin"] for o in orders}
    extra = []
    for symbol, qty in (open_positions or {}).items():
        if qty == 0.0 or symbol in covered:
            continue
        tgt = float(target_weights.get(symbol, 0.0))
        if abs(tgt) <= dust_threshold:  # 목표에 없음(사실상 0) → 실보유는 청산 대상
            extra.append({"coin": symbol, "current_weight": None, "target_weight": 0.0,
                          "delta": -tgt, "side": "sell" if qty > 0 else "buy",
                          "close": True, "orphan": True})
            log.info("고아 청산 추가: %s (실보유 qty=%s, 목표 없음/마크가격 무관)", symbol, qty)
    if extra:
        # 청산(노출 축소)이므로 기존 청산군과 함께 맨 앞에서 처리되도록 앞에 붙인다.
        return extra + orders
    return orders


def _client_order_id(today, symbol, side):
    """날짜·심볼·방향으로 결정적 client order id 생성(멱등 키). 같은 날 같은 방향 주문을
    재전송하면 거래소가 중복으로 거부 → 재실행 시 이중 진입 방지. 바이낸스 clientOrderId
    허용 문자(영숫자/-/_ 등)만 쓰고 36자 이내로 맞춘다."""
    import re
    raw = f"q-{today.isoformat()}-{symbol}-{side}"
    safe = re.sub(r"[^A-Za-z0-9_-]", "", raw)
    return safe[:36]


def _already_sent_today(today) -> bool:
    """daily guard: 오늘자 orders 기록이 real & not skipped'이고, 그중 실제로 거래소에
    전송 성공한 주문(exchange_result 존재)이 하나라도 있어야 '이미 전송됨'으로 본다.
    전부 error(프리플라이트 실패/마크가격 없음 등)뿐이면 아무것도 안 나간 것이므로
    재시도를 막지 않는다(예전엔 전부 실패해도 top-level skipped=False라 daily guard에
    걸려 하루 종일 재시도 불가였음)."""
    p = Path(SETTINGS.data_dir) / "runtime" / "live" / f"orders_{today.isoformat()}.json"
    if not p.exists():
        return False
    try:
        record = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False
    if record.get("mode") != "real" or record.get("skipped"):
        return False
    already = any(o.get("exchange_result") for o in record.get("orders", []))
    log.info("_already_sent_today(%s): record.mode=%s record.skipped=%s -> %s",
             today, record.get("mode"), record.get("skipped"), already)
    return already


# 체결로 인정하는 주문 상태(그 외 REJECTED/EXPIRED 등은 '전송했으나 미체결'로 본다).
_FILLED_STATUSES = {"NEW", "FILLED", "PARTIALLY_FILLED", None}


def _order_ok(result):
    """거래소 응답 status 가 체결/접수로 볼 수 있는 값인지. status 필드가 없으면(None)
    구버전 SDK 호환 위해 성공으로 간주(로그로 원본은 남는다)."""
    status = getattr(result, "status", None)
    return status in _FILLED_STATUSES, status


def _send_real_orders(orders, aum_usd, today, open_positions=None):
    """각 주문 dict 에 exchange_result 또는 error 를 채운다.
    aum_usd: 그 사이클에서 조회한 실제 계좌 총자산 -- 주문 수량 = delta * aum_usd / price.
    open_positions: 이미 조회한 실보유(청산 수량 계산용). 없으면 여기서 조회."""
    from src.live.exchange import exchange

    log.info("_send_real_orders 시작: %d건 %s (aum_usd=%.2f)",
             len(orders), [o["coin"] for o in orders], aum_usd)

    # 프리플라이트(fail-closed): 하나라도 어긋나면 전량 미전송.
    #   1) 원웨이 모드(계정 전역 설정 — positionSide="BOTH" 전제)
    #   2) 크로스 마진(심볼별 설정 — 오늘 매매할 종목만 검사, 롱숏 상호 증거금 전제)
    order_symbols = [o["coin"] for o in orders]
    preflight = []
    if SETTINGS.require_one_way_mode:
        preflight.append(exchange.assert_one_way_mode)
    if SETTINGS.require_cross_margin:
        preflight.append(lambda: exchange.assert_cross_margin(symbols=order_symbols))
    for check in preflight:
        try:
            check()
        except Exception as exc:
            log.exception("프리플라이트 실패")
            for o in orders:
                o["error"] = f"프리플라이트 실패(주문 미전송): {exc}"
            return

    prices = exchange.get_mark_prices()
    log.info("get_mark_prices 결과: %d개 심볼 (예: %s)",
             len(prices), list(prices.items())[:3])

    # 청산(close=True) 주문은 '실제 보유 수량'을 그대로 닫는다(reduce_only).
    # weight 기반 delta 환산·LOT 반올림을 거치면 잔여 수량이 남을 수 있기 때문.
    if open_positions is None:
        open_positions = {}
        if any(o.get("close") for o in orders):
            try:
                open_positions = exchange.get_open_positions()
            except Exception:
                log.exception("청산용 포지션 조회 실패 -- close 주문은 개별 에러 처리됨")

    for o in orders:
        symbol = o["coin"]
        if o.get("close"):
            pos_qty = float(open_positions.get(symbol, 0.0))
            if pos_qty == 0.0:
                o["error"] = "청산 대상인데 거래소 포지션 없음 -- 미전송"
                log.warning("%s: 청산 주문인데 실제 포지션 0", symbol)
                continue
            try:
                side = "SELL" if pos_qty > 0 else "BUY"
                coid = _client_order_id(today, symbol, side)
                result = exchange.place_market_order(symbol, side, abs(pos_qty),
                                                     reduce_only=True, client_order_id=coid)
                ok, status = _order_ok(result)
                if not ok:
                    o["error"] = f"청산 주문 거부/미체결 status={status}"
                    log.warning("%s: 청산 주문 미체결 status=%s", symbol, status)
                    continue
                o["exchange_result"] = {"status": status, "sent_quantity": abs(pos_qty),
                                        "reduce_only": True, "client_order_id": coid}
            except Exception as exc:
                log.exception("%s 청산 주문 처리 중 예외", symbol)
                o["error"] = f"{type(exc).__name__}: {exc}"
            continue
        price = prices.get(symbol)
        if not price:
            o["error"] = f"{symbol} mark price 없음 -- 미전송"
            log.warning("%s: mark price 조회 실패 (prices dict 에 없음)", symbol)
            continue
        raw_qty = (o["delta"] * aum_usd) / price
        try:
            exchange.set_leverage(symbol)
            qty = exchange.round_quantity(symbol, raw_qty)
            if qty == 0.0:
                o["error"] = "반올림 수량 0 (minQty 미만) -- 미전송"
                log.warning("%s: 반올림 후 수량 0 (raw_qty=%s)", symbol, raw_qty)
                continue
            # 최소 명목가(MIN_NOTIONAL) 로컬 검증 — 미만이면 -4164 거부 전에 스킵.
            notional = abs(qty) * price
            floor = exchange.min_notional(symbol)
            if notional < floor:
                o["error"] = f"명목가 {notional:.2f} < 최소 {floor:.2f} -- 미전송(스킵)"
                log.warning("%s: 명목가 %.2f < 최소 %.2f 스킵", symbol, notional, floor)
                continue
            side = "BUY" if raw_qty > 0 else "SELL"
            coid = _client_order_id(today, symbol, side)
            result = exchange.place_market_order(symbol, side, abs(qty), client_order_id=coid)
            ok, status = _order_ok(result)
            if not ok:
                o["error"] = f"주문 거부/미체결 status={status}"
                log.warning("%s: 주문 미체결 status=%s", symbol, status)
                continue
            o["exchange_result"] = {"status": status, "sent_quantity": abs(qty),
                                    "client_order_id": coid}
        except Exception as exc:
            log.exception("%s 주문 처리 중 예외", symbol)
            o["error"] = f"{type(exc).__name__}: {exc}"


def _write_order_record(record, today):
    """orders_<date>.json 저장.

    보호 로직 1(실전송): 오늘 이미 '실전송 성공' 기록이 있는데 새 record 에는 성공 전송이
    없으면 (daily guard skip, all_alphas_stale skip, 페이퍼 기록 등) 덮어쓰지 않는다.
    예전엔 guard 의 skip 기록이 원본 전송 기록을 덮어써서, 다음 실행 때
    _already_sent_today 가 skip 기록을 보고 '안 보냈다'고 판단 -> 주문 재전송되는
    (잠김<->풀림 반복) 버그가 있었다.

    보호 로직 2(재실행 기록 보존): 하루에 두 번 이상 실행되면(스케줄 중복·수동실행·배포)
    첫 실행이 리밸런싱을 마치고 positions.json 을 목표로 저장하므로, 그날 나중 실행은
    이미 목표=현재라 주문 0건이 나온다. 이 0건 기록이 아침의 진짜 리밸런싱 기록(43건 등)을
    덮어써 대시보드/텔레메트리가 '오늘 0건'으로 보이던 문제 -> 오늘 기록에 실제 주문이
    있었으면(n_orders>0), 새 기록의 주문이 그보다 적을 때는 덮어쓰지 않는다(진짜 매매 보존)."""
    out_path = Path(SETTINGS.data_dir) / "runtime" / "live" / f"orders_{today.isoformat()}.json"
    new_has_sent = any(o.get("exchange_result") for o in record.get("orders", []))
    if not new_has_sent and _already_sent_today(today):
        log.info("orders_%s.json 에 실전송 성공 기록 존재 -- 새 기록(전송 없음)으로 덮어쓰지 않음",
                 today.isoformat())
        return out_path
    # 보호 로직 2: 오늘 이미 더 많은 주문(진짜 리밸런싱) 기록이 있으면 no-op 재실행으로 덮지 않음.
    new_n = record.get("n_orders", len(record.get("orders", [])))
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            prev_n = prev.get("n_orders", len(prev.get("orders", [])))
            if prev_n > new_n:
                log.info("orders_%s.json 에 더 많은 주문 기록(%d건) 존재 -- 재실행 %d건으로 덮어쓰지 않음(진짜 매매 보존)",
                         today.isoformat(), prev_n, new_n)
                return out_path
        except Exception:
            pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


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
        aum_usd = None
        open_positions = None
        if mode == "real":
            from src.live.exchange import exchange
            try:
                aum_usd = exchange.get_account_equity()
            except Exception as exc:
                # fail-closed: 실제 계좌 크기를 모르면 주문 수량을 계산할 수 없다.
                # (예전엔 고정 가정치 book_aum_usd 로 폴백 -> 실계좌보다 크면 과대 주문 위험)
                log.exception("실제 계좌 총자산 조회 실패 -- fail-closed, 이번 사이클 주문 보류")
                record = {"date": today.isoformat(), "mode": mode, "skipped": True,
                          "skip_reason": f"계좌 총자산 조회 실패(fail-closed, 주문 미전송): {exc}",
                          "orders": [], "n_orders": 0}
                _write_order_record(record, today)
                return record
            # [원칙2] 실보유 포지션을 한 번만 조회해 (a)현재 가중치 환산, (b)고아 청산 판단,
            # (c)청산 수량 계산에 재사용한다(중복 API 방지).
            try:
                open_positions = exchange.get_open_positions()
            except Exception:
                log.exception("실보유 포지션 조회 실패 -- 고아 청산/청산수량은 개별 처리")
                open_positions = {}
        current = (_real_current_weights(aum_usd, positions=open_positions) if mode == "real"
                   else load_positions(positions_path))
        target = target_record.get("weights", {})
        drift = portfolio_drift(target, current)
        if rebalance_band > 0.0 and drift < rebalance_band:
            # 드리프트가 밴드 미만 -> 리밸런싱 보류, 현재 포지션 그대로.
            # (참고: 목표에 없는 '고아'(상폐 등) 포지션 청산은 다음 활성 리밸런싱에서 처리된다.
            #  밴드 미만은 회전율 절약용 보류일 뿐이며, 고아 청산이 분 단위로 급하진 않다.)
            record = {"date": today.isoformat(), "mode": mode, "skipped": True,
                      "skip_reason": (f"드리프트 {drift:.4f} < rebalance_band {rebalance_band:.4f} "
                                      "-- 아직 충분히 다르지 않음, 리밸런싱 보류(포지션 유지)"),
                      "orders": [], "n_orders": 0, "drift": drift,
                      "rebalance_band": rebalance_band}
            _write_order_record(record, today)
            return record
        orders = compute_orders(target, current, dust_threshold=dust_threshold)
        if mode == "real":
            # [원칙2] 실보유 기준으로 '목표에 없는데 아직 안 닫힌' 포지션을 청산 주문에 추가.
            orders = _reconcile_orphan_closes(orders, target, open_positions,
                                              dust_threshold=dust_threshold)
            if orders:
                _send_real_orders(orders, aum_usd, today, open_positions=open_positions)
        record = {"date": today.isoformat(), "mode": mode, "skipped": False,
                  "orders": orders, "n_orders": len(orders), "drift": drift,
                  "rebalance_band": rebalance_band,
                  "aum_usd": aum_usd,
                  "target_weights": target, "previous_weights": current}
        if mode == "paper":
            save_positions(target, positions_path)  # 즉시·완전 체결 가정

    _write_order_record(record, today)
    return record
