"""trade_history — 텔레메트리 스냅샷으로 '끝난 매매(진입→종료)'를 복원.

텔레그램 /기록 명령의 계산 엔진. 별도의 매매 일지 파일을 새로 만들지 않고,
이미 매일 쌓이는 telemetry-<date>.json (플라이트 레코더)만으로 각 코인의
보유 에피소드(진입일→종료일→손익)를 역산한다.

원리
----
스냅샷 하나(날짜 T)에는 세 가지가 있다.
  prev_positions : 오늘 리밸런싱 '직전' 보유(= 어제 정한 포지션). 오늘 수익을 번 주체.
  day_returns    : 오늘 실현 수익률 {coin: return}.
  target_weights : 오늘 리밸런싱 '후' 보유(주문 SKIP이면 prev 유지).

그래서 날짜순으로 훑으며,
  1) 열려 있는 에피소드에 오늘 손익(prev × day_return)을 적립하고,
  2) 리밸런싱 후 보유(eff)를 보고 에피소드를 열고/닫는다.
     - eff 에서 사라짐(또는 방향이 뒤집힘) → 그 날짜로 종료(close).
     - eff 에 새로 등장 → 그 날짜로 진입(open).

손익 단위는 '전체 계좌 대비 비율'(예: +0.0012 = 계좌의 +0.12%)로,
paper.mark_to_market 과 동일한 회계라 /잔고 누적치와 합이 맞는다.

주의: 텔레메트리는 90일 보존(prune)이므로 그보다 오래된 기록은 복원 불가.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("quant.live.trade_history")


def _effective_positions(snap: dict) -> dict:
    """스냅샷의 리밸런싱 '후' 실제 보유. 주문 SKIP이면 직전 보유 그대로."""
    orders = snap.get("orders") or {}
    if orders.get("skipped"):
        return {c: float(w) for c, w in (snap.get("prev_positions") or {}).items()}
    return {c: float(w) for c, w in (snap.get("target_weights") or {}).items()}


def build_episodes(snapshots: list[dict]) -> tuple[list[dict], list[dict]]:
    """날짜순 스냅샷 리스트 -> (closed, still_open) 에피소드 리스트.

    에피소드 dict:
      coin, side(+1 롱/-1 숏), entry_date, exit_date(열려있으면 None),
      entry_approx(관측 시작 전부터 이미 보유했으면 True), days(보유 스냅샷 수),
      avg_weight/max_weight(|비중| 평균/최대), pnl(계좌 대비 비율)
    """
    closed: list[dict] = []
    open_eps: dict[str, dict] = {}
    first = True

    for snap in sorted(snapshots, key=lambda s: s.get("date", "")):
        date = snap.get("date")
        prev = snap.get("prev_positions") or {}
        rets = snap.get("day_returns") or {}

        # 1) 오늘 하루 손익 적립: 오늘 수익을 번 건 '리밸런싱 전' 보유(prev)다.
        for coin, w in prev.items():
            ep = open_eps.get(coin)
            r = rets.get(coin)
            if ep is not None and r is not None:
                ep["pnl"] += float(w) * float(r)

        # 2) 리밸런싱 후 보유로 에피소드 갱신
        eff = _effective_positions(snap)

        # 2-a) 닫기: 사라졌거나 방향이 뒤집힌 코인
        for coin in list(open_eps):
            w = eff.get(coin, 0.0)
            ep = open_eps[coin]
            flipped = w != 0.0 and (w > 0) != (ep["side"] > 0)
            if w == 0.0 or flipped:
                ep["exit_date"] = date
                closed.append(open_eps.pop(coin))

        # 2-b) 열기/이어가기
        for coin, w in eff.items():
            if w == 0.0:
                continue
            ep = open_eps.get(coin)
            if ep is None:
                open_eps[coin] = {
                    "coin": coin,
                    "side": 1 if w > 0 else -1,
                    "entry_date": date,
                    # 첫 스냅샷에서 이미 직전 보유에도 있었다면 실제 진입은 관측 이전
                    "entry_approx": bool(first and coin in prev),
                    "exit_date": None,
                    "pnl": 0.0,
                    "_weights": [abs(w)],
                }
            else:
                ep["_weights"].append(abs(w))
        first = False

    def _finish(ep: dict) -> dict:
        ws = ep.pop("_weights", []) or [0.0]
        ep["days"] = len(ws)
        ep["avg_weight"] = sum(ws) / len(ws)
        ep["max_weight"] = max(ws)
        return ep

    closed = [_finish(e) for e in closed]
    still_open = [_finish(e) for e in open_eps.values()]
    closed.sort(key=lambda e: (e["exit_date"] or "", e["coin"]))
    return closed, still_open


def load_closed_trades(days: int = 90, today=None):
    """최근 `days`일 텔레메트리를 읽어 (closed, still_open, (시작일, 끝일)) 반환."""
    from src.live import ledger as LG

    files = LG.list_recent_telemetry(days=days, today=today)
    snaps = []
    for p in files:
        try:
            snaps.append(json.loads(Path(p).read_text(encoding="utf-8")))
        except Exception as e:
            log.warning("텔레메트리 파싱 실패(건너뜀) %s: %s", p, e)
    if not snaps:
        return [], [], (None, None)
    snaps.sort(key=lambda s: s.get("date", ""))
    closed, still_open = build_episodes(snaps)
    span = (snaps[0].get("date"), snaps[-1].get("date"))
    return closed, still_open, span
