"""dashboard — 초보자용 '한눈에 보는' 라이브 사이클 대시보드 PNG 생성.

왜 필요한가
-----------
기존 사이클 보고는 개발자용 텍스트라 한눈에 안 들어온다. 이 모듈은 matplotlib 로
매 사이클 결과를 그림 한 장(PNG)으로 그려, 텔레그램에 사진으로 띄운다.

담는 정보 (위→아래) — '고객이 궁금한 순서'대로
------------------------------------------------
  1) 날짜/모드 + 총자산·오늘 손익($)             ← "지금 얼마 벌었나/잃었나"
  2) 핵심 지표 4칸(최근30일·최대낙폭·승률·운용일수) ← "요즘 잘하고 있나"
  3) 자산 곡선 + '비트코인 그냥 보유' 비교선       ← "이 전략이 의미가 있나"
  4) 일별 손익 막대(최근 14일)                    ← "어느 날 벌고 잃었나"
  5) 롱/숏/순노출 막대                            ← "시장중립이 지켜지나"
  6) 오늘의 매매 / 상위 보유                      ← "오늘 뭘 사고팔고 뭘 들고 있나"
  7) 오늘 코인별 손익 기여(TOP 승/패)             ← "누가 벌어주고 누가 까먹었나"
  8) 요약 + 경고 배지(알파 stale / 킬스위치 등)   ← 문제 있을 때만

전부 fail-open 지향: 렌더 실패해도 호출측(telegram_bot)이 기존 텍스트로 폴백한다.
BTC 비교선은 paper_equity.jsonl 의 btc_return 필드(paper.py 가 기록)를 쓰므로
필드가 없는 옛 기록 구간에서는 자동으로 생략된다.
한국어 폰트는 data/strategy/assets/fonts 에 번들된 Noto Sans CJK 를 우선 사용하고,
없으면 시스템 폰트를 찾는다(로컬 개발/서버 어디서든 깨지지 않게).
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 서버/Lambda: 화면 없는 환경에서도 그리기
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

from src.config.backtest_settings import SETTINGS

log = logging.getLogger("quant.live.dashboard")

# ── 색상 팔레트 (다크, 폰 가독성 위주) ─────────────────────────────
BG = "#0e1117"        # 전체 배경
CARD = "#1a2029"      # 카드 배경
FG = "#e6edf3"        # 기본 글자
MUTED = "#8b949e"     # 보조 글자
GREEN = "#2ecc71"     # 이익/롱/매수
RED = "#e74c3c"       # 손실/숏/매도
BLUE = "#58a6ff"      # 강조선
YELLOW = "#f1c40f"    # 경고
ORANGE = "#f39c12"    # BTC 비교선

EQUITY_PATH = SETTINGS.data_dir / "runtime" / "live" / "paper_equity.jsonl"


# ── 한국어 폰트 로드 ───────────────────────────────────────────────
def _load_korean_font() -> fm.FontProperties:
    """번들 폰트 → 시스템 폰트 → 기본값 순으로 한국어 폰트를 찾는다."""
    bundled = SETTINGS.data_dir / "strategy" / "assets" / "fonts" / "NotoSansCJK-Regular.ttc"
    candidates = [
        bundled,
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]
    for c in candidates:
        try:
            if c.exists():
                fp = fm.FontProperties(fname=str(c))
                fm.fontManager.addfont(str(c))
                return fp
        except Exception:
            continue
    log.warning("한국어 폰트를 못 찾음 — 기본 폰트로 그립니다(한글 깨질 수 있음).")
    return fm.FontProperties()


_FONT = _load_korean_font()


def _pct(x: float) -> str:
    """비율(0.0123)을 사람이 읽는 % 문자열로. 부호 항상 표기."""
    return f"{x * 100:+.2f}%"


def _load_equity_history(limit: int = 365) -> list[dict]:
    """paper_equity.jsonl 마지막 limit 줄을 [{date, equity, day_pnl, ...}] 로."""
    p = Path(EQUITY_PATH)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-limit:]


def _compute_stats(rows: list[dict]) -> dict:
    """자산곡선에서 '요즘 잘하나' 핵심 지표를 계산.

    mdd(최대 낙폭): 누적수익률 곡선의 고점 대비 최대 하락폭. '중간에 최악일 때
    얼마나 까졌었나'라서 고객이 심리적으로 가장 궁금해하는 위험 지표다.
    winrate: 하루 손익이 +였던 날의 비율."""
    if not rows:
        return {"cum": 0.0, "ret30": 0.0, "mdd": 0.0, "winrate": None, "n_days": 0}
    eq = [float(r.get("equity", 0.0)) for r in rows]
    pnl = [float(r.get("day_pnl", 0.0)) for r in rows]
    cum = eq[-1]
    base30 = eq[-31] if len(eq) >= 31 else 0.0
    ret30 = cum - base30
    peak, mdd = float("-inf"), 0.0
    for e in eq:
        peak = max(peak, e)
        mdd = max(mdd, peak - e)
    wins = sum(1 for x in pnl if x > 0)
    return {"cum": cum, "ret30": ret30, "mdd": mdd,
            "winrate": wins / len(pnl) if pnl else None, "n_days": len(rows)}


def render_cycle_dashboard(res: dict, out_path: str | Path | None = None,
                           live_equity: float | None = None) -> Path | None:
    """사이클 결과(res)를 대시보드 PNG 로 그려 경로를 반환. 실패 시 None.

    res = {"target": {...}, "orders": {...}, "prev_positions": {...}}
    live_equity: /잔고 처럼 실시간 자산을 알면 넣어 큰 숫자에 반영(없으면 종가 기준).
    """
    try:
        return _render(res, out_path, live_equity)
    except Exception as e:
        log.error("대시보드 렌더 실패: %s", e, exc_info=True)
        return None


def _render(res, out_path, live_equity):
    t, o = res["target"], res["orders"]
    weights = t.get("weights", {}) or {}
    diag = t.get("diagnostics", {}) or {}
    orders = o.get("orders", []) or []
    prev_positions = res.get("prev_positions", {}) or {}
    day_returns = t.get("day_returns", {}) or {}

    hist_all = _load_equity_history(limit=365)
    hist = hist_all[-60:]  # 곡선/막대 표시 구간
    stats = _compute_stats(hist_all)
    cum_equity = live_equity if live_equity is not None else stats["cum"]
    today_pnl = float(hist[-1].get("day_pnl", 0.0)) if hist else 0.0

    # 롱/숏/순노출
    long_sum = sum(w for w in weights.values() if w > 0)
    short_sum = -sum(w for w in weights.values() if w < 0)
    gross = long_sum + short_sum
    net = long_sum - short_sum

    if out_path is None:
        out_path = SETTINGS.data_dir / "runtime" / "live" / "dashboard.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.6, 15.4), dpi=130)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def card(x, y, w, h, color=CARD):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.015",
                                    linewidth=0, facecolor=color, mutation_aspect=0.6))

    def text(x, y, s, size=13, color=FG, weight="normal", ha="left", va="center"):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                fontproperties=_FONT, fontweight=weight, transform=ax.transAxes)

    # 섹션을 위에서 아래로 쌓는 커서 — 좌표 실수를 막는다.
    GAP = 0.010
    y_cursor = [0.995]

    def section(h, split=False):
        """높이 h 카드를 얹고 (y0, y1) 반환. split=True 면 좌/우 두 카드."""
        y1 = y_cursor[0]
        y0 = y1 - h
        if split:
            card(0.04, y0, 0.44, h)
            card(0.52, y0, 0.44, h)
        else:
            card(0.04, y0, 0.92, h)
        y_cursor[0] = y0 - GAP
        return y0, y1

    mode = o.get("mode", "?")
    mode_kr = {"paper": "모의(paper)", "testnet": "테스트넷", "live": "실거래"}.get(mode, mode)

    # 총 자산(달러): 실계좌면 조회된 총자산, 모의면 기준자본 × (1+누적수익률)
    aum_usd = o.get("aum_usd")
    book_aum = float(getattr(SETTINGS, "book_aum_usd", 100_000.0))
    if aum_usd:
        total_assets = float(aum_usd)
        base_label = "실계좌 총자산"
    else:
        total_assets = book_aum * (1 + cum_equity)
        base_label = f"모의투자 · 시작자본 ${book_aum:,.0f}"
    profit_usd = total_assets - book_aum
    today_usd = today_pnl * book_aum

    pnl_color = GREEN if cum_equity >= 0 else RED
    tcolor = GREEN if today_pnl >= 0 else RED

    # ── 1) 헤더 + 총 자산 ────────────────────────────────────────
    y0, y1 = section(0.128)
    text(0.075, y1 - 0.016, f"{t.get('date','')}", size=15, color=MUTED)
    text(0.925, y1 - 0.016, f"{mode_kr}", size=13, color=MUTED, ha="right")
    text(0.075, y1 - 0.042, f"총 자산  ({base_label})", size=13, color=MUTED)
    text(0.075, y1 - 0.075, f"${total_assets:,.0f}", size=40, color=pnl_color, weight="bold")
    arrow = "▲" if cum_equity >= 0 else "▼"
    text(0.075, y1 - 0.108, f"{arrow} 시작 대비 {profit_usd:+,.0f}$  ({_pct(cum_equity)})",
         size=13, color=pnl_color)
    text(0.925, y1 - 0.060, f"누적 {_pct(cum_equity)}", size=15, color=pnl_color,
         weight="bold", ha="right")
    text(0.925, y1 - 0.088, f"오늘 {today_usd:+,.0f}$ ({_pct(today_pnl)})", size=14,
         color=tcolor, weight="bold", ha="right")

    # ── 2) 핵심 지표 4칸 ─────────────────────────────────────────
    y0, y1 = section(0.056)
    r30c = GREEN if stats["ret30"] >= 0 else RED
    win = stats["winrate"]
    cells = [
        ("최근 30일", _pct(stats["ret30"]), r30c),
        ("최대 낙폭(MDD)", f"-{stats['mdd'] * 100:.2f}%", YELLOW if stats["mdd"] > 0.03 else FG),
        ("하루 단위 승률", f"{win * 100:.0f}%" if win is not None else "—",
         GREEN if (win or 0) >= 0.5 else RED),
        ("운용 일수", f"{stats['n_days']}일", FG),
    ]
    centers = [0.155, 0.385, 0.615, 0.845]
    for (label, value, vcol), cx in zip(cells, centers):
        text(cx, y1 - 0.016, label, size=10, color=MUTED, ha="center")
        text(cx, y1 - 0.040, value, size=15, color=vcol, weight="bold", ha="center")

    # ── 3) 자산 곡선 + BTC 비교선 ────────────────────────────────
    y0, y1 = section(0.185)
    text(0.075, y1 - 0.016, "자산 곡선 (최근)", size=14, color=FG, weight="bold")
    eq = [float(r.get("equity", 0.0)) for r in hist]
    has_btc = any("btc_return" in r for r in hist)
    if has_btc:
        text(0.925, y1 - 0.016, "─ 내 전략   ┄ 비트코인 그냥 보유", size=10,
             color=MUTED, ha="right")
    if len(hist) >= 2:
        axc = fig.add_axes([0.10, y0 + 0.014, 0.80, 0.185 - 0.050]); axc.set_facecolor(CARD)
        base0 = eq[0]
        strat = [e - base0 for e in eq]  # 표시구간 시작을 0으로 → BTC 와 같은 출발점
        xs = list(range(len(strat)))
        line_c = GREEN if strat[-1] >= 0 else RED
        axc.plot(xs, strat, color=line_c, linewidth=2.4)
        axc.fill_between(xs, strat, 0, color=line_c, alpha=0.10)
        if has_btc:
            btc, c = [], 0.0
            for r in hist:
                c += float(r.get("btc_return", 0.0))
                btc.append(c)
            axc.plot(xs, btc, color=ORANGE, linewidth=1.6, linestyle="--", alpha=0.9)
        axc.axhline(0, color=MUTED, linewidth=0.7, linestyle="--", alpha=0.6)
        for s in axc.spines.values():
            s.set_visible(False)
        axc.tick_params(colors=MUTED, labelsize=8)
        axc.set_xticks([0, len(xs) - 1])
        axc.set_xticklabels([hist[0]["date"][5:], hist[-1]["date"][5:]], fontproperties=_FONT)
        axc.margins(x=0.02)
    else:
        text(0.5, (y0 + y1) / 2, "데이터가 아직 부족해요 (2일 이상 쌓이면 그려집니다)",
             size=12, color=MUTED, ha="center")

    # ── 4) 일별 손익 막대 (최근 14일) ────────────────────────────
    y0, y1 = section(0.105)
    text(0.075, y1 - 0.015, "일별 손익 (최근 14일)", size=13, color=FG, weight="bold")
    recent = hist[-14:]
    if len(recent) >= 2:
        axb = fig.add_axes([0.10, y0 + 0.012, 0.80, 0.105 - 0.045]); axb.set_facecolor(CARD)
        vals = [float(r.get("day_pnl", 0.0)) * 100 for r in recent]
        cols = [GREEN if v >= 0 else RED for v in vals]
        axb.bar(range(len(vals)), vals, color=cols, width=0.62)
        axb.axhline(0, color=MUTED, linewidth=0.7, alpha=0.6)
        for s in axb.spines.values():
            s.set_visible(False)
        axb.tick_params(colors=MUTED, labelsize=8)
        axb.set_xticks([0, len(vals) - 1])
        axb.set_xticklabels([recent[0]["date"][5:], recent[-1]["date"][5:]],
                            fontproperties=_FONT)
        axb.set_yticks(axb.get_yticks()[::2])  # y 눈금 절반만(빽빽함 방지)
        axb.margins(x=0.02)
    else:
        text(0.5, (y0 + y1) / 2 - 0.008, "아직 표시할 일별 기록이 없어요",
             size=11, color=MUTED, ha="center")

    # ── 5) 롱/숏/순노출 ──────────────────────────────────────────
    y0, y1 = section(0.064)
    text(0.075, y1 - 0.014, "롱 / 숏 균형", size=14, color=FG, weight="bold")
    text(0.925, y1 - 0.014, f"순노출 {net:+.1%}", size=13,
         color=(GREEN if abs(net) < 0.1 else YELLOW), ha="right")
    bx, bw, bh = 0.075, 0.85, 0.014
    by = y0 + 0.022
    total = gross if gross > 1e-9 else 1.0
    lw = bw * (long_sum / total)
    ax.add_patch(plt.Rectangle((bx, by), lw, bh, transform=ax.transAxes, facecolor=GREEN, linewidth=0))
    ax.add_patch(plt.Rectangle((bx + lw, by), bw - lw, bh, transform=ax.transAxes, facecolor=RED, linewidth=0))
    text(0.075, y0 + 0.010, f"롱 {long_sum:.0%}", size=11, color=GREEN)
    text(0.925, y0 + 0.010, f"숏 {short_sum:.0%}", size=11, color=RED, ha="right")

    # ── 6) 오늘의 매매 / 상위 보유 ───────────────────────────────
    y0, y1 = section(0.155, split=True)
    text(0.06, y1 - 0.015, "오늘의 매매", size=13, color=FG, weight="bold")
    n_orders = o.get("n_orders", 0)
    if o.get("skipped"):
        text(0.06, y1 - 0.055, "리밸런싱 건너뜀", size=12, color=YELLOW)
        text(0.06, y1 - 0.078, str(o.get("skip_reason", ""))[:40], size=9, color=MUTED)
    else:
        text(0.46, y1 - 0.015, f"{n_orders}건", size=13, color=MUTED, ha="right")
        y = y1 - 0.040
        for od in orders[:6]:
            is_buy = od["side"] == "buy"
            tag = "매수" if is_buy else "매도"
            col = GREEN if is_buy else RED
            coin = od["coin"].replace("USDT", "")
            text(0.06, y, f"{'▲' if is_buy else '▼'} {tag}", size=10, color=col)
            text(0.185, y, coin[:9], size=10, color=FG)
            text(0.46, y, f"{od['current_weight']:+.3f}→{od['target_weight']:+.3f}",
                 size=8.5, color=MUTED, ha="right")
            y -= 0.019

    text(0.54, y1 - 0.015, "상위 보유", size=13, color=FG, weight="bold")
    text(0.94, y1 - 0.015, f"{len(weights)}종목", size=11, color=MUTED, ha="right")
    top = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:6]
    y = y1 - 0.040
    for coin, w in top:
        is_long = w > 0
        col = GREEN if is_long else RED
        name = coin.replace("USDT", "")
        text(0.54, y, ("롱 " if is_long else "숏 ") + name[:9], size=10, color=FG)
        text(0.94, y, f"{w:+.3f}", size=10, color=col, ha="right")
        y -= 0.018

    # ── 7) 오늘 코인별 손익 기여 ─────────────────────────────────
    # '오늘 번 돈은 누가 벌어줬나' — 어제 보유(prev_positions) × 당일 수익률.
    y0, y1 = section(0.098)
    text(0.075, y1 - 0.015, "오늘 손익 기여 — 누가 벌어주고 누가 까먹었나",
         size=13, color=FG, weight="bold")
    contrib = {c: float(w) * float(day_returns.get(c, 0.0))
               for c, w in prev_positions.items() if day_returns.get(c) is not None}
    contrib = {c: v for c, v in contrib.items() if abs(v) > 1e-9}
    if contrib:
        ranked = sorted(contrib.items(), key=lambda kv: -kv[1])
        winners = [kv for kv in ranked if kv[1] > 0][:3]
        losers = [kv for kv in sorted(contrib.items(), key=lambda kv: kv[1]) if kv[1] < 0][:3]
        text(0.075, y1 - 0.038, "벌어준 코인", size=10, color=MUTED)
        text(0.535, y1 - 0.038, "까먹은 코인", size=10, color=MUTED)
        y = y1 - 0.058
        for coin, v in winners:
            text(0.075, y, coin.replace("USDT", "")[:10], size=10, color=FG)
            text(0.46, y, f"{v * 100:+.3f}%p", size=10, color=GREEN, ha="right")
            y -= 0.017
        y = y1 - 0.058
        for coin, v in losers:
            text(0.535, y, coin.replace("USDT", "")[:10], size=10, color=FG)
            text(0.925, y, f"{v * 100:+.3f}%p", size=10, color=RED, ha="right")
            y -= 0.017
    else:
        text(0.5, (y0 + y1) / 2 - 0.008, "오늘 기여 데이터가 없어요 (수익률/보유 정보 부족)",
             size=11, color=MUTED, ha="center")

    # ── 8) 하단 요약 + 경고 ──────────────────────────────────────
    y0, y1 = section(0.102)
    text(0.075, y1 - 0.015, "요약", size=14, color=FG, weight="bold")
    drift = o.get("drift", 0.0)
    text(0.075, y1 - 0.038,
         f"이번 목표: {len(weights)}종목 보유  →  오늘 사고판 횟수: {n_orders}건",
         size=11.5, color=FG)
    text(0.075, y1 - 0.058,
         f"어긋난 정도(드리프트): {drift * 100:.1f}%  "
         f"— 어제 포지션이 목표에서 이만큼 벌어져 있어 그만큼 조정했어요",
         size=10, color=MUTED)
    alphas = ", ".join(t.get("held_alphas", []))
    text(0.075, y1 - 0.076, f"작동 전략(알파): {alphas}", size=9.5, color=MUTED)

    warns = []
    if diag.get("all_alphas_stale"):
        warns.append("[경고] 모든 알파 STALE — 포지션 유지(신규 목표 없음)")
    elif diag.get("stale_alphas"):
        warns.append(f"[주의] 일부 알파 stale 제외: {diag.get('stale_alphas')}")
    if o.get("killswitch") or diag.get("killswitch"):
        warns.append("[경고] 킬스위치 작동 — 리스크 축소")
    if warns:
        text(0.075, y0 + 0.008, warns[0][:60], size=10.5, color=YELLOW)
    else:
        text(0.075, y0 + 0.008, "특이사항 없음 — 정상 작동 중", size=10.5, color=GREEN)

    text(0.5, max(y_cursor[0] - 0.004, 0.008),
         "Quant 자동매매 · 개별 종목 숫자(+0.021 등)는 전체 자본 대비 비중입니다",
         size=9, color=MUTED, ha="center")

    fig.savefig(out_path, facecolor=BG, bbox_inches=None)
    plt.close(fig)
    return out_path
