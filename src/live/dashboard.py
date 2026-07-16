"""dashboard — 초보자용 '한눈에 보는' 라이브 사이클 대시보드 PNG 생성.

왜 필요한가
-----------
기존 사이클 보고는 개발자용 텍스트라 한눈에 안 들어온다. 이 모듈은 matplotlib 로
매 사이클 결과를 그림 한 장(PNG)으로 그려, 텔레그램에 사진으로 띄운다.

담는 정보 (위→아래)
--------------------
  1) 날짜/모드 + 큰 손익 숫자(누적 %, 오늘 %)   ← "지금 얼마 벌었나/잃었나"
  2) 자산 곡선(equity curve)                     ← "잘 되고 있나" 추세
  3) 롱/숏/순노출 막대                            ← "시장중립이 지켜지나"
  4) 오늘의 매매(산 것/판 것)                     ← "오늘 뭘 사고팔았나"
  5) 상위 보유 포지션                             ← "지금 뭘 크게 들고 있나"
  6) 경고 배지(알파 stale / 킬스위치 등)          ← 문제 있을 때만

전부 fail-open 지향: 렌더 실패해도 호출측(telegram_bot)이 기존 텍스트로 폴백한다.
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


def _load_equity_history(limit: int = 60) -> list[dict]:
    """paper_equity.jsonl 마지막 limit 줄을 [{date, equity, day_pnl}] 로."""
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


def render_cycle_dashboard(res: dict, out_path: str | Path | None = None,
                           live_equity: float | None = None) -> Path | None:
    """사이클 결과(res)를 대시보드 PNG 로 그려 경로를 반환. 실패 시 None.

    res = {"target": {...}, "orders": {...}}  (handler.run_cycle 반환형)
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

    hist = _load_equity_history()
    cum_equity = live_equity if live_equity is not None else (hist[-1]["equity"] if hist else 0.0)
    today_pnl = hist[-1]["day_pnl"] if hist else 0.0

    # 롱/숏/순노출
    long_sum = sum(w for w in weights.values() if w > 0)
    short_sum = -sum(w for w in weights.values() if w < 0)
    gross = long_sum + short_sum
    net = long_sum - short_sum

    if out_path is None:
        out_path = SETTINGS.data_dir / "runtime" / "live" / "dashboard.png"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.6, 12.4), dpi=130)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    def card(x, y, w, h, color=CARD):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                    linewidth=0, facecolor=color, mutation_aspect=0.74))

    def text(x, y, s, size=13, color=FG, weight="normal", ha="left", va="center"):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                fontproperties=_FONT, fontweight=weight, transform=ax.transAxes)

    mode = o.get("mode", "?")
    mode_kr = {"paper": "모의(paper)", "testnet": "테스트넷", "live": "실거래"}.get(mode, mode)

    # 총 자산(달러) 계산: 실계좌면 조회된 총자산, 모의면 기준자본 × (1+누적수익률)
    aum_usd = o.get("aum_usd")
    book_aum = float(getattr(SETTINGS, "book_aum_usd", 100_000.0))
    if aum_usd:
        total_assets = float(aum_usd)
        base_label = "실계좌 총자산"
    else:
        total_assets = book_aum * (1 + cum_equity)
        base_label = f"모의투자 · 시작자본 ${book_aum:,.0f}"
    profit_usd = total_assets - book_aum

    # ── 1) 헤더 + 총 자산(크게) ───────────────────────────────────
    card(0.04, 0.775, 0.92, 0.20)
    text(0.075, 0.945, f"{t.get('date','')}", size=15, color=MUTED)
    text(0.925, 0.945, f"{mode_kr}", size=13, color=MUTED, ha="right")

    pnl_color = GREEN if cum_equity >= 0 else RED
    tcolor = GREEN if today_pnl >= 0 else RED
    arrow = "▲" if cum_equity >= 0 else "▼"

    text(0.075, 0.898, f"총 자산  ({base_label})", size=13, color=MUTED)
    text(0.075, 0.85, f"${total_assets:,.0f}", size=42, color=pnl_color, weight="bold")
    text(0.075, 0.805, f"{arrow} 시작 대비 {profit_usd:+,.0f}$  ({_pct(cum_equity)})",
         size=13, color=pnl_color)

    # 오른쪽: 누적 / 오늘 요약
    text(0.925, 0.865, f"누적 {_pct(cum_equity)}", size=15, color=pnl_color,
         weight="bold", ha="right")
    text(0.925, 0.825, f"오늘 {_pct(today_pnl)}", size=15, color=tcolor,
         weight="bold", ha="right")

    # ── 2) 자산 곡선 ─────────────────────────────────────────────
    card(0.04, 0.585, 0.92, 0.19)
    text(0.075, 0.755, "자산 곡선 (최근)", size=14, color=FG, weight="bold")
    if len(hist) >= 2:
        axc = fig.add_axes([0.10, 0.60, 0.80, 0.125]); axc.set_facecolor(CARD)
        eq = [r["equity"] for r in hist]
        xs = list(range(len(eq)))
        line_c = GREEN if eq[-1] >= eq[0] else RED
        axc.plot(xs, eq, color=line_c, linewidth=2.4)
        axc.fill_between(xs, eq, min(eq), color=line_c, alpha=0.12)
        axc.axhline(0, color=MUTED, linewidth=0.7, linestyle="--", alpha=0.6)
        for s in axc.spines.values():
            s.set_visible(False)
        axc.tick_params(colors=MUTED, labelsize=8)
        axc.set_xticks([0, len(eq) - 1])
        axc.set_xticklabels([hist[0]["date"][5:], hist[-1]["date"][5:]], fontproperties=_FONT)
        axc.margins(x=0.02)
    else:
        text(0.5, 0.66, "데이터가 아직 부족해요 (2일 이상 쌓이면 그려집니다)",
             size=12, color=MUTED, ha="center")

    # ── 3) 롱/숏/순노출 ──────────────────────────────────────────
    card(0.04, 0.455, 0.92, 0.115)
    text(0.075, 0.552, "롱 / 숏 균형", size=14, color=FG, weight="bold")
    text(0.925, 0.552, f"순노출 {net:+.1%}", size=13,
         color=(GREEN if abs(net) < 0.1 else YELLOW), ha="right")
    # 막대 (총노출을 1로 정규화)
    bx, bw, by, bh = 0.075, 0.85, 0.478, 0.03
    total = gross if gross > 1e-9 else 1.0
    lw = bw * (long_sum / total)
    ax.add_patch(plt.Rectangle((bx, by), lw, bh, transform=ax.transAxes, facecolor=GREEN, linewidth=0))
    ax.add_patch(plt.Rectangle((bx + lw, by), bw - lw, bh, transform=ax.transAxes, facecolor=RED, linewidth=0))
    text(0.075, 0.468, f"롱 {long_sum:.0%}", size=12, color=GREEN)
    text(0.925, 0.468, f"숏 {short_sum:.0%}", size=12, color=RED, ha="right")

    # ── 4) 오늘의 매매 ───────────────────────────────────────────
    card(0.04, 0.25, 0.44, 0.19)
    text(0.06, 0.422, "오늘의 매매", size=13, color=FG, weight="bold")
    n_orders = o.get("n_orders", 0)
    if o.get("skipped"):
        text(0.06, 0.35, "리밸런싱 건너뜀", size=12, color=YELLOW)
        text(0.06, 0.315, str(o.get("skip_reason", ""))[:40], size=9, color=MUTED)
    else:
        text(0.46, 0.422, f"{n_orders}건", size=13, color=MUTED, ha="right")
        y = 0.388
        for od in orders[:6]:
            is_buy = od["side"] == "buy"
            tag = "매수" if is_buy else "매도"
            col = GREEN if is_buy else RED
            coin = od["coin"].replace("USDT", "")
            text(0.06, y, f"{'▲' if is_buy else '▼'} {tag}", size=10, color=col)
            text(0.185, y, coin[:9], size=10, color=FG)
            text(0.46, y, f"{od['current_weight']:+.3f}→{od['target_weight']:+.3f}",
                 size=8.5, color=MUTED, ha="right")
            y -= 0.024

    # ── 5) 상위 보유 포지션 ──────────────────────────────────────
    card(0.52, 0.25, 0.44, 0.19)
    text(0.54, 0.422, "상위 보유", size=13, color=FG, weight="bold")
    text(0.94, 0.422, f"{len(weights)}종목", size=11, color=MUTED, ha="right")
    top = sorted(weights.items(), key=lambda kv: -abs(kv[1]))[:6]
    y = 0.388
    for coin, w in top:
        is_long = w > 0
        col = GREEN if is_long else RED
        name = coin.replace("USDT", "")
        text(0.54, y, ("롱 " if is_long else "숏 ") + name[:9], size=10, color=FG)
        text(0.94, y, f"{w:+.3f}", size=10, color=col, ha="right")
        y -= 0.024

    # ── 6) 하단 요약 + 경고 ──────────────────────────────────────
    card(0.04, 0.085, 0.92, 0.14)
    text(0.075, 0.208, "요약", size=14, color=FG, weight="bold")
    # 목표/주문을 초보자도 알게 풀어서
    drift = o.get("drift", 0.0)
    text(0.075, 0.176,
         f"이번 목표: {len(weights)}종목 보유  →  오늘 사고판 횟수: {n_orders}건",
         size=11.5, color=FG)
    text(0.075, 0.148,
         f"어긋난 정도(드리프트): {drift * 100:.1f}%  "
         f"— 어제 포지션이 목표에서 이만큼 벌어져 있어 그만큼 조정했어요",
         size=10, color=MUTED)
    alphas = ", ".join(t.get("held_alphas", []))
    text(0.075, 0.122, f"작동 전략(알파): {alphas}", size=9.5, color=MUTED)

    warns = []
    if diag.get("all_alphas_stale"):
        warns.append("[경고] 모든 알파 STALE — 포지션 유지(신규 목표 없음)")
    elif diag.get("stale_alphas"):
        warns.append(f"[주의] 일부 알파 stale 제외: {diag.get('stale_alphas')}")
    if o.get("killswitch") or diag.get("killswitch"):
        warns.append("[경고] 킬스위치 작동 — 리스크 축소")
    if warns:
        text(0.075, 0.098, warns[0][:60], size=10.5, color=YELLOW)
    else:
        text(0.075, 0.098, "특이사항 없음 — 정상 작동 중", size=10.5, color=GREEN)

    text(0.5, 0.03,
         "Quant 자동매매 · 개별 종목 숫자(+0.021 등)는 전체 자본 대비 비중입니다",
         size=9, color=MUTED, ha="center")

    fig.savefig(out_path, facecolor=BG, bbox_inches=None)
    plt.close(fig)
    return out_path
