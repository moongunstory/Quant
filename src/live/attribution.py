"""attribution — 라이브 텔레메트리로 '성과/리스크 기여도'를 로컬 역추적.

진입점은 루트 main.py 의 `attribution` 서브커맨드다(이 모듈은 직접 실행하지 않음):

    python main.py attribution <telemetry.zip | telemetry폴더 | telemetry-*.json>
    python main.py attribution <경로> --out-dir ./분석결과

무엇을 하나 — 텔레그램으로 받은 telemetry zip(또는 telemetry-*.json 폴더)을 넣으면
백테스트를 다시 돌리지 않고도 아래를 계산해 표/CSV/그래프로 보여준다.

  1) 알파 기여도(Alpha Attribution)
       alpha_pnl[a] = Σ_coin (알파 a 의 결합북 기여분) × (다음날 실현수익률)
     → 어떤 알파가 실제로 돈을 벌었/잃었는지. 합 = '리스크 적용 전(gross)' 손익.

  2) 리스크 오버레이 기여도(Risk Drag / Benefit)
       gross_pnl = Σ (리스크 前 결합북)   × (다음날 수익률)
       net_pnl   = Σ (리스크 後 목표가중) × (다음날 수익률)
       risk_effect = net_pnl - gross_pnl   (음수=Drag 수익깎임, 양수=Benefit 방어이득)

핵심 타이밍(lag-1): T일 정한 목표가 T+1일 수익률을 번다. 스냅샷을 날짜순 정렬해
(T 스냅샷의 가중치) × (T+1 스냅샷의 day_returns) 로 짝지어 계산한다(paper 손익과 동일).

의존성: 표준 라이브러리만 필수. matplotlib 이 있으면 그래프 PNG 도 저장한다.
"""
from __future__ import annotations

import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path


# ------------------------------------------------------------------ 로드
def load_snapshots(source) -> list[dict]:
    """zip 또는 폴더/단일 json 에서 telemetry 스냅샷을 읽어 날짜 오름차순 리스트로."""
    p = Path(source)
    snapshots = []
    if p.is_dir():
        for f in sorted(p.glob("telemetry-*.json")):
            snapshots.append(json.loads(f.read_text(encoding="utf-8")))
    elif p.suffix == ".zip":
        with zipfile.ZipFile(p) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith(".json"):
                    snapshots.append(json.loads(zf.read(name).decode("utf-8")))
    elif p.suffix == ".json":
        snapshots.append(json.loads(p.read_text(encoding="utf-8")))
    else:
        raise ValueError(f"지원하지 않는 입력입니다(zip/폴더/json 만): {source}")
    if not snapshots:
        raise ValueError(f"텔레메트리 스냅샷을 찾지 못했습니다: {source}")
    snapshots.sort(key=lambda s: s.get("date", ""))
    return snapshots


# ------------------------------------------------------------------ 계산
def _dot(weights: dict, returns: dict) -> float:
    """Σ_coin weight[coin] * return[coin] (겹치는 코인만)."""
    return sum(float(w) * float(returns.get(c, 0.0)) for c, w in weights.items())


def analyze(snapshots: list[dict]) -> dict:
    """lag-1 짝짓기로 일별 알파/리스크 기여를 계산해 누적 결과를 반환."""
    daily = []
    alpha_cum = defaultdict(float)
    gross_cum = net_cum = 0.0
    skipped_stale = 0

    for src, nxt in zip(snapshots, snapshots[1:]):
        r = nxt.get("day_returns") or {}
        if not r:
            continue
        pre = src.get("pre_risk_weights") or {}
        post = src.get("target_weights") or {}
        contribs = src.get("alpha_contributions") or {}

        # 소스일에 신선한 목표가 없었으면(전부 stale) 기여 분해 불가 → 건너뜀.
        if src.get("all_alphas_stale") or (not pre and not post):
            skipped_stale += 1
            continue

        gross = _dot(pre, r)
        net = _dot(post, r)
        alpha_pnl = {a: _dot(w, r) for a, w in contribs.items()}
        for a, v in alpha_pnl.items():
            alpha_cum[a] += v
        gross_cum += gross
        net_cum += net
        daily.append({
            "return_date": nxt.get("date"),
            "signal_date": src.get("date"),
            "gross_pnl": gross,
            "net_pnl": net,
            "risk_effect": net - gross,
            "gross_cum": gross_cum,
            "net_cum": net_cum,
            "alpha_pnl": alpha_pnl,
        })

    return {
        "daily": daily,
        "alpha_cum": dict(alpha_cum),
        "gross_cum": gross_cum,
        "net_cum": net_cum,
        "risk_effect_cum": net_cum - gross_cum,
        "n_days": len(daily),
        "skipped_stale": skipped_stale,
        "span": (snapshots[0].get("date"), snapshots[-1].get("date")),
    }


# ------------------------------------------------------------------ 출력
def _fmt(x: float) -> str:
    return f"{x:+.4%}"


def print_report(res: dict):
    span0, span1 = res["span"]
    print("\n" + "=" * 60)
    print(f" 텔레메트리 기여도 분석  ({span0} ~ {span1})")
    print("=" * 60)
    print(f" 분석 대상 일수 : {res['n_days']}일  (stale 제외 {res['skipped_stale']}일)")
    print(f" 누적 gross(리스크前) : {_fmt(res['gross_cum'])}")
    print(f" 누적 net  (리스크後) : {_fmt(res['net_cum'])}")
    eff = res["risk_effect_cum"]
    verdict = "Benefit(방어 이득)" if eff >= 0 else "Drag(수익 깎임)"
    print(f" 리스크 오버레이 효과 : {_fmt(eff)}  → {verdict}")

    print("\n [알파별 누적 기여도] (gross 기준, 내림차순)")
    print(" " + "-" * 52)
    print(f" {'알파':<28}{'누적기여':>12}{'비중':>10}")
    print(" " + "-" * 52)
    total = sum(abs(v) for v in res["alpha_cum"].values()) or 1.0
    for a, v in sorted(res["alpha_cum"].items(), key=lambda kv: -kv[1]):
        print(f" {a:<28}{_fmt(v):>12}{abs(v) / total:>9.1%}")
    print(" " + "-" * 52)


def write_csv(res: dict, out_dir: Path) -> Path:
    """일별 gross/net/risk_effect + 알파별 기여를 CSV 로 저장."""
    alphas = sorted(res["alpha_cum"])
    path = out_dir / "attribution_report.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["return_date", "signal_date", "gross_pnl", "net_pnl",
                    "risk_effect", "gross_cum", "net_cum"]
                   + [f"alpha:{a}" for a in alphas])
        for d in res["daily"]:
            w.writerow([d["return_date"], d["signal_date"],
                        f"{d['gross_pnl']:.8f}", f"{d['net_pnl']:.8f}",
                        f"{d['risk_effect']:.8f}", f"{d['gross_cum']:.8f}",
                        f"{d['net_cum']:.8f}"]
                       + [f"{d['alpha_pnl'].get(a, 0.0):.8f}" for a in alphas])
    return path


def write_chart(res: dict, out_dir: Path):
    """matplotlib 이 있으면 자산곡선 + 알파 기여 막대 그래프를 PNG 로 저장(없으면 None)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    daily = res["daily"]
    if not daily:
        return None
    dates = [d["return_date"] for d in daily]
    gross = [d["gross_cum"] for d in daily]
    net = [d["net_cum"] for d in daily]

    # 라벨은 ASCII 고정 — 로컬(윈도우 기본 matplotlib)에 한글 폰트가 없으면 네모(tofu)로
    # 깨지므로 그래프 텍스트는 영문으로(콘솔/CSV 는 한글 그대로).
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9))
    ax1.plot(dates, [g * 100 for g in gross], label="Gross (pre-risk)", lw=1.8)
    ax1.plot(dates, [n * 100 for n in net], label="Net (post-risk)", lw=1.8)
    ax1.set_title("Cumulative PnL (%)")
    ax1.axhline(0, color="gray", lw=0.7)
    ax1.legend()
    step = max(1, len(dates) // 12)
    ax1.set_xticks(range(0, len(dates), step))
    ax1.set_xticklabels(dates[::step], rotation=45, ha="right", fontsize=8)

    items = sorted(res["alpha_cum"].items(), key=lambda kv: kv[1])
    names = [a for a, _ in items]
    vals = [v * 100 for _, v in items]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
    ax2.barh(names, vals, color=colors)
    ax2.set_title("Alpha attribution (cumulative %, gross basis)")
    ax2.axvline(0, color="gray", lw=0.7)

    fig.tight_layout()
    path = out_dir / "attribution_report.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run(source, out_dir=".") -> dict | None:
    """main.py attribution 진입점: 로드→분석→표/CSV/PNG 저장. 결과 dict 반환."""
    snapshots = load_snapshots(source)
    if len(snapshots) < 2:
        print(f"⚠️ 스냅샷이 2개 미만이라 lag-1 기여도 계산을 할 수 없습니다 "
              f"(읽은 스냅샷 {len(snapshots)}개). 하루 이상 더 쌓인 뒤 다시 실행하세요.")
        return None

    res = analyze(snapshots)
    print_report(res)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n CSV 저장 : {write_csv(res, out)}")
    png = write_chart(res, out)
    print(f" 그래프 저장 : {png}" if png
          else " (matplotlib 미설치 → 그래프 생략. `pip install matplotlib` 로 활성화)")
    return res
