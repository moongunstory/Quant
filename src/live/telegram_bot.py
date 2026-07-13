"""telegram_bot — 라이브 봇 텔레그램 한글 명령어 핸들러 및 메시지 전송 모듈.

추가 라이브러리 설치가 필요 없도록 파이썬 내장 urllib.request 모듈을 사용하며,
.env 에 지정된 ALLOWED_CHAT_ID 보안 규칙을 엄격하게 준수합니다.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from src.live.state import load_live_state, save_live_state
from src.live import orders as OR

log = logging.getLogger("quant.live.telegram_bot")

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ALLOWED_CHAT_ID = os.environ.get("TELEGRAM_ALLOWED_CHAT_ID")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or ALLOWED_CHAT_ID


def send_telegram_message(text: str, parse_mode: str = "HTML") -> dict | None:
    """텔레그램 채널로 포맷팅된 메시지를 전송합니다."""
    if not TOKEN or not CHAT_ID:
        log.warning("텔레그램 설정(TOKEN, CHAT_ID)이 유실되어 전송을 건너뜁니다.")
        return None
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = urllib.parse.urlencode({
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": parse_mode
    }).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        log.error(f"텔레그램 메시지 전송 실패: {e}")
        return None


def send_cycle_report(res: dict):
    """라이브 사이클 실행 요약을 보기 좋게 포맷하여 텔레그램으로 전송합니다."""
    t, o = res["target"], res["orders"]
    lines = []
    lines.append(f"<b>=== 라이브 사이클 실행 완료 ({t['date']}) ===</b>")
    lines.append(f"• 실행 모드: <code>{o['mode']}</code>")

    if t["diagnostics"].get("all_alphas_stale"):
        lines.append("⚠️ <b>모든 알파 STALE -> 포지션 유지(SKIP)</b>")
        lines.append(f"• stale: {t['diagnostics'].get('stale_alphas')}")
    else:
        if t["diagnostics"].get("stale_alphas"):
            lines.append(f"• 일부 stale(제외): {t['diagnostics']['stale_alphas']}")
        lines.append(f"• 보유 알파: <code>{t['held_alphas']}</code>")
        lines.append(f"• 목표 코인 수: <b>{len(t['weights'])}종목</b>")

        if o.get("skipped"):
            lines.append(f"• 주문 건너뜀: <i>{o.get('skip_reason')}</i>")
        else:
            lines.append(f"• 주문 건수: <b>{o['n_orders']}건</b> (Drift: {o.get('drift', 0.0):.4f})")
            if o["orders"]:
                lines.append("\n<b>[주요 포지션 이동 (상위 5개)]</b>")
                for od in o["orders"][:5]:
                    lines.append(f"• {od['side'].upper()} <code>{od['coin']}</code> ({od['current_weight']:+.4f} → {od['target_weight']:+.4f})")

    send_telegram_message("\n".join(lines))


# ---- 명령어 핸들러 구현 ----

def cmd_status():
    cfg = load_live_state()
    pos = OR.load_positions()

    lines = []
    lines.append("<b>[봇 상태 정보]</b>")
    lines.append(f"• 작동 상태: <b>{'🟢 작동 중' if cfg['enabled'] else '🔴 중단됨'}</b>")
    lines.append(f"• 실매매 모드: <code>{cfg['mode']}</code>")
    lines.append(f"• 포트폴리오 설정: <code>{cfg['config_path']}</code>")

    # 마지막 실행일 체크
    last_run = "기록 없음"
    equity_file = Path("data/runtime/live/paper_equity.jsonl")
    if equity_file.exists():
        eq_lines = [l for l in equity_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if eq_lines:
            try:
                last_run = json.loads(eq_lines[-1]).get("date", "기록 없음")
            except Exception:
                pass
    lines.append(f"• 마지막 실행일: <code>{last_run}</code>")

    # 포지션 요약
    lines.append(f"\n• 현재 가상 포지션 수: <b>{len(pos)}종목</b>")
    if pos:
        sorted_pos = sorted(pos.items(), key=lambda x: -abs(x[1]))
        lines.append("<b>[상위 5개 포지션]</b>")
        for coin, weight in sorted_pos[:5]:
            lines.append(f"  • <code>{coin}</code>: {weight:+.4f}")

    send_telegram_message("\n".join(lines))


def cmd_positions():
    pos = OR.load_positions()
    if not pos:
        send_telegram_message("현재 보유 중인 포지션이 없습니다.")
        return

    lines = ["<b>[보유 포지션 전체 목록]</b>"]
    sorted_pos = sorted(pos.items(), key=lambda x: -abs(x[1]))
    for i, (coin, w) in enumerate(sorted_pos):
        lines.append(f"{i+1}. <code>{coin}</code>: {w:+.4f}")

    chunk = []
    for line in lines:
        if sum(len(l) for l in chunk) + len(line) > 3800:
            send_telegram_message("\n".join(chunk))
            chunk = []
        chunk.append(line)
    if chunk:
        send_telegram_message("\n".join(chunk))


def cmd_balance():
    # 가상 자산 로드
    equity = 0.0
    day_pnl = 0.0
    equity_file = Path("data/runtime/live/paper_equity.jsonl")
    if equity_file.exists():
        eq_lines = [l for l in equity_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        if eq_lines:
            try:
                last_data = json.loads(eq_lines[-1])
                equity = last_data.get("equity", 0.0)
                day_pnl = last_data.get("day_pnl", 0.0)
            except Exception:
                pass

    lines = []
    lines.append("<b>[자산 현황]</b>")
    lines.append(f"• 누적 가상 수익률(PnL): {equity:+.6f}")
    lines.append(f"• 당일 가상 손익: {day_pnl:+.6f}")

    cfg = load_live_state()
    if cfg["mode"] == "real":
        lines.append("\n<b>[실제 거래소 잔고 (바이낸스 데모)]</b>")
        try:
            from src.live.exchange import client
            cli = client.get_client("real")
            resp = cli.rest_api.account()
            data = resp.data() if hasattr(resp, "data") else resp
            assets = data.get("assets", [])
            found = False
            for asset in assets:
                asset_name = asset.get("asset")
                wallet_balance = float(asset.get("walletBalance", 0.0))
                unrealized_pnl = float(asset.get("unrealizedProfit", 0.0))
                if wallet_balance > 0.001:
                    lines.append(f"• <b>{asset_name}</b> 잔고: {wallet_balance:.2f} (미실현 손익: {unrealized_pnl:+.2f})")
                    found = True
            if not found:
                lines.append("• 보유 잔고 없음")
        except Exception as e:
            lines.append(f"• 거래소 잔고 조회 실패: <i>{e}</i>")

    send_telegram_message("\n".join(lines))


def cmd_mode(arg: str):
    cfg = load_live_state()
    arg = arg.strip()
    if not arg:
        send_telegram_message(f"현재 매매 모드: <code>{cfg['mode']}</code>\n• 변경 방법: <code>/모드 실매매</code> 또는 <code>/모드 가상매매</code>")
        return

    if arg in ("실매매", "real"):
        cfg["mode"] = "real"
        save_live_state(cfg)
        send_telegram_message("📢 매매 모드가 <b>실매매 (real)</b>로 전환되었습니다.")
    elif arg in ("가상매매", "paper"):
        cfg["mode"] = "paper"
        save_live_state(cfg)
        send_telegram_message("📢 매매 모드가 <b>가상매매 (paper)</b>로 전환되었습니다.")
    else:
        send_telegram_message("❌ 올바른 아규먼트가 아닙니다.\n• 사용법: <code>/모드 실매매</code> 또는 <code>/모드 가상매매</code>")


def cmd_toggle():
    cfg = load_live_state()
    cfg["enabled"] = not cfg["enabled"]
    save_live_state(cfg)
    status_str = "🟢 작동 시작" if cfg["enabled"] else "🔴 작동 중단"
    send_telegram_message(f"📢 봇 상태가 <b>{status_str}</b> 상태로 변경되었습니다.")


def cmd_run():
    cfg = load_live_state()
    if not cfg["enabled"]:
        send_telegram_message("⚠️ 봇이 중단 상태(disabled)입니다. 먼저 <code>/토글</code>로 활성화해 주세요.")
        return

    send_telegram_message("⚡ 라이브 사이클을 즉시 실행합니다. (데이터 최신화 동반...)")
    try:
        from src.live import handler as H
        res = H.run_cycle(cfg["config_path"], mode=cfg["mode"], refresh=True)
        # H.run_cycle 안에서 send_cycle_report 가 호출되므로 여기선 성공 응답만
        send_telegram_message("✅ 즉시 실행이 완료되었습니다.")
    except Exception as e:
        send_telegram_message(f"❌ 즉시 실행 중 에러 발생: <code>{e}</code>")


def cmd_clear():
    try:
        OR.save_positions({})
        send_telegram_message("🧹 로컬 가상 포지션(positions.json)이 성공적으로 초기화되었습니다.")
    except Exception as e:
        send_telegram_message(f"❌ 초기화 중 에러 발생: <code>{e}</code>")


def cmd_snapshot():
    snap_dir = Path("data/market/universe")
    if not snap_dir.exists():
        send_telegram_message("❌ universe_snapshots 폴더가 존재하지 않습니다.")
        return

    files = sorted(snap_dir.glob("????-??.json"))
    if not files:
        send_telegram_message("❌ 사용 가능한 월 스냅샷 파일이 없습니다.")
        return

    target_file = files[-1]
    try:
        snap_data = json.loads(target_file.read_text(encoding="utf-8"))
        rebalance_date = snap_data.get("rebalance_date", "알 수 없음")
        member_count = snap_data.get("member_count", 0)
        members = snap_data.get("members", [])

        lines = []
        lines.append(f"<b>[월별 유니버스 스냅샷 정보]</b>")
        lines.append(f"• 스냅샷 파일명: <code>{target_file.name}</code>")
        lines.append(f"• 재밸런싱 실행일(마지막 실행일): <code>{rebalance_date}</code>")
        lines.append(f"• 대상 종목 수: <b>{member_count}종목</b>")

        if members:
            lines.append(f"\n<b>[스냅샷 대상 코인 목록 (상위 20개)]</b>")
            lines.append(", ".join(members[:20]) + f" ... 외 {max(0, len(members)-20)}개 종목")

        send_telegram_message("\n".join(lines))
    except Exception as e:
        send_telegram_message(f"❌ 스냅샷 파싱 실패: <code>{e}</code>")


def cmd_risk():
    cfg = load_live_state()
    config_path = cfg.get("config_path", "data/portfolio_4alpha.json")
    p = Path(config_path)
    if not p.exists():
        send_telegram_message(f"❌ 포트폴리오 설정 파일(<code>{config_path}</code>)을 찾을 수 없습니다.")
        return

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        risk_pipeline = data.get("risk_pipeline", [])

        lines = []
        lines.append(f"<b>[리스크 관리 파이프라인 상태]</b>")
        lines.append(f"• 포트폴리오 명: <code>{data.get('name', '알 수 없음')}</code>")

        if not risk_pipeline:
            lines.append("• 비어있음 (적용된 리스크 모듈 없음)")
        else:
            for i, r in enumerate(risk_pipeline):
                enabled = r.get("enabled", True)
                status_icon = "🟢" if enabled else "⚪"
                params_str = ", ".join(f"{k}: {v}" for k, v in r.get("params", {}).items())
                lines.append(f"{status_icon} <b>{i+1}. {r['type']}</b>")
                if params_str:
                    lines.append(f"   └ 파라미터: <code>{params_str}</code>")

        send_telegram_message("\n".join(lines))
    except Exception as e:
        send_telegram_message(f"❌ 리스크 정보 파싱 실패: <code>{e}</code>")


def cmd_help():
    lines = [
        "<b>[Quant 라이브 텔레그램 봇 명령어 도움말]</b>",
        "• <code>/상태</code>: 봇 작동 모드, 작동 여부, 최종 실행 일시, 보유 자산 요약",
        "• <code>/포지션</code>: 현재 로컬 가상 포지션 및 가중치 목록 전체 조회",
        "• <code>/잔고</code>: 가상 누적 수익률(PnL) 및 실매매 시 실제 지갑 잔고 조회",
        "• <code>/모드 &lt;실매매|가상매매&gt;</code>: 실매매(real)와 가상매매(paper) 상태 동적 변경",
        "• <code>/토글</code>: 봇의 자동 거래 로직 임시 활성화/비활성화",
        "• <code>/실행</code>: 당일 라이브 사이클 즉시 강제 수행 (데이터 갱신)",
        "• <code>/스냅샷</code>: 최신 월 스냅샷 정보 및 최종 재밸런싱 실행일 조회",
        "• <code>/리스크</code>: 현재 작동 중인 리스크 오버레이 모듈 상태 요약",
        "• <code>/초기화</code>: 로컬 가상 포지션 초기화",
        "• <code>/도움말</code>: 봇 도움말 보기"
    ]
    send_telegram_message("\n".join(lines))


def handle_message(msg: dict):
    """단일 메시지에 대해 권한 필터를 적용하여 한글 명령어를 분기 처리합니다."""
    chat_id = str(msg.get("chat", {}).get("id", ""))
    if chat_id != ALLOWED_CHAT_ID:
        log.warning(f"허용되지 않은 사용자 차단 (Chat ID: {chat_id})")
        return

    text = msg.get("text", "").strip()
    if not text.startswith("/"):
        return

    parts = text.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/상태":
        cmd_status()
    elif cmd == "/포지션":
        cmd_positions()
    elif cmd == "/잔고":
        cmd_balance()
    elif cmd == "/모드":
        cmd_mode(arg)
    elif cmd == "/토글":
        cmd_toggle()
    elif cmd == "/실행":
        cmd_run()
    elif cmd == "/초기화":
        cmd_clear()
    elif cmd == "/스냅샷":
        cmd_snapshot()
    elif cmd == "/리스크":
        cmd_risk()
    elif cmd in ("/도움말", "/help", "/start"):
        cmd_help()


def start_polling_bot():
    """텔레그램 getUpdates API를 사용한 롱 폴링(Long Polling) 루프를 돌며 대기합니다."""
    if not TOKEN:
        log.error("TELEGRAM_BOT_TOKEN 이 설정되어 있지 않아 봇을 시작할 수 없습니다.")
        return
    log.info("텔레그램 봇 Polling 루프를 대기 시작합니다...")
    offset = None
    while True:
        url = f"https://api.telegram.org/bot{TOKEN}/getUpdates"
        params = {"timeout": 30}
        if offset:
            params["offset"] = offset

        url_with_params = url + "?" + urllib.parse.urlencode(params)
        try:
            req = urllib.request.Request(url_with_params)
            with urllib.request.urlopen(req, timeout=35) as response:
                res = json.loads(response.read().decode("utf-8"))
                if res.get("ok"):
                    for update in res.get("result", []):
                        offset = update["update_id"] + 1
                        message = update.get("message")
                        if message:
                            handle_message(message)
        except urllib.error.URLError as ue:
            # 네트워크 타임아웃/간헐적 끊김은 가볍게 로깅 후 대기
            log.debug("네트워크 타임아웃 또는 연결 대기 중...")
            time.sleep(2)
        except Exception as e:
            log.error(f"Polling 루프 중 에러 발생: {e}")
            time.sleep(5)
