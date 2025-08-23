# ai_binance/live/reporting.py
"""
Reporting utilities (rev)
- 기존 인터페이스 유지: update_trade_log(), generate_report()
- 안전성 강화: 경로 자동 생성, 빈 파일 헤더 보장, UTC 표기
- 누락 필드 견고 처리(get + 기본값), 윈도우 개행(newline='') 대응
"""

import os
import csv
from datetime import datetime, timezone


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def update_trade_log(log_path: str, trade_info: dict) -> None:
    """
    매매 내역(run_log.csv)을 업데이트.
    - 파일/디렉터리 자동 생성
    - 빈 파일이면 헤더 작성
    기대 키:
        timestamp, type, position, price, profit(옵션), duration(옵션)
    """
    _ensure_dir(log_path)
    file_exists = os.path.exists(log_path)
    need_header = (not file_exists) or (os.path.getsize(log_path) == 0)

    row = [
        trade_info.get("timestamp", ""),
        trade_info.get("type", ""),
        trade_info.get("position", ""),
        trade_info.get("price", ""),
        trade_info.get("profit", ""),
        trade_info.get("duration", ""),
    ]

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["Timestamp", "Type", "Position", "Price", "Profit", "Holding Duration"])
        writer.writerow(row)


def generate_report(report_path: str, report_data: dict, is_new_session: bool) -> None:
    """
    사람이 읽기 쉬운 리포트(trading_report.md)를 생성/갱신.
    - is_new_session=True: 새로운 세션 블록을 append
    - False: 같은 세션 최신 상태로 overwrite
    기대 키(없으면 기본값):
        session_start_time, initial_capital, position, total_equity,
        unrealized_pnl_amount, unrealized_pnl_percent,
        total_trades, win_rate, long_trades, long_win_rate,
        short_trades, short_win_rate, hold_trades
    """
    _ensure_dir(report_path)

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    g = lambda k, dv=0.0: report_data.get(k, dv)
    pos = report_data.get("position", "STANDBY")

    content = (
        "\n\n" if is_new_session else ""
    ) + f"""==================== TRADING REPORT ====================
매매 시작 시각 : {report_data.get('session_start_time', '')}
최근 갱신 시각 : {now_utc}

[ 현재 상태 ]
- 초기 자산: ${float(g('initial_capital')):,.2f}
- 총자산 (현재): ${float(g('total_equity')):,.2f}
- 미실현 손익: ${float(g('unrealized_pnl_amount')):,.2f} ({float(g('unrealized_pnl_percent')):.2f}%)
- 현재 포지션: {pos}

[ 매매 요약 (이번 세션) ]
- 전체 매매 횟수: {int(report_data.get('total_trades', 0))}회
- 전체 승률: {float(g('win_rate')):.1f}%

- 롱 포지션: {int(report_data.get('long_trades', 0))}회 (승률: {float(g('long_win_rate')):.1f}%)
- 숏 포지션: {int(report_data.get('short_trades', 0))}회 (승률: {float(g('short_win_rate')):.1f}%)
- 홀딩 횟수: {int(report_data.get('hold_trades', 0))}회

========================================================
"""

    mode = "a" if is_new_session else "w"
    with open(report_path, mode, encoding="utf-8") as f:
        f.write(content)
