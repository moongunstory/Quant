# ai_binance/live/reporting.py
import os
import csv
from datetime import datetime

def update_trade_log(log_path: str, trade_info: dict):
    """
    매매 내역(run_log.csv)을 업데이트합니다.
    파일이 없으면 헤더와 함께 새로 생성합니다.
    """
    file_exists = os.path.exists(log_path)
    
    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Type', 'Position', 'Price', 'Profit', 'Holding Duration'])
        
        writer.writerow([
            trade_info.get('timestamp'),
            trade_info.get('type'),
            trade_info.get('position'),
            trade_info.get('price'),
            trade_info.get('profit', ''),
            trade_info.get('duration', '')
        ])

def generate_report(report_path: str, report_data: dict, is_new_session: bool):
    """
    사람이 읽기 쉬운 형식의 리포트(trading_report.md)를 생성하거나 덮어씁니다.
    """
    content = f"""
==================== TRADING REPORT ====================
매매 시작 시각 : {report_data['session_start_time']}
최근 갱신 시각 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

[ 현재 상태 ]
- 초기 자산: ${report_data['initial_capital']:,.2f}
- 총자산 (현재): ${report_data['total_equity']:,.2f}
- 미실현 손익: ${report_data['unrealized_pnl_amount']:,.2f} ({report_data['unrealized_pnl_percent']:.2f}%)
- 현재 포지션: {report_data['position']}

[ 매매 요약 (이번 세션) ]
- 전체 매매 횟수: {report_data['total_trades']}회
- 전체 승률: {report_data['win_rate']:.1f}%

- 롱 포지션: {report_data['long_trades']}회 (승률: {report_data['long_win_rate']:.1f}%)
- 숏 포지션: {report_data['short_trades']}회 (승률: {report_data['short_win_rate']:.1f}%)
- 홀딩 횟수: {report_data['hold_trades']}회

========================================================
"""
    
    mode = 'a' if is_new_session else 'w'
    if is_new_session:
        # 새 세션일 경우, 이전 내용과 구분을 위해 줄바꿈을 추가합니다.
        content = "\n\n" + content

    with open(report_path, mode, encoding='utf-8') as f:
        f.write(content)
