#!/usr/bin/env python3
"""
독립 진단 모듈 - 주문 실행 전 실패 가능성 사전 탐지
실행: python diagnosis.py

요구사항:
- 내부 모듈 import 금지
- 모든 함수/상수 파일 내 정의
- ETHUSDT long 기준 진단
"""

import os
import math
from decimal import Decimal, ROUND_DOWN
from binance.client import Client
from dotenv import load_dotenv

# 상수 정의
SYMBOL = "ETHUSDT"
DIRECTION = "long"
TARGET_LEVERAGE = 5
MIN_BALANCE_USDT = 5.0
RISK_PERCENT = 0.95  # 잔고의 95%만 사용

def load_api_credentials():
    """API 키 로드 (.env 우선, 없으면 하드코딩)"""
    load_dotenv()
    
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET_KEY')
    
    # .env가 없는 경우 하드코딩 (실제 운영시에는 수정 필요)
    if not api_key or not api_secret:
        print("⚠️ .env 파일에서 API 키를 찾을 수 없습니다. 하드코딩된 값 사용")
        api_key = "YOUR_API_KEY_HERE"
        api_secret = "YOUR_API_SECRET_HERE"
    
    return api_key, api_secret

def test_binance_connection(client):
    """Binance API 연결 테스트"""
    try:
        # 서버 시간 확인으로 연결 테스트
        server_time = client.get_server_time()
        account_info = client.futures_account()
        print(f"[✔] Binance 연결 성공")
        return True
    except Exception as e:
        print(f"[✖] Binance 연결 실패: {e}")
        return False

def get_usdt_balance(client):
    """USDT 잔고 확인"""
    try:
        account = client.futures_account()
        for asset in account['assets']:
            if asset['asset'] == 'USDT':
                balance = float(asset['crossWalletBalance'])
                print(f"[✔] USDT 잔고: {balance} USDT")
                return balance
        
        print(f"[✖] USDT 잔고 정보를 찾을 수 없습니다")
        return 0.0
    except Exception as e:
        print(f"[✖] 잔고 조회 실패: {e}")
        return 0.0

def check_symbol_status(client, symbol):
    """심볼 거래 가능 상태 확인"""
    try:
        exchange_info = client.futures_exchange_info()
        for symbol_info in exchange_info['symbols']:
            if symbol_info['symbol'] == symbol:
                status = symbol_info['status']
                if status == 'TRADING':
                    print(f"[✔] 마켓 상태: {status}")
                    return True, symbol_info
                else:
                    print(f"[✖] 마켓 상태: {status} (거래 불가)")
                    return False, symbol_info
        
        print(f"[✖] {symbol} 정보를 찾을 수 없습니다")
        return False, None
    except Exception as e:
        print(f"[✖] 심볼 상태 확인 실패: {e}")
        return False, None

def get_current_leverage(client, symbol):
    """현재 레버리지 확인"""
    try:
        positions = client.futures_position_information(symbol=symbol)
        if positions:
            leverage = int(positions[0]['leverage'])
            if leverage == TARGET_LEVERAGE:
                print(f"[✔] 현재 레버리지: {leverage}x (설정값과 일치)")
                return True, leverage
            else:
                print(f"[⚠] 현재 레버리지: {leverage}x (설정값: {TARGET_LEVERAGE}x와 다름)")
                return False, leverage
        else:
            print(f"[✖] 포지션 정보를 찾을 수 없습니다")
            return False, 0
    except Exception as e:
        print(f"[✖] 레버리지 확인 실패: {e}")
        return False, 0

def get_current_price(client, symbol):
    """현재 가격 조회"""
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        return price
    except Exception as e:
        print(f"[✖] 가격 조회 실패: {e}")
        return 0.0

def extract_trading_rules(symbol_info):
    """거래 규칙 추출 (최소 수량, 수량 단위, 최소 주문 금액)"""
    rules = {
        'minQty': 0.0,
        'maxQty': 0.0,
        'stepSize': 0.0,
        'minNotional': 0.0
    }
    
    try:
        for filter_info in symbol_info['filters']:
            if filter_info['filterType'] == 'LOT_SIZE':
                rules['minQty'] = float(filter_info['minQty'])
                rules['maxQty'] = float(filter_info['maxQty'])
                rules['stepSize'] = float(filter_info['stepSize'])
            elif filter_info['filterType'] == 'MIN_NOTIONAL':
                rules['minNotional'] = float(filter_info['minNotional'])
            elif filter_info['filterType'] == 'MARKET_LOT_SIZE':
                # 마켓 주문의 경우 별도 제한이 있을 수 있음
                market_min = float(filter_info['minQty'])
                if market_min > rules['minQty']:
                    rules['minQty'] = market_min
        
        return rules
    except Exception as e:
        print(f"[✖] 거래 규칙 추출 실패: {e}")
        return rules

def calculate_order_quantity(balance, price, leverage, step_size, risk_percent=RISK_PERCENT):
    """주문 수량 계산"""
    try:
        # 사용 가능한 마진 계산
        available_margin = balance * risk_percent
        
        # 레버리지를 고려한 최대 포지션 가치
        max_position_value = available_margin * leverage
        
        # 수량 계산
        raw_quantity = max_position_value / price
        
        # stepSize에 맞춰 내림 처리
        if step_size > 0:
            # Decimal을 사용해 정확한 계산
            decimal_qty = Decimal(str(raw_quantity))
            decimal_step = Decimal(str(step_size))
            
            # stepSize로 나눈 후 내림하고 다시 곱함
            steps = decimal_qty / decimal_step
            floored_steps = steps.quantize(Decimal('1'), rounding=ROUND_DOWN)
            final_quantity = float(floored_steps * decimal_step)
        else:
            final_quantity = raw_quantity
        
        return final_quantity, max_position_value
    except Exception as e:
        print(f"[✖] 수량 계산 실패: {e}")
        return 0.0, 0.0

def validate_order_conditions(quantity, price, trading_rules):
    """주문 조건 검증"""
    issues = []
    
    # 최소 수량 확인
    if quantity < trading_rules['minQty']:
        issues.append(f"수량 부족 (최소: {trading_rules['minQty']}, 계산: {quantity})")
    
    # 최대 수량 확인
    if quantity > trading_rules['maxQty']:
        issues.append(f"수량 초과 (최대: {trading_rules['maxQty']}, 계산: {quantity})")
    
    # 최소 주문 금액 확인
    order_value = quantity * price
    if order_value < trading_rules['minNotional']:
        issues.append(f"주문 금액 부족 (최소: {trading_rules['minNotional']} USDT, 계산: {order_value:.2f} USDT)")
    
    return issues

def set_leverage_if_needed(client, symbol, target_leverage, current_leverage):
    """필요시 레버리지 설정"""
    if current_leverage != target_leverage:
        try:
            print(f"[⚙] 레버리지를 {current_leverage}x에서 {target_leverage}x로 변경 중...")
            client.futures_change_leverage(symbol=symbol, leverage=target_leverage)
            print(f"[✔] 레버리지 변경 완료: {target_leverage}x")
            return True
        except Exception as e:
            if "No need to change margin type" in str(e) or "-4046" in str(e):
                print(f"[✔] 레버리지가 이미 {target_leverage}x로 설정되어 있습니다")
                return True
            else:
                print(f"[✖] 레버리지 변경 실패: {e}")
                return False
    return True

def main():
    """메인 진단 함수"""
    print(f"🚀 주문 진단 시작 ({SYMBOL} {DIRECTION.upper()})")
    print("-" * 50)
    
    all_checks_passed = True
    
    # 1. API 연결
    try:
        api_key, api_secret = load_api_credentials()
        client = Client(api_key, api_secret)
        
        if not test_binance_connection(client):
            return False
    except Exception as e:
        print(f"[✖] API 클라이언트 초기화 실패: {e}")
        return False
    
    # 2. 잔고 확인
    usdt_balance = get_usdt_balance(client)
    if usdt_balance < MIN_BALANCE_USDT:
        print(f"[✖] 잔고 부족 (최소 {MIN_BALANCE_USDT} USDT 필요)")
        all_checks_passed = False
    
    # 3. 심볼 상태 확인
    is_trading, symbol_info = check_symbol_status(client, SYMBOL)
    if not is_trading:
        all_checks_passed = False
        return False
    
    # 4. 현재 가격 조회
    current_price = get_current_price(client, SYMBOL)
    if current_price <= 0:
        all_checks_passed = False
        return False
    
    # 5. 레버리지 확인 및 설정
    leverage_ok, current_leverage = get_current_leverage(client, SYMBOL)
    if not leverage_ok:
        if not set_leverage_if_needed(client, SYMBOL, TARGET_LEVERAGE, current_leverage):
            all_checks_passed = False
    
    # 6. 거래 규칙 추출
    trading_rules = extract_trading_rules(symbol_info)
    print(f"[ℹ] 거래 규칙 - 최소수량: {trading_rules['minQty']}, 수량단위: {trading_rules['stepSize']}, 최소금액: {trading_rules['minNotional']} USDT")
    
    # 7. 수량 계산
    quantity, position_value = calculate_order_quantity(
        usdt_balance, current_price, TARGET_LEVERAGE, trading_rules['stepSize']
    )
    
    if quantity <= 0:
        print(f"[✖] 계산된 수량이 0 이하입니다")
        all_checks_passed = False
    else:
        order_value = quantity * current_price
        print(f"[✔] 계산 수량: {quantity} {SYMBOL.replace('USDT', '')}, 주문금액: {order_value:.2f} USDT")
    
    # 8. 주문 조건 검증
    if quantity > 0:
        validation_issues = validate_order_conditions(quantity, current_price, trading_rules)
        if validation_issues:
            print(f"[✖] 주문 조건 검증 실패:")
            for issue in validation_issues:
                print(f"    - {issue}")
            all_checks_passed = False
        else:
            print(f"[✔] 최소 주문 조건 충족 (minQty, minNotional)")
    
    # 9. 최종 결과
    print("-" * 50)
    if all_checks_passed:
        print("✅ 모든 조건 이상 없음. 주문 가능.")
        
        # 추가 정보 출력
        print(f"\n📊 주문 예상 정보:")
        print(f"   심볼: {SYMBOL}")
        print(f"   방향: {DIRECTION.upper()}")
        print(f"   현재가: ${current_price:.2f}")
        print(f"   레버리지: {TARGET_LEVERAGE}x")
        print(f"   사용 잔고: {usdt_balance * RISK_PERCENT:.2f} USDT ({RISK_PERCENT*100}%)")
        print(f"   예상 수량: {quantity}")
        print(f"   예상 주문 가치: ${quantity * current_price:.2f}")
        
    else:
        print("❌ 일부 조건 미충족. 주문 전 문제 해결 필요.")
    
    return all_checks_passed

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n💥 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()