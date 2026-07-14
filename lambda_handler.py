"""Lambda 진입점 — EventBridge 크론에서 하루 1회 라이브 사이클 실행.

AWS Lambda에서 호출될 때 lambda_handler를 찾음.
로컬 CLI는 cli.py를 사용.

배포 설정 (AWS Console):
  - Handler: lambda_handler.lambda_handler
  - Runtime: Python 3.11+
"""
import os
import sys
import json
import logging
import time

# Lambda 환경에서도 프로젝트 src/ 를 임포트하기 위해 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.live.handler import run_cycle

log = logging.getLogger()
log.setLevel(logging.INFO)

# 데이터 최신화(콜드스타트 백필 포함)에 시간을 다 써버리면 목표가중치 계산/주문/텔레메트리/
# 텔레그램 발송이 진행될 시간이 안 남는다. Lambda 남은 실행시간에서 이만큼은 항상 남겨둔다.
POST_REFRESH_SAFETY_MARGIN_SECONDS = 90


def lambda_handler(event, context):
    """
    AWS Lambda 핸들러 진입점.

    Args:
        event: EventBridge 크론 이벤트 (일반적으로 빈 dict)
        context: Lambda 런타임 컨텍스트

    Returns:
        dict: {"statusCode": int, "body": json_string}
    """
    try:
        log.info("라이브 사이클 시작 (Lambda)")

        # 설정 경로
        config_path = "data/strategy/portfolio/config.json"

        # 데이터가 없어서 live_refresh가 콜드스타트 전체 백필(유니버스 재구성 +
        # 심볼별 전체 히스토리 수집)을 자동으로 돌리는 경우, 이 함수 실행시간(최대 15분)
        # 안에 다 못 끝날 수 있다. context.get_remaining_time_in_millis()로 실제 남은
        # 시간을 계산해 collector_deadline으로 넘기면, 시간이 다 되기 전에 수집기가
        # 스스로 깨끗하게 멈추고(작업 중이던 심볼/월 단위까지는 저장됨) 목표가중치
        # 계산 이후 단계를 위한 여유시간을 남긴다. 다음 크론 실행이 멈춘 지점부터
        # 자연스럽게 이어서 계속 수집한다 — 데이터 없을 때 며칠에 걸쳐 자동으로
        # 채워지는 구조.
        collector_deadline = None
        if context is not None and hasattr(context, "get_remaining_time_in_millis"):
            remaining_seconds = context.get_remaining_time_in_millis() / 1000
            budget = max(0, remaining_seconds - POST_REFRESH_SAFETY_MARGIN_SECONDS)
            collector_deadline = time.monotonic() + budget
            log.info("Lambda 남은 시간 %.0f초 중 수집기에 %.0f초 배정(안전마진 %d초 확보)",
                      remaining_seconds, budget, POST_REFRESH_SAFETY_MARGIN_SECONDS)

        # run_cycle 실행
        result = run_cycle(
            config_path,
            mode=None,  # 설정 파일에서 읽음 (기본값 "paper")
            refresh=True,  # 데이터 최신화
            rebuild=True,  # 패널 캐시 재빌드
            collector_deadline=collector_deadline
        )

        log.info("라이브 사이클 완료")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "success",
                "date": result["target"]["date"],
                "n_coins": len(result["target"]["weights"]),
                "n_orders": result["orders"].get("n_orders", 0)
            })
        }

    except Exception as e:
        log.error("라이브 사이클 실패: %s", e, exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "message": "error",
                "error": str(e)
            })
        }
