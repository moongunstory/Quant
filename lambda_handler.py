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

# Lambda 환경에서도 프로젝트 src/ 를 임포트하기 위해 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.live.handler import run_cycle

log = logging.getLogger()
log.setLevel(logging.INFO)


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

        # run_cycle 실행
        result = run_cycle(
            config_path,
            mode=None,  # 설정 파일에서 읽음 (기본값 "paper")
            refresh=True,  # 데이터 최신화
            rebuild=True  # 패널 캐시 재빌드
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
