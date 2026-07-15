"""Lambda 진입점 — EventBridge 크론에서 하루 1회 라이브 사이클 실행.

AWS Lambda에서 호출될 때 lambda_handler를 찾음.
로컬 CLI는 cli.py를 사용.

동작 개요 (R2/S3 호환 스토리지 연동)
------------------------------------
Lambda 파일시스템은 /tmp 를 빼면 읽기전용이라, 매매 데이터를 Lambda 바깥의 영속 저장소
(Cloudflare R2 등)에 둔다. 매 실행마다:
  1) 데이터 루트를 /tmp/quant-data 로 잡고(쓰기 가능),
  2) 이미지에 구워둔 strategy(설정/알파)를 /tmp 로 시드,
  3) R2 에서 어제까지의 데이터(원본·패널·유니버스·런타임)를 다운로드,
  4) 매매 사이클 실행(오늘치 증분 수집 → 패널 재빌드 → 목표가중치 → 주문),
  5) 갱신된 데이터·런타임 상태를 R2 로 업로드.

배포 설정 (Lambda 콘솔/CLI 에서 지정할 환경변수):
  - REMOTE_STORE=1
  - S3_ENDPOINT_URL, S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, (S3_REGION=auto)
  - QUANT_DATA_DIR=/tmp/quant-data   (미설정 시 코드가 기본값으로 넣음)
  - TRADING_MODE=live 또는 testnet, 그리고 BINANCE/TELEGRAM 키들
  - (선택) QUANT_RETENTION_DAYS=90 → 오래된 날짜별 런타임 파일 자동정리
"""
import os
import sys
import json
import logging
import shutil
import time

# ── 중요: src 의 어떤 모듈보다 먼저 데이터 루트를 잡아야 한다 ──────────────────
# backtest_settings.SETTINGS 는 import 시점에 QUANT_DATA_DIR 을 읽어 고정되므로,
# src.* 를 import 하기 전에 환경변수를 세팅해야 /tmp 가 반영된다.
os.environ.setdefault("QUANT_DATA_DIR", "/tmp/quant-data")

# Lambda 환경에서도 프로젝트 src/ 를 임포트하기 위해 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.live.handler import run_cycle
from src.live import remote_store as RS

log = logging.getLogger()
log.setLevel(logging.INFO)

# 데이터 최신화(콜드스타트 백필 포함)에 시간을 다 써버리면 목표가중치 계산/주문/텔레메트리/
# 텔레그램 발송이 진행될 시간이 안 남는다. Lambda 남은 실행시간에서 이만큼은 항상 남겨둔다.
POST_REFRESH_SAFETY_MARGIN_SECONDS = 90


def _seed_strategy_into_data_dir():
    """이미지에 구워둔 data/strategy(설정·알파·메타)를 /tmp 데이터 루트로 복사.

    strategy 는 Lambda 에서 바뀌지 않는 읽기전용 자료라 R2 대신 이미지에 싣고, 실행마다
    /tmp 로 시드한다. 이미 있으면 건너뛴다(같은 컨테이너 재사용 시 중복복사 방지)."""
    data_dir = os.environ["QUANT_DATA_DIR"]
    dest = os.path.join(data_dir, "strategy")
    if os.path.isdir(dest):
        return
    task_root = os.environ.get("LAMBDA_TASK_ROOT", os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(task_root, "data", "strategy")
    if os.path.isdir(src):
        shutil.copytree(src, dest)
        log.info("strategy 시드 완료: %s → %s", src, dest)
    else:
        log.warning("이미지에 data/strategy 가 없습니다(%s). 배포에 포함됐는지 확인 필요.", src)


# 웹훅 조회 명령이 '실제 데이터'를 보게 하려면 R2 에서 최소한의 상태 파일만 빠르게 받아온다.
# (전체 prefix 를 훑는 sync_down 은 파일이 많아 느려서 텔레그램이 재시도할 수 있다.)
WEBHOOK_LIGHT_KEYS = [
    "runtime/live/config.json",         # 모드/온오프/설정 경로
    "runtime/live/positions.json",      # 현재 가상 포지션
    "runtime/live/positions_entry.json",# 실시간 손익용 진입가 스냅샷
    "runtime/live/paper_equity.jsonl",  # 누적 가상 자산 곡선
]


def _is_webhook_event(event) -> bool:
    """Function URL(또는 API Gateway) 로 들어온 HTTP 요청인지 판별.
    EventBridge 크론 이벤트는 requestContext.http 가 없다."""
    if not isinstance(event, dict):
        return False
    method = event.get("requestContext", {}).get("http", {}).get("method")
    return method is not None or ("body" in event and "headers" in event)


def _handle_webhook(event, context):
    """텔레그램 웹훅 요청 처리: 비밀토큰 검증 → 경량 R2 동기화 → 명령 처리 → (변경 시) 업로드.

    항상 200 을 돌려준다(텔레그램이 실패로 보고 재시도하는 것을 막기 위해). 인증 실패/무시할
    메시지도 200 + 짧은 바디로 응답하고, 실제 처리 여부만 로그로 남긴다."""
    ok_200 = {"statusCode": 200, "body": "ok"}

    data_dir = os.environ["QUANT_DATA_DIR"]
    os.makedirs(data_dir, exist_ok=True)
    _seed_strategy_into_data_dir()

    # telegram_bot 은 import 시점에 env(토큰/시크릿)를 읽으므로 여기서 늦게 import.
    from src.live import telegram_bot as TB
    from src.live import remote_store as RS

    # 1) 비밀토큰 검증 — 텔레그램이 실어 보낸 헤더와 대조(Function URL 은 헤더 키가 소문자)
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    got_secret = headers.get("x-telegram-bot-api-secret-token")
    if TB.WEBHOOK_SECRET and got_secret != TB.WEBHOOK_SECRET:
        log.warning("웹훅 비밀토큰 불일치 -- 요청 무시")
        return ok_200

    # 2) 본문(텔레그램 update) 파싱
    body = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        import base64
        body = base64.b64decode(body).decode("utf-8")
    try:
        update = json.loads(body)
    except Exception as e:
        log.error("웹훅 본문 파싱 실패: %s", e)
        return ok_200

    message = update.get("message") or update.get("edited_message")
    if not message:
        return ok_200

    cmd, _arg = TB.parse_command(message)
    if not cmd:
        return ok_200

    # 3) 명령이 볼 데이터만 R2 에서 빠르게 내려받는다.
    try:
        RS.sync_down_files(WEBHOOK_LIGHT_KEYS)
        if cmd == "/스냅샷":
            RS.sync_down(("market/universe",))      # 최신 월 스냅샷 필요
        elif cmd in ("/텔레메트리", "/로그파일"):
            RS.sync_down(("runtime",))               # 날짜별 텔레메트리 파일 필요
    except Exception as e:
        log.error("웹훅 R2 다운로드 실패(계속 진행): %s", e, exc_info=True)

    # 4) 명령 처리 (webhook=True: /실행 은 접수 안내만)
    mutated = False
    try:
        mutated = TB.handle_message(message, webhook=True)
    except Exception as e:
        log.error("웹훅 명령 처리 실패: %s", e, exc_info=True)
        try:
            TB.send_telegram_message(f"❌ 명령 처리 중 에러: <code>{e}</code>")
        except Exception:
            pass

    # 5) 상태를 바꾼 명령(/모드·/토글·/초기화)이면 R2 로 다시 올린다(다음 크론이 안 덮어쓰게).
    if mutated:
        try:
            RS.sync_up(("runtime/live",))
        except Exception as e:
            log.error("웹훅 상태 업로드 실패: %s", e, exc_info=True)

    return ok_200


def lambda_handler(event, context):
    """
    AWS Lambda 핸들러 진입점.

    두 종류의 호출을 하나의 함수가 처리한다:
      - EventBridge 크론: 하루 1회 라이브 사이클(데이터 수집→목표→주문).
      - Function URL 웹훅: 텔레그램 명령어에 즉시 응답(_handle_webhook).

    Args:
        event: EventBridge 크론 이벤트(빈 dict) 또는 Function URL HTTP 이벤트.
        context: Lambda 런타임 컨텍스트

    Returns:
        dict: {"statusCode": int, "body": json_string}
    """
    if _is_webhook_event(event):
        return _handle_webhook(event, context)

    try:
        log.info("라이브 사이클 시작 (Lambda)")

        data_dir = os.environ["QUANT_DATA_DIR"]
        os.makedirs(data_dir, exist_ok=True)

        # 1) 이미지에 구운 strategy 를 /tmp 로 시드
        _seed_strategy_into_data_dir()

        # 2) R2 → /tmp 다운로드 (패널·유니버스·런타임). 비활성이면 0개.
        try:
            RS.sync_down()
        except Exception as e:
            log.error("R2 다운로드 실패(진행은 계속 시도): %s", e, exc_info=True)

        config_path = os.path.join(data_dir, "strategy", "portfolio", "config.json")

        # Lambda 는 매매 전용이지만 '데이터 수집'은 직접 한다(refresh=True). R2 에서 내려받은
        # 어제까지의 데이터에 오늘치 증분만 이어붙이므로 콜드스타트 전체 백필은 첫 실행에서만
        # 일어난다(그 초기 데이터는 로컬에서 미리 push 해 회피). 시간초과 안전장치로 deadline 전달.
        collector_deadline = None
        if context is not None and hasattr(context, "get_remaining_time_in_millis"):
            remaining_seconds = context.get_remaining_time_in_millis() / 1000
            budget = max(0, remaining_seconds - POST_REFRESH_SAFETY_MARGIN_SECONDS)
            collector_deadline = time.monotonic() + budget
            log.info("Lambda 남은 시간 %.0f초 중 수집기에 %.0f초 배정(안전마진 %d초)",
                     remaining_seconds, budget, POST_REFRESH_SAFETY_MARGIN_SECONDS)

        # 3) 매매 사이클 실행 (데이터 최신화 + 패널 재빌드 + 목표가중치 + 주문)
        result = run_cycle(
            config_path,
            mode=None,               # config.json 에서 읽음 (기본 "paper")
            refresh=True,            # 오늘치 데이터 증분 수집
            rebuild=True,            # 패널 캐시 재빌드
            collector_deadline=collector_deadline,
        )

        # 4) 보존기간 초과 런타임 파일 정리(선택) 후 런타임 상태를 R2 로 업로드
        try:
            RS.prune_old_runtime()
            RS.sync_up()
        except Exception as e:
            log.error("R2 업로드 실패: %s", e, exc_info=True)

        log.info("라이브 사이클 완료")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "success",
                "date": result["target"]["date"],
                "n_coins": len(result["target"]["weights"]),
                "n_orders": result["orders"].get("n_orders", 0),
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
