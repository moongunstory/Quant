#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Quant 라이브 봇 — Lambda 컨테이너 배포 + 자동 검증 스크립트
#
# 그동안 "push 는 했는데 Lambda 에 반영이 안 되던" 진짜 이유:
#   ECR 에 push 만 하고 `aws lambda update-function-code` 를 안 했기 때문.
#   컨테이너 Lambda 는 이미지 다이제스트에 고정돼 있어서, push 만으로는 절대
#   새 코드로 안 바뀐다. 반드시 update-function-code 로 "새 이미지 봐" 라고 알려줘야 함.
#
# 이 스크립트가 대신 지켜주는 것:
#   1) 매번 고유 태그(git sha + 시각)로 빌드 → :latest 재사용 다이제스트 캐싱 함정 제거
#   2) push 후 update-function-code 를 자동 실행 (빠지지 않게)
#   3) 배포 전/후로 이미지 안에 새 명령(cmd_history/cmd_why)이 실제로 들어있는지 검증
#
# 사용법:  이 폴더에서  bash deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 설정 (네 값으로 이미 채워둠) ──────────────────────────────────────────────
ACCT="324471170327"
REGION="ap-northeast-2"
REPO="quant-live-cycle"
REGISTRY="${ACCT}.dkr.ecr.${REGION}.amazonaws.com"
# Lambda 함수 이름: 비워두면 ECR 이미지를 쓰는 함수를 자동으로 찾음.
# 자동탐지가 실패하면 여기에 직접 이름을 적어줘.  예: FN="quant-live-cycle"
FN="${FN:-}"

cd "$(dirname "$0")"

# ── 0) 사전 점검: 로컬 소스에 새 명령이 있는지부터 (없으면 빌드해봐야 소용없음) ──
echo "▶ 0/6  로컬 소스 확인..."
if ! grep -q 'cmd_history' src/live/telegram_bot.py; then
  echo "✖ 로컬 telegram_bot.py 에 cmd_history 가 없음. 코드부터 확인해."; exit 1
fi
echo "  로컬 코드 OK (cmd_history 존재)"

# ── 1) 고유 태그 생성 ────────────────────────────────────────────────────────
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
TAG="$(date +%Y%m%d-%H%M%S)-${SHA}"
IMAGE="${REGISTRY}/${REPO}:${TAG}"
LATEST="${REGISTRY}/${REPO}:latest"
echo "▶ 1/6  태그: ${TAG}"

# ── 2) ECR 로그인 ────────────────────────────────────────────────────────────
echo "▶ 2/6  ECR 로그인..."
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
echo "  로그인 OK"

# ── 3) 빌드 (Lambda 필수 옵션 2개 유지) ──────────────────────────────────────
#   --platform linux/amd64 : Lambda 는 amd64 만 지원
#   --provenance=false     : OCI 형식이면 Lambda 가 못 읽음
echo "▶ 3/6  빌드..."
docker build --platform linux/amd64 --provenance=false -t "$IMAGE" -t "$LATEST" .

# ── 3.5) 빌드된 이미지 안에 새 코드가 실제로 들어갔는지 검증 (push 전에) ──────
echo "  빌드 이미지 내부 검증..."
CNT="$(docker run --rm --entrypoint sh "$IMAGE" -c \
       'grep -c cmd_history "$LAMBDA_TASK_ROOT/src/live/telegram_bot.py"' || echo 0)"
if [ "${CNT:-0}" -lt 1 ]; then
  echo "✖ 빌드된 이미지에 cmd_history 가 없음. Dockerfile COPY 나 빌드 컨텍스트 확인."; exit 1
fi
echo "  이미지 내부 OK (cmd_history ${CNT}건)"

# ── 4) push (고유 태그 + latest 둘 다) ───────────────────────────────────────
echo "▶ 4/6  push..."
docker push "$IMAGE"
docker push "$LATEST"
echo "  push OK: ${IMAGE}"

# ── 5) Lambda 함수 찾기 + 실제 배포(update-function-code) ────────────────────
if [ -z "$FN" ]; then
  echo "▶ 5/6  ECR 이미지를 쓰는 Lambda 함수 자동 탐지..."
  for f in $(aws lambda list-functions --region "$REGION" \
             --query 'Functions[?PackageType==`Image`].FunctionName' --output text); do
    uri="$(aws lambda get-function --function-name "$f" --region "$REGION" \
           --query 'Code.ImageUri' --output text 2>/dev/null || true)"
    case "$uri" in *"${REPO}"*) FN="$f"; break;; esac
  done
fi
if [ -z "$FN" ]; then
  echo "✖ 함수를 못 찾음. 스크립트 상단 FN=\"...\" 에 Lambda 함수 이름을 직접 적어줘."; exit 1
fi
echo "  대상 함수: ${FN}"

echo "  update-function-code (← 그동안 빠졌던 진짜 배포 단계)..."
aws lambda update-function-code --function-name "$FN" --region "$REGION" \
  --image-uri "$IMAGE" >/dev/null
aws lambda wait function-updated --function-name "$FN" --region "$REGION"
echo "  배포 반영 완료"

# ── 6) 배포 사후 검증: Lambda 가 실제로 새 이미지를 물었는지 ──────────────────
echo "▶ 6/6  사후 검증..."
LIVE_URI="$(aws lambda get-function --function-name "$FN" --region "$REGION" \
            --query 'Code.ImageUri' --output text)"
MOD="$(aws lambda get-function --function-name "$FN" --region "$REGION" \
       --query 'Configuration.LastModified' --output text)"
echo "  현재 Lambda 이미지: ${LIVE_URI}"
echo "  마지막 수정 시각  : ${MOD}"
if [ "$LIVE_URI" = "$IMAGE" ]; then
  echo "✅ 성공 — Lambda 가 방금 올린 이미지(${TAG})를 실행 중."
  echo "   텔레그램에서 /도움말 치면 이제 /이유·/기록 이 보일 거야."
else
  echo "⚠ 이미지 URI 가 방금 올린 것과 다름. 위 두 값을 확인해."
fi
