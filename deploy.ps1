# ─────────────────────────────────────────────────────────────────────────────
# Quant 라이브 봇 — Lambda 컨테이너 배포 + 자동 검증 (PowerShell 판)
#
# 그동안 "push 는 했는데 Lambda 에 반영이 안 되던" 진짜 이유:
#   ECR 에 push 만 하고  aws lambda update-function-code  를 안 했기 때문.
#   컨테이너 Lambda 는 이미지 다이제스트에 고정돼 있어서 push 만으로는 절대 안 바뀐다.
#
# 사용법:  이 폴더에서  powershell -ExecutionPolicy Bypass -File .\deploy.ps1
#          (또는 그냥)   .\deploy.ps1
# ─────────────────────────────────────────────────────────────────────────────
$ErrorActionPreference = 'Stop'

# ── 설정 (네 값으로 채워둠) ──────────────────────────────────────────────────
$ACCT     = '324471170327'
$REGION   = 'ap-northeast-2'
$REPO     = 'quant-live-cycle'
$REGISTRY = "$ACCT.dkr.ecr.$REGION.amazonaws.com"
# Lambda 함수 이름: 비워두면 ECR 이미지를 쓰는 함수를 자동 탐지.
# 자동탐지 실패 시 여기에 직접 적어줘.  예: $FN = 'quant-live-cycle'
$FN = ''

Set-Location -Path $PSScriptRoot

function Check($msg) { if ($LASTEXITCODE -ne 0) { throw $msg } }

# ── 0) 로컬 소스에 새 명령이 있는지부터 ──────────────────────────────────────
Write-Host "▶ 0/6  로컬 소스 확인..."
if (-not (Select-String -Path 'src/live/telegram_bot.py' -Pattern 'cmd_history' -Quiet)) {
  throw "로컬 telegram_bot.py 에 cmd_history 가 없음. 코드부터 확인해."
}
Write-Host "  로컬 코드 OK (cmd_history 존재)"

# ── 1) 고유 태그 ─────────────────────────────────────────────────────────────
$SHA = (git rev-parse --short HEAD 2>$null); if ($LASTEXITCODE -ne 0 -or -not $SHA) { $SHA = 'nogit' }
$TAG    = "{0}-{1}" -f (Get-Date -Format 'yyyyMMdd-HHmmss'), $SHA
$IMAGE  = "$REGISTRY/$REPO`:$TAG"
$LATEST = "$REGISTRY/$REPO`:latest"
Write-Host "▶ 1/6  태그: $TAG"

# ── 2) ECR 로그인 ────────────────────────────────────────────────────────────
Write-Host "▶ 2/6  ECR 로그인..."
(aws ecr get-login-password --region $REGION) | docker login --username AWS --password-stdin $REGISTRY
Check "ECR 로그인 실패"
Write-Host "  로그인 OK"

# ── 3) 빌드 (Lambda 필수 옵션 2개 유지) ──────────────────────────────────────
#   --platform linux/amd64 : Lambda 는 amd64 만 지원
#   --provenance=false     : OCI 형식이면 Lambda 가 못 읽음
Write-Host "▶ 3/6  빌드..."
docker build --platform linux/amd64 --provenance=false -t $IMAGE -t $LATEST .
Check "docker build 실패"

# ── 3.5) 빌드된 이미지 안에 새 코드가 실제로 들어갔는지 (push 전에) ──────────
Write-Host "  빌드 이미지 내부 검증..."
$CNT = docker run --rm --entrypoint sh $IMAGE -c 'grep -c cmd_history "$LAMBDA_TASK_ROOT/src/live/telegram_bot.py"'
if ($LASTEXITCODE -ne 0 -or [int]$CNT -lt 1) { throw "빌드 이미지에 cmd_history 없음. Dockerfile COPY/빌드 컨텍스트 확인." }
Write-Host "  이미지 내부 OK (cmd_history $CNT 건)"

# ── 4) push ──────────────────────────────────────────────────────────────────
Write-Host "▶ 4/6  push..."
docker push $IMAGE;  Check "push 실패($IMAGE)"
docker push $LATEST; Check "push 실패($LATEST)"
Write-Host "  push OK: $IMAGE"

# ── 5) 함수 찾기 + 실제 배포(update-function-code) ───────────────────────────
if (-not $FN) {
  Write-Host "▶ 5/6  ECR 이미지를 쓰는 Lambda 함수 자동 탐지..."
  $imageFns = aws lambda list-functions --region $REGION --query 'Functions[?PackageType==`Image`].FunctionName' --output text
  foreach ($f in ($imageFns -split '\s+')) {
    if (-not $f) { continue }
    $uri = aws lambda get-function --function-name $f --region $REGION --query 'Code.ImageUri' --output text 2>$null
    if ($uri -like "*$REPO*") { $FN = $f; break }
  }
}
if (-not $FN) { throw "함수를 못 찾음. 스크립트 상단 `$FN = '...' 에 Lambda 함수 이름을 직접 적어줘." }
Write-Host "  대상 함수: $FN"

Write-Host "  update-function-code (← 그동안 빠졌던 진짜 배포 단계)..."
aws lambda update-function-code --function-name $FN --region $REGION --image-uri $IMAGE | Out-Null
Check "update-function-code 실패"
aws lambda wait function-updated --function-name $FN --region $REGION
Check "function-updated 대기 실패"
Write-Host "  배포 반영 완료"

# ── 6) 사후 검증 ─────────────────────────────────────────────────────────────
Write-Host "▶ 6/6  사후 검증..."
$LIVE_URI = aws lambda get-function --function-name $FN --region $REGION --query 'Code.ImageUri' --output text
$MOD      = aws lambda get-function --function-name $FN --region $REGION --query 'Configuration.LastModified' --output text
Write-Host "  현재 Lambda 이미지: $LIVE_URI"
Write-Host "  마지막 수정 시각  : $MOD"
if ($LIVE_URI -eq $IMAGE) {
  Write-Host "✅ 성공 — Lambda 가 방금 올린 이미지($TAG)를 실행 중." -ForegroundColor Green
  Write-Host "   텔레그램에서 /도움말 치면 이제 /이유·/기록 이 보일 거야."
} else {
  Write-Host "⚠ 이미지 URI 가 방금 올린 것과 다름. 위 두 값을 확인해." -ForegroundColor Yellow
}
