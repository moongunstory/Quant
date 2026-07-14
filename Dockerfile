# Lambda용 컨테이너 이미지.
# zip 방식(구 lambda_package)은 압축 해제 후 249MB로 Lambda 한도(250MB)에 거의 걸쳐 있었고,
# 의존성(binance-sdk 등)이 늘어나면 바로 한도를 넘길 위험이 있어서 컨테이너 이미지로 전환.
# 컨테이너 이미지는 10GB까지 허용되어 여유가 훨씬 크다.
FROM public.ecr.aws/lambda/python:3.11

# 의존성 설치 (matplotlib 등 실제로 안 쓰는 건 requirements.txt에서 이미 제외됨)
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt --no-cache-dir

# 코드 복사.
# 주의: src 폴더는 통째로 복사해야 한다 — src/backtest 안의 panel.py/spec.py를
# src/live/target_weights.py가 직접 import해서 쓰기 때문에, 예전 zip 배포 스크립트처럼
# "backtest 폴더 제외"를 하면 매일 크론 실행이 ModuleNotFoundError로 100% 실패한다.
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}
COPY src ${LAMBDA_TASK_ROOT}/src
COPY data/strategy ${LAMBDA_TASK_ROOT}/data/strategy

CMD [ "lambda_handler.lambda_handler" ]
