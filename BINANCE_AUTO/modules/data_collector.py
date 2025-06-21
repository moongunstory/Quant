import torch
import os

path = os.path.join("data", "models", "ppo_staging", "long_imitation.pt")

if not os.path.exists(path):
    print(f"❌ 파일이 존재하지 않습니다: {path}")
else:
    print(f"📂 모델 파일 로딩 중: {path}")
    state_dict = torch.load(path, map_location='cpu')

    for k, v in state_dict.items():
        if torch.isnan(v).any():
            print(f"❌ NaN detected in → {k}")
    print("✅ 검사 완료")
