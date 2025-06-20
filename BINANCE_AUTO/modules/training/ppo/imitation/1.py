import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import joblib

def check_numpy_array(array, name="Numpy Array"):
    print(f"[🔍 {name}] shape: {array.shape}")
    print(f"  - NaN 개수: {np.isnan(array).sum()}")
    print(f"  - Inf 개수: {np.isinf(array).sum()}")
    print(f"  - 최소값: {np.min(array):.6f}, 최대값: {np.max(array):.6f}")
    print(f"  - 평균: {np.mean(array):.6f}, 표준편차: {np.std(array):.6f}")
    print("-" * 60)

def check_tensor(tensor, name="Torch Tensor"):
    array = tensor.detach().cpu().numpy()
    check_numpy_array(array, name)

def check_dataframe(df: pd.DataFrame, name="DataFrame"):
    print(f"[🔍 {name}] shape: {df.shape}")
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        print(f"  - NaN 포함 컬럼: {nan_cols}")
    else:
        print("  - NaN 포함 컬럼 없음")
    print(f"  - 전체 NaN 수: {df.isna().sum().sum()}")
    print("-" * 60)

def load_and_check(path: str):
    ext = os.path.splitext(path)[-1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        check_dataframe(df, name=os.path.basename(path))
    elif ext == ".pkl":
        obj = joblib.load(path)
        if isinstance(obj, pd.DataFrame):
            check_dataframe(obj, name=os.path.basename(path))
        elif isinstance(obj, np.ndarray):
            check_numpy_array(obj, name=os.path.basename(path))
        elif isinstance(obj, dict):
            print(f"[📦 {path}]: dict 객체")
            for k, v in obj.items():
                if isinstance(v, np.ndarray):
                    check_numpy_array(v, name=f"{k}")
                elif isinstance(v, torch.Tensor):
                    check_tensor(v, name=f"{k}")
        else:
            print(f"[⚠️] 지원하지 않는 pkl 타입: {type(obj)}")
    else:
        print(f"[❌] 지원하지 않는 파일 확장자: {ext}")

def main():
    parser = argparse.ArgumentParser(description="🩺 데이터 NaN 진단 도구")
    parser.add_argument("--path", type=str, required=True, help="검사할 .csv 또는 .pkl 파일 경로")
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        print(f"[❌] 파일이 존재하지 않음: {path}")
        sys.exit(1)

    print(f"🔎 진단 시작: {path}")
    load_and_check(path)

if __name__ == "__main__":
    main()
