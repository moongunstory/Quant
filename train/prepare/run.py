# train/prepare/run.py

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from ai_binance.config.settings import DEFAULT_START_DATE, DEFAULT_END_DATE
from ai_binance.train.prepare.collection.fetch_binance import fetch_binance_data
from ai_binance.train.prepare.collection.fetch_dune import fetch_dune_data
from config.paths import PROCESSED_DATA_DIR

def main():
    print("▶ 원시 데이터 수집 시작")
    fetch_binance_data(DEFAULT_START_DATE, DEFAULT_END_DATE)
    fetch_dune_data()
    print(f"✅ 원시 데이터 저장 완료 → {PROCESSED_DATA_DIR / 'raw'}")

if __name__ == "__main__":
    main()
