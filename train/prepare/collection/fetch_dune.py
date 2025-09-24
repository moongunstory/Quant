# train/prepare/fetch_dune.py

import os
import time
import pandas as pd
from dune_client.client import DuneClient
from dotenv import load_dotenv
from dune_client.query import QueryBase

from ai_binance.config.paths import get_dune_path
from ai_binance.config.settings import DUNE_QUERY_IDS, DUNE_QUERY_BATCH_SIZE

load_dotenv()

DUNE_API_KEY = os.getenv("DUNE_API_KEY")
if not DUNE_API_KEY:
    raise EnvironmentError("환경변수 DUNE_API_KEY가 설정되지 않았습니다.")

def fetch_query(query_id: int, refresh: bool = True) -> pd.DataFrame:
    dune = DuneClient(api_key=DUNE_API_KEY)
    query = QueryBase(name=f"Dune Query {query_id}", query_id=query_id)
    return dune.run_query_dataframe(query) if refresh else dune.get_latest_result_dataframe(query_id)

def save_query_result(query_id: int, df: pd.DataFrame):
    # 모든 결과를 같은 이름으로 저장 → 나중에 process_dune_data()에서 concat
    path = get_dune_path("ETHUSDT", f"query_{query_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[DUNE] Saved query {query_id} → {path}")

def fetch_dune_data(refresh: bool = True):
    for i in range(0, len(DUNE_QUERY_IDS), DUNE_QUERY_BATCH_SIZE):
        batch = DUNE_QUERY_IDS[i:i + DUNE_QUERY_BATCH_SIZE]

        for query_id in batch:
            try:
                print(f"[DUNE] Fetching query {query_id}...")
                df = fetch_query(query_id, refresh=refresh)
                save_query_result(query_id, df)
                time.sleep(1)
            except Exception as e:
                print(f"[DUNE] Error fetching query {query_id}: {e}")
