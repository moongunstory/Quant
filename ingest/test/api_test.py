# test_apis.py
#
# .env 에 들어있는 각 API 키로 간단한 요청을 보내서
# "키가 제대로 작동하는지"만 체크하는 스모크 테스트 스크립트.


import os
from textwrap import shorten
import requests
from dotenv import load_dotenv


def load_env():
    # Correctly locate the .env file in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(project_root, ".env")
    load_dotenv(dotenv_path=env_path)


def log_ok(name: str, msg: str = ""):
    if msg:
        print(f"[OK]   {name}: {msg}")
    else:
        print(f"[OK]   {name}")


def log_skip(name: str, reason: str):
    print(f"[SKIP] {name}: {reason}")


def log_fail(name: str, info: str):
    print(f"[FAIL] {name}: {info}")


def safe_snippet(text: str, length: int = 160) -> str:
    try:
        return shorten(text.replace("\n", " "), width=length, placeholder=" ...")
    except Exception:
        return ""


# -----------------------------
# 1. Market Aggregators / Price
# -----------------------------

def test_coingecko(session, key: str):
    """
    CoinGecko Demo API ping 테스트.
    Demo key → x-cg-demo-api-key
    URL     → https://api.coingecko.com/api/v3/ping
    """
    name = "CoinGecko"
    if not key:
        log_skip(name, "COINGECKO_KEY not set")
        return

    url = "https://api.coingecko.com/api/v3/ping"
    headers = {"x-cg-demo-api-key": key}

    try:
        r = session.get(url, headers=headers, timeout=5)
        if r.ok:
            log_ok(name, f"status={r.status_code}, body={r.json()}")
        else:
            log_fail(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
    except requests.RequestException as e:
        log_fail(name, f"exception={e}")


# -----------------------------
# 2. Macro / Traditional
# -----------------------------

def test_fred(session, key: str):
    name = "FRED"
    if not key:
        log_skip(name, "FRED_API_KEY not set")
        return

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DGS10",
        "api_key": key,
        "file_type": "json",
        "limit": 1,
    }

    try:
        r = session.get(url, params=params, timeout=5)
        if r.ok:
            log_ok(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
        else:
            log_fail(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
    except requests.RequestException as e:
        log_fail(name, f"exception={e}")


def test_finnhub(session, key: str):
    name = "Finnhub"
    if not key:
        log_skip(name, "FINNHUB_API_KEY not set")
        return

    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": "AAPL"} # Use a common stock symbol
    headers = {"X-Finnhub-Token": key}

    try:
        r = session.get(url, params=params, headers=headers, timeout=5)
        if r.ok and r.json().get('c', 0) != 0: # Check for valid data
            log_ok(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
        else:
            log_fail(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
    except requests.RequestException as e:
        log_fail(name, f"exception={e}")


def test_alpha_vantage(session, key: str, key_index: int):
    name = f"Alpha Vantage {key_index}"
    if not key:
        log_skip(name, f"ALPHA_VANTAGE_API_KEY_{key_index} not set")
        return

    url = "https://www.alphavantage.co/query"
    params = {"function": "GLOBAL_QUOTE", "symbol": "SPY", "apikey": key}

    try:
        r = session.get(url, params=params, timeout=10)
        # Alpha Vantage often returns 200 OK with a "Note" about API limits on failure
        if r.ok and "Note" not in r.text and "Information" not in r.text:
            log_ok(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
        else:
            log_fail(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
    except requests.RequestException as e:
        log_fail(name, f"exception={e}")


# -----------------------------
# 3. Sentiment / News
# -----------------------------

def test_coinstats(session, key: str):
    name = "CoinStats"
    if not key:
        log_skip(name, "COINSTATS_PUBLIC_API_KEY not set")
        return

    url = "https://openapi.coinstats.app/public/v1/coins"
    params = {"limit": 1, "currency": "USD"}
    headers = {"X-API-KEY": key}

    try:
        r = session.get(url, params=params, headers=headers, timeout=10)
        if r.ok:
            log_ok(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
        else:
            log_fail(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
    except requests.RequestException as e:
        log_fail(name, f"exception={e}")


def test_newsapi(session, key: str):
    name = "NewsAPI"
    if not key:
        log_skip(name, "NEWS_API_KEY not set")
        return

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "business",
        "language": "en",
        "pageSize": 1,
        "apiKey": key,
    }

    try:
        r = session.get(url, params=params, timeout=5)
        if r.ok:
            log_ok(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
        else:
            log_fail(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
    except requests.RequestException as e:
        log_fail(name, f"exception={e}")


# -----------------------------
# 4. On-chain helpers (optional)
# -----------------------------

def test_etherscan(session, key: str):
    name = "Etherscan"
    if not key:
        log_skip(name, "ETHERSCAN_API_KEY not set")
        return

    url = "https://api.etherscan.io/api" # Corrected endpoint
    params = {
        "module": "stats",
        "action": "ethprice",
        "apikey": key,
    }

    try:
        r = session.get(url, params=params, timeout=5)
        if r.ok:
            log_ok(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
        else:
            log_fail(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
    except requests.RequestException as e:
        log_fail(name, f"exception={e}")


def test_tronscan(session, key: str):
    name = "TronScan"
    if not key:
        log_skip(name, "TRONSCAN_API_KEY not set")
        return

    url = "https://apilist.tronscanapi.com/api/transaction"
    params = {
        "sort": "-timestamp",
        "limit": 1,
    }
    headers = {
        "TRON-PRO-API-KEY": key,
    }

    try:
        r = session.get(url, params=params, headers=headers, timeout=5)
        if r.ok:
            log_ok(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
        else:
            log_fail(name, f"status={r.status_code}, body={safe_snippet(r.text)}")
    except requests.RequestException as e:
        log_fail(name, f"exception={e}")


# -----------------------------
# main
# -----------------------------

def main():
    load_env()

    coingecko_key = os.getenv("COINGECKO_KEY", "")
    fred_key = os.getenv("FRED_API_KEY", "")
    finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    coinstats_key = os.getenv("COINSTATS_PUBLIC_API_KEY", "")
    newsapi_key = os.getenv("NEWS_API_KEY", "")
    etherscan_key = os.getenv("ETHERSCAN_API_KEY", "")
    tronscan_key = os.getenv("TRONSCAN_API_KEY", "")

    session = requests.Session()

    print("=== API Key Smoke Test ===")
    test_coingecko(session, coingecko_key)
    test_fred(session, fred_key)
    test_finnhub(session, finnhub_key)
    
    # Test all three Alpha Vantage keys
    for i in range(1, 4):
        av_key = os.getenv(f"ALPHA_VANTAGE_API_KEY_{i}", "")
        test_alpha_vantage(session, av_key, i)
        
    test_coinstats(session, coinstats_key)
    test_newsapi(session, newsapi_key)
    test_etherscan(session, etherscan_key)
    test_tronscan(session, tronscan_key)


if __name__ == "__main__":
    main()
