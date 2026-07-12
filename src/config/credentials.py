"""
credentials.py

.env에서 읽어오는 비밀값/실행 모드. API 키가 대부분이라 'settings'보다
내용을 그대로 드러내는 이름으로 뒀다.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DEFAULT_TRADING_MODE = os.getenv("TRADING_MODE").lower()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

TESTNET_API_KEY = os.getenv("TESTNET_API_KEY")
TESTNET_SECRET_KEY = os.getenv("TESTNET_SECRET_KEY")
