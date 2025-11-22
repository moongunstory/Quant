# ingest/orchestrator.py
"""
Data collection orchestrator.

Coordinates all data collection sources:
- Binance (market data)
- Macro (economic indicators)
- News (RSS feeds)
- On-chain metrics
- Derivatives (e.g., DVOL)

Data Retention Policy:
- Re-fetchable data: 540-day sliding window
- Non-re-fetchable data: Permanent accumulation
"""

import os
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from dotenv import load_dotenv

# Import collection functions
from ingest.sources import binance, macro, news, onchain, derivatives
# Note: sentiment module exists but is not currently used in the pipeline


class DataOrchestrator:
    """
    Orchestrates all data collection and storage.
    
    Retention Policy:
    - Permanent: OI (1h), LS Ratio (1h), News (cannot re-fetch full history)
    - Sliding 540d: Everything else (can re-fetch anytime)
    """
    
    # Data retention policies
    SLIDING_WINDOW_DAYS = 540
    
    # Files that should be permanently kept (cannot re-fetch historical data)
    PERMANENT_FILES = {
        'oi_1h.parquet',
        'ls_ratio_top_1h.parquet',
        'news_raw.parquet',
    }
    
    def __init__(self):
        load_dotenv()
        
        self.logger = self._setup_logger()
        self.session = self._create_session()
        self.data_dir = Path("data/raw")
        
        # ✨ sources 모듈 로그 숨기기 (WARNING 이상만 출력)
        logging.getLogger("orchestrator").setLevel(logging.INFO)
        
        # Load API keys
        self.api_keys = {
            'fred': os.getenv('FRED_API_KEY', ''),
            'finnhub': os.getenv('FINNHUB_API_KEY', ''),
            'coingecko': os.getenv('COINGECKO_KEY', ''),
            'coinstats': os.getenv('COINSTATS_PUBLIC_API_KEY', ''),
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup main orchestrator logger."""
        logger = logging.getLogger("orchestrator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - ORCHESTRATOR - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _create_session(self) -> requests.Session:
        """
        Create requests session with automatic retry logic and a browser-like User-Agent.
        
        Retry strategy:
        - 3 retries for 5xx errors and 429
        - Exponential backoff: 1s, 2s, 4s
        """
        session = requests.Session()
        
        # Add a common browser User-Agent to avoid being blocked
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    # ==================== Storage ====================
    
    def should_apply_sliding_window(self, filename: str) -> bool:
        """
        Determine if sliding window should be applied to this file.
        
        Args:
            filename: Parquet filename
            
        Returns:
            False if permanent retention, True if sliding window
        """
        return filename not in self.PERMANENT_FILES
    
    def save_parquet(self, df: pd.DataFrame, module: str, filename: str):
        """Save DataFrame to parquet."""
        filepath = self.data_dir / module / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, index=False, compression='snappy')
    
    def load_parquet(self, module: str, filename: str) -> Optional[pd.DataFrame]:
        """Load parquet file if it exists."""
        filepath = self.data_dir / module / filename
        
        if not filepath.exists():
            return None
        
        try:
            df = pd.read_parquet(filepath)
            # 로그 제거
            return df
        except Exception as e:
            self.logger.error(f"파일 로드 실패 {filepath}: {e}")
            return None
    
    def apply_sliding_window(
        self, 
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """Apply 540-day sliding window."""
        cutoff = pd.Timestamp.now() - timedelta(days=self.SLIDING_WINDOW_DAYS)
        filtered = df[df[timestamp_col] >= cutoff].copy()
        # 로그 제거
        return filtered
        
    def merge_and_dedupe(
        self,
        new_df: pd.DataFrame,
        existing_df: Optional[pd.DataFrame],
        key_columns: list
    ) -> pd.DataFrame:
        """
        Merge new data with existing, remove duplicates, and sort.
        
        Args:
            new_df: Newly collected data
            existing_df: Existing data (or None)
            key_columns: Columns to use for deduplication
            
        Returns:
            Merged and deduplicated DataFrame
        """
        if existing_df is None:
            return new_df
        
        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_columns, keep='last')
        merged = merged.sort_values(key_columns[0]).reset_index(drop=True)
        
        return merged
    
    # ==================== Binance ====================
    
    def collect_binance_data(self, days: int):
        """Collect all Binance data (1h OHLCV / OI / LS + 8h funding)."""
        self.logger.info("📈 바이낸스 수집 중...")
        
        start_dt = pd.Timestamp.now() - timedelta(days=days)
        end_dt = pd.Timestamp.now()
        
        # OI, LS Ratio는 30일 제한
        start_dt_30d = pd.Timestamp.now() - timedelta(days=30)
        
        success_count = 0
        total_count = 0
        
        # 현재 binance.TIMEFRAMES = ['1h'] 이므로 1h만 수집
        for tf in binance.TIMEFRAMES:
            total_count += 4  # OHLCV Futures, Spot, OI, LS
            
            try:
                # OHLCV Futures
                df = binance.collect_ohlcv('futures', tf, start_dt, end_dt, self.session, self.logger)
                if not df.empty:
                    filename = f'ohlcv_futures_{tf}.parquet'
                    existing = self.load_parquet('binance', filename)
                    df = self.merge_and_dedupe(df, existing, ['timestamp'])
                    if self.should_apply_sliding_window(filename):
                        df = self.apply_sliding_window(df)
                    self.save_parquet(df, 'binance', filename)
                    success_count += 1
                    self.logger.info(f"  ✅ OHLCV Futures {tf}: {len(df)}행")
                
                # OHLCV Spot
                df = binance.collect_ohlcv('spot', tf, start_dt, end_dt, self.session, self.logger)
                if not df.empty:
                    filename = f'ohlcv_spot_{tf}.parquet'
                    existing = self.load_parquet('binance', filename)
                    df = self.merge_and_dedupe(df, existing, ['timestamp'])
                    if self.should_apply_sliding_window(filename):
                        df = self.apply_sliding_window(df)
                    self.save_parquet(df, 'binance', filename)
                    success_count += 1
                    self.logger.info(f"  ✅ OHLCV Spot {tf}: {len(df)}행")
                
                # Open Interest (30일 제한, 1h만 영구보관)
                df = binance.collect_open_interest(tf, start_dt_30d, end_dt, self.session, self.logger)
                if not df.empty:
                    filename = f'oi_{tf}.parquet'
                    existing = self.load_parquet('binance', filename)
                    df = self.merge_and_dedupe(df, existing, ['timestamp'])
                    # OI 1h는 PERMANENT_FILES에 포함 → 슬라이딩 윈도우 미적용
                    if self.should_apply_sliding_window(filename):
                        df = self.apply_sliding_window(df)
                    self.save_parquet(df, 'binance', filename)
                    success_count += 1
                    self.logger.info(f"  ✅ OI {tf}: {len(df)}행{' [영구보관]' if filename in self.PERMANENT_FILES else ''}")
                
                # Long/Short Ratio (30일 제한, 1h만 영구보관)
                df = binance.collect_long_short_ratio(tf, 'top', start_dt_30d, end_dt, self.session, self.logger)
                if not df.empty:
                    filename = f'ls_ratio_top_{tf}.parquet'
                    existing = self.load_parquet('binance', filename)
                    df = self.merge_and_dedupe(df, existing, ['timestamp'])
                    # LS 1h는 PERMANENT_FILES에 포함 → 슬라이딩 윈도우 미적용
                    if self.should_apply_sliding_window(filename):
                        df = self.apply_sliding_window(df)
                    self.save_parquet(df, 'binance', filename)
                    success_count += 1
                    self.logger.info(f"  ✅ LS Ratio {tf}: {len(df)}행{' [영구보관]' if filename in self.PERMANENT_FILES else ''}")
                    
            except Exception as e:
                self.logger.error(f"  ❌ {tf} 실패: {str(e)[:50]}")
        
        # Funding Rate
        total_count += 1
        try:
            df = binance.collect_funding_rate(start_dt, end_dt, self.session, self.logger)
            if not df.empty:
                filename = 'funding_rate.parquet'
                existing = self.load_parquet('binance', filename)
                df = self.merge_and_dedupe(df, existing, ['timestamp'])
                if self.should_apply_sliding_window(filename):
                    df = self.apply_sliding_window(df)
                self.save_parquet(df, 'binance', filename)
                success_count += 1
                self.logger.info(f"  ✅ Funding Rate: {len(df)}행")
        except Exception as e:
            self.logger.error(f"  ❌ Funding Rate 실패: {str(e)[:50]}")
        
        self.logger.info(f"✅ 바이낸스 완료 ({success_count}/{total_count})\n")
    
    # ==================== Macro ====================
    
    def collect_macro_data(self, days: int):
        """Collect all macro data."""
        self.logger.info("📊 매크로 수집 중...")
        
        start_dt = pd.Timestamp.now() - timedelta(days=days)
        end_dt = pd.Timestamp.now()
        
        success_count = 0
        
        # FRED
        try:
            fred_data = macro.collect_all_fred(
                start_dt, end_dt,
                self.api_keys['fred'],
                self.session, self.logger
            )
            
            for series_id, df in fred_data.items():
                if not df.empty:
                    filename = f"fred_{series_id.lower()}.parquet"
                    existing = self.load_parquet('macro', filename)
                    df = self.merge_and_dedupe(df, existing, ['date'])
                    if self.should_apply_sliding_window(filename):
                        df = self.apply_sliding_window(df, 'date')
                    self.save_parquet(df, 'macro', filename)
                    success_count += 1
            
            self.logger.info(f"  ✅ FRED: {len(fred_data)}개 시리즈")
        except Exception as e:
            self.logger.error(f"  ❌ FRED 실패: {str(e)[:50]}")
        
        # Yahoo Finance (for Indices/Assets, replacing Finnhub)
        try:
            self.logger.info("\n--- Yahoo Finance Indices/Assets ---")
            yahoo_success_count = 0
            # macro.FINNHUB_SYMBOLS를 그대로 사용하여 Yahoo Finance에서 동일한 심볼들을 가져옵니다.
            for symbol in macro.FINNHUB_SYMBOLS.keys():
                df = macro.collect_yahoo_finance(symbol, start_dt, end_dt, self.session, self.logger)
                if not df.empty:
                    clean_symbol = symbol.replace('^', '').replace('-', '').replace('.', '_').lower()
                    filename = f"yahoo_{clean_symbol}.parquet"
                    existing = self.load_parquet('macro', filename)
                    df = self.merge_and_dedupe(df, existing, ['date'])
                    
                    if self.should_apply_sliding_window(filename):
                        df = self.apply_sliding_window(df, 'date')
                    
                    self.save_parquet(df, 'macro', filename)
                    success_count += 1
                    yahoo_success_count += 1
                time.sleep(2)  # 요청 사이에 2초 지연 추가
            
            self.logger.info(f"  ✅ Yahoo Finance: {yahoo_success_count}/{len(macro.FINNHUB_SYMBOLS)}개 심볼")
        except Exception as e:
            self.logger.error(f"  ❌ Yahoo Finance 실패: {str(e)[:50]}")
        
        self.logger.info(f"✅ 매크로 완료 (총 {success_count}개)\n")
    
    # ==================== Sentiment ====================
    # 현재 버전에서는 센티먼트 모듈을 사용하지 않음.
    # collect_sentiment_data는 남겨두되, initial_collection/smart_update에서 호출하지 않는다.
    
    def collect_sentiment_data(self, days: int):
        """Collect all sentiment data. (현재 파이프라인에서는 미사용)"""
        self.logger.info("💭 센티먼트 수집 중... (현재는 파이프라인에서 호출하지 않음)")
        
        start_dt = pd.Timestamp.now() - timedelta(days=days)
        end_dt = pd.Timestamp.now()
        
        success_count = 0
        
        # Fear & Greed
        try:
            df = sentiment.collect_fear_greed(start_dt, end_dt, self.session, self.logger)
            if not df.empty:
                filename = 'fear_greed.parquet'
                existing = self.load_parquet('sentiment', filename)
                df = self.merge_and_dedupe(df, existing, ['timestamp'])
                if self.should_apply_sliding_window(filename):
                    df = self.apply_sliding_window(df)
                self.save_parquet(df, 'sentiment', filename)
                success_count += 1
                self.logger.info(f"  ✅ Fear & Greed: {len(df)}행")
        except Exception as e:
            self.logger.error(f"  ❌ Fear & Greed 실패: {str(e)[:50]}")
        
        # CoinGecko
        try:
            df = sentiment.collect_coingecko_sentiment(
                self.api_keys['coingecko'],
                self.session, self.logger
            )
            if not df.empty:
                filename = 'coingecko_sentiment.parquet'
                existing = self.load_parquet('sentiment', filename)
                df = self.merge_and_dedupe(df, existing, ['timestamp'])
                self.save_parquet(df, 'sentiment', filename)
                success_count += 1
                self.logger.info(f"  ✅ CoinGecko: {len(df)}행 [영구보관 후보]")
        except Exception as e:
            self.logger.error(f"  ❌ CoinGecko 실패: {str(e)[:50]}")
        
        self.logger.info(f"✅ 센티먼트 완료 ({success_count}/2)\n")
    
    # ==================== News ====================
    
    def collect_news_data(self, days: int):
        """Collect all news data."""
        self.logger.info("📰 뉴스 수집 중...")
        
        start_dt = pd.Timestamp.now() - timedelta(days=days)
        end_dt = pd.Timestamp.now()
        
        try:
            df = news.collect_all_news(start_dt, end_dt, self.logger)
            if not df.empty:
                filename = 'news_raw.parquet'
                existing = self.load_parquet('news', filename)
                df = self.merge_and_dedupe(df, existing, ['url'])
                self.save_parquet(df, 'news', filename)
                self.logger.info(f"  ✅ 뉴스: {len(df)}개 기사 [영구보관]")
        except Exception as e:
            self.logger.error(f"  ❌ 뉴스 실패: {str(e)[:50]}")
        
        self.logger.info(f"✅ 뉴스 완료\n")
        
    # ==================== On-chain ====================
    
    def collect_onchain_data(self, days: int):
        """Collect all on-chain data from Blockchain.com."""
        self.logger.info("🔗 온체인 데이터 수집 중...")
        
        success_count = 0
        
        for metric, description in onchain.BLOCKCHAIN_COM_METRICS.items():
            try:
                df = onchain.collect_blockchain_com_metric(metric, self.session, self.logger)
                if not df.empty:
                    # The collected data is daily, so apply sliding window based on 'days'
                    df = self.apply_sliding_window(df, timestamp_col='date')
                    
                    filename = f"blockchain_com_{metric}.parquet"
                    existing = self.load_parquet('onchain', filename)
                    df = self.merge_and_dedupe(df, existing, ['date'])
                    
                    self.save_parquet(df, 'onchain', filename)
                    success_count += 1
            except Exception as e:
                self.logger.error(f"  ❌ {description} 수집 실패: {str(e)[:50]}")
        
        self.logger.info(f"✅ 온체인 완료 ({success_count}/{len(onchain.BLOCKCHAIN_COM_METRICS)})\n")

    # ==================== Derivatives ====================

    def collect_derivatives_data(self, days: int):
        """Collect all derivatives data from Deribit."""
        self.logger.info("📈 파생상품 데이터 수집 중...")
        
        success_count = 0
        
        # Currently only collects DVOL for BTC
        for currency, description in derivatives.DERIBIT_METRICS.items():
            try:
                df = derivatives.collect_deribit_dvol(currency, self.session, self.logger)
                if not df.empty:
                    df = self.apply_sliding_window(df, timestamp_col='date')
                    
                    filename = f"deribit_{currency.lower()}_dvol.parquet"
                    existing = self.load_parquet('derivatives', filename)
                    df = self.merge_and_dedupe(df, existing, ['date'])
                    
                    self.save_parquet(df, 'derivatives', filename)
                    success_count += 1
            except Exception as e:
                self.logger.error(f"  ❌ {description} 수집 실패: {str(e)[:50]}")
        
        self.logger.info(f"✅ 파생상품 완료 ({success_count}/{len(derivatives.DERIBIT_METRICS)})\n")

    # ==================== High-Level ====================
    
    def initial_collection(self, days: int = 540, targets: Optional[list[str]] = None):
        """
        Initial data collection. Can be targeted to specific modules.
        
        Args:
            days: Number of days to collect.
            targets: List of modules to collect (e.g., ['binance', 'macro']). 
                     If None, collects all.
        """
        # 센티먼트는 현재 파이프라인에서 제외
        all_targets = ['binance', 'macro', 'news', 'onchain', 'derivatives']
        
        if targets is None:
            targets_to_run = all_targets
            self.logger.info("=" * 60)
            self.logger.info(f"📥 전체 초기 수집 시작 (최대 {days}일)")
            self.logger.info("=" * 60 + "\n")
        else:
            targets_to_run = targets
            self.logger.info("=" * 60)
            self.logger.info(f"📥 부분 재수집 시작: {', '.join(targets_to_run)}")
            self.logger.info("=" * 60 + "\n")

        results = {}
        
        if 'binance' in targets_to_run:
            try:
                self.collect_binance_data(days)
                results['바이낸스'] = '✅'
            except Exception as e:
                results['바이낸스'] = '❌'
                self.logger.error(f"바이낸스 전체 실패: {e}", exc_info=True)
        
        if 'macro' in targets_to_run:
            try:
                self.collect_macro_data(days)
                results['매크로'] = '✅'
            except Exception as e:
                results['매크로'] = '❌'
                self.logger.error(f"매크로 전체 실패: {e}", exc_info=True)

        # 센티먼트 블록은 비활성화
        # if 'sentiment' in targets_to_run:
        #     try:
        #         self.collect_sentiment_data(days)
        #         results['센티먼트'] = '✅'
        #     except Exception as e:
        #         results['센티먼트'] = '❌'
        #         self.logger.error(f"센티먼트 전체 실패: {e}", exc_info=True)

        if 'news' in targets_to_run:
            try:
                self.collect_news_data(days)
                results['뉴스'] = '✅'
            except Exception as e:
                results['뉴스'] = '❌'
                self.logger.error(f"뉴스 전체 실패: {e}", exc_info=True)
        
        if 'onchain' in targets_to_run:
            try:
                self.collect_onchain_data(days)
                results['온체인'] = '✅'
            except Exception as e:
                results['온체인'] = '❌'
                self.logger.error(f"온체인 전체 실패: {e}", exc_info=True)

        if 'derivatives' in targets_to_run:
            try:
                self.collect_derivatives_data(days)
                results['파생상품'] = '✅'
            except Exception as e:
                results['파생상품'] = '❌'
                self.logger.error(f"파생상품 전체 실패: {e}", exc_info=True)

        # Summary
        self.logger.info("=" * 60)
        self.logger.info("📊 수집 완료")
        self.logger.info("=" * 60)
        if results:
            status_line = " | ".join([f"{k}: {v}" for k, v in results.items()])
            self.logger.info(status_line)
        else:
            self.logger.info("수집 대상으로 지정된 모듈이 없습니다.")
        self.logger.info("=" * 60)
        
        return all('✅' in status for status in results.values())
    
    def _is_stale(self, file_path: Path, staleness_hours: int) -> bool:
        """Check if a file is stale based on its modification time."""
        if not file_path.exists():
            return True  # File doesn't exist, so it's "stale"
        
        mtime = file_path.stat().st_mtime
        last_modified_dt = datetime.fromtimestamp(mtime)
        
        return (datetime.now() - last_modified_dt) > timedelta(hours=staleness_hours)

    def smart_update(self):
        """
        Intelligently updates modules based on their individual staleness.
        """
        self.logger.info("=" * 80)
        self.logger.info("🤖 스마트 업데이트 시작")
        self.logger.info(f"⏰ 시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80 + "\n")
        
        results = {}
        updated_modules = []
        
        # Define update policies (representative file and staleness threshold in hours)
        policies = {
            '바이낸스': ('binance/ohlcv_futures_1h.parquet', 1),
            '뉴스': ('news/news_raw.parquet', 2),
            # '센티먼트': ('sentiment/fear_greed.parquet', 4),  # 현재 비활성화
            '매크로': ('macro/fred_dgs10.parquet', 24),
            '온체인': ('onchain/blockchain_com_n-transactions.parquet', 24),
            '파생상품': ('derivatives/deribit_btc_dvol.parquet', 24),
        }
        
        # Check and update each module
        for module_name, (file_to_check, staleness_hours) in policies.items():
            file_path = self.data_dir / file_to_check
            
            if self._is_stale(file_path, staleness_hours):
                self.logger.info(f"🔄 '{module_name}' 모듈이 오래되어 갱신을 시작합니다 (기준: {staleness_hours}시간).")
                updated_modules.append(module_name)
                try:
                    if module_name == '바이낸스':
                        self.collect_binance_data(days=2)
                    elif module_name == '뉴스':
                        self.collect_news_data(days=2)
                    # elif module_name == '센티먼트':
                    #     self.collect_sentiment_data(days=2)
                    elif module_name == '매크로':
                        self.collect_macro_data(days=2)
                    elif module_name == '온체인':
                        self.collect_onchain_data(days=2)
                    elif module_name == '파생상품':
                        self.collect_derivatives_data(days=2)
                    results[module_name] = '✅ 성공'
                except Exception as e:
                    results[module_name] = f'❌ 실패: {e}'
                    self.logger.error(f"{module_name} 업데이트 실패: {e}", exc_info=True)
            else:
                self.logger.info(f"👍 '{module_name}' 모듈은 최신 상태입니다. 건너뜁니다.")
                results[module_name] = '✅ 최신'

        # Summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 스마트 업데이트 결과 요약")
        self.logger.info("=" * 80)
        if not updated_modules:
            self.logger.info("   모든 데이터가 최신 상태입니다. 업데이트된 모듈이 없습니다.")
        else:
            for module, status in results.items():
                self.logger.info(f"   {module}: {status}")
        self.logger.info(f"\n⏰ 완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        
        return all('성공' in status or '최신' in status for status in results.values())
