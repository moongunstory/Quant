# ingest/sources/news.py
"""
News data collection functions.

Pure functions for collecting:
- Coindesk RSS
- Cointelegraph RSS

All functions return pd.DataFrame without side effects.
"""

from datetime import datetime
import pandas as pd
import time
import feedparser

# ==================== Constants ====================

RSS_FEEDS = {
    'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'cointelegraph': 'https://cointelegraph.com/rss',
}

# ==================== RSS Collection ====================

def collect_rss_feed(
    source: str,
    feed_url: str,
    start_dt: datetime,
    end_dt: datetime,
    logger
) -> pd.DataFrame:
    """
    Collect news from an RSS feed.
    
    Args:
        source: Source name (e.g., 'coindesk')
        feed_url: RSS feed URL
        start_dt: Filter articles after this datetime
        end_dt: Filter articles before this datetime
        logger: Logger instance
        
    Returns:
        DataFrame with columns: timestamp, title, description, source, url
    """
    logger.info(f"Collecting RSS from {source}: {feed_url}")
    
    try:
        # Parse RSS feed
        feed = feedparser.parse(feed_url)
        
        if not feed.entries:
            logger.warning(f"No entries found in {source} RSS feed")
            return pd.DataFrame()
        
        articles = []
        
        for entry in feed.entries:
            # Extract published date
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6])
            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                pub_date = datetime(*entry.updated_parsed[:6])
            else:
                # No date available, use current time
                pub_date = datetime.now()
            
            # Apply date filter
            if pub_date < start_dt or pub_date > end_dt:
                continue
            
            # Filter for Bitcoin-related content
            title = entry.get('title', '')
            description = entry.get('summary', entry.get('description', ''))
            
            # Simple keyword filter
            content = (title + ' ' + description).lower()
            if not any(kw in content for kw in ['bitcoin', 'btc', 'crypto', 'sec', 'etf']):
                continue
            
            article = {
                'timestamp': pub_date,
                'title': title,
                'description': description,
                'source': source,
                'url': entry.get('link', ''),
            }
            
            articles.append(article)
        
        df = pd.DataFrame(articles)
        
        if df.empty:
            logger.info(f"No relevant articles from {source}")
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Collected {len(df)} articles from {source}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to parse RSS feed from {source}: {e}")
        return pd.DataFrame()


def collect_all_news(
    start_dt: datetime,
    end_dt: datetime,
    logger
) -> pd.DataFrame:
    """
    Collect news from all RSS sources.
    
    Args:
        start_dt: Start datetime
        end_dt: End datetime
        logger: Logger instance
        
    Returns:
        Combined DataFrame from all sources
    """
    logger.info(f"Collecting news: {start_dt.date()} to {end_dt.date()}")
    
    all_articles = []
    
    for source, feed_url in RSS_FEEDS.items():
        df = collect_rss_feed(source, feed_url, start_dt, end_dt, logger)
        
        if not df.empty:
            all_articles.append(df)
        
        time.sleep(2)  # Be nice to RSS servers
    
    if not all_articles:
        logger.warning("No articles collected from any source")
        return pd.DataFrame()
    
    # Combine all sources
    df = pd.concat(all_articles, ignore_index=True)
    
    # Remove duplicates (same title from different sources)
    df = df.drop_duplicates(subset=['title'], keep='first')
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Total collected: {len(df)} unique articles")
    
    return df
