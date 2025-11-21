# process/modules/news.py

from __future__ import annotations
import pandas as pd

from .utils import ensure_sorted_datetime

# ==================== Sentiment (VADER + fallback) ====================

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    _VADER_AVAILABLE = True
    _analyzer = SentimentIntensityAnalyzer()
except Exception:
    _VADER_AVAILABLE = False
    _analyzer = None


FALLBACK_POS_WORDS = [
    "surge", "surges", "rally", "rallies", "bullish", "gain", "gains",
    "soar", "soars", "spike", "spikes", "record high", "all-time high",
    "approved", "approval", "supports", "supportive", "positive",
]

FALLBACK_NEG_WORDS = [
    "plunge", "plunges", "drop", "drops", "crash", "crashes", "bearish",
    "loss", "losses", "falls", "sinks", "dump", "dumps",
    "ban", "banned", "lawsuit", "sue", "sues", "charged", "charges",
    "hack", "hacked", "exploit", "breach", "leak", "scam", "fraud",
    "crackdown", "shutdown",
]


BTC_KEYWORDS = ["bitcoin", "btc"]

GOOD_EVENT_KEYWORDS = [
    "etf", "spot etf", "approval", "approved", "launch", "launched",
    "listing", "listed", "upgrade", "partnership", "integration",
    "adoption", "record inflow", "inflow",
]

BAD_EVENT_KEYWORDS = [
    "hack", "hacked", "exploit", "breach", "bug", "vulnerability",
    "rug pull", "rugpull", "scam", "fraud",
    "lawsuit", "sue", "sues", "charged", "charges", "fine", "fined",
    "ban", "banned", "crackdown", "halt", "suspension",
]


TOPIC_KEYWORDS = {
    "etf_reg": [
        "etf", "sec", "cftc", "spot etf", "futures etf",
        "application", "filing", "approval", "approved", "rejected",
        "surveillance sharing",
    ],
    "hack_risk": [
        "hack", "hacked", "exploit", "breach", "attack", "attacker",
        "vulnerability", "bug", "leak", "rug pull", "rugpull",
        "scam", "fraud", "phishing",
    ],
    "regulation": [
        "ban", "banned", "illegal", "unregistered",
        "crackdown", "enforcement", "lawsuit", "sue", "sues",
        "charged", "charges", "fine", "fined", "sanction",
        "regulator", "regulation", "regulatory",
    ],
    "adoption": [
        "adoption", "adopt", "accept", "accepts", "accepting",
        "payment", "payments", "merchant", "merchants",
        "integration", "integrates", "partner", "partnership",
    ],
    "macro": [
        "inflation", "cpi", "ppi", "interest rate", "interest rates",
        "rate hike", "rate cut", "fed", "federal reserve",
        "recession", "treasury", "bond", "yields",
        "dollar", "usd", "liquidity",
    ],
    "exchange": [
        "binance", "coinbase", "kraken", "bybit", "okx", "deribit",
        "exchange", "trading platform", "spot market", "derivatives",
    ],
    "mining": [
        "mining", "miner", "miners", "hashrate", "hash rate",
        "difficulty", "halving", "block reward",
    ],
}


def _compute_sentiment(text: str) -> dict:
    text = text or ""
    if _VADER_AVAILABLE:
        return _analyzer.polarity_scores(text)

    # fallback: 아주 단순한 룰 기반
    t = text.lower()
    pos = sum(t.count(w) for w in FALLBACK_POS_WORDS)
    neg = sum(t.count(w) for w in FALLBACK_NEG_WORDS)
    total = pos + neg

    if total == 0:
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    compound = (pos - neg) / total
    pos_ratio = pos / total
    neg_ratio = neg / total
    neu_ratio = max(0.0, 1.0 - pos_ratio - neg_ratio)

    return {
        "compound": compound,
        "pos": pos_ratio,
        "neg": neg_ratio,
        "neu": neu_ratio,
    }


def _event_score(text_lower: str) -> int:
    score = 0
    for kw in GOOD_EVENT_KEYWORDS:
        if kw in text_lower:
            score += 1
    for kw in BAD_EVENT_KEYWORDS:
        if kw in text_lower:
            score -= 1
    return score


def build_news_features(df_news: pd.DataFrame) -> pd.DataFrame:
    """
    뉴스 raw DataFrame -> 일단위 피처.

    df_news: 최소 ['timestamp', 'title', 'description'] 포함.
    """
    if df_news is None or df_news.empty:
        raise RuntimeError("Empty news dataframe given to build_news_features")

    df = ensure_sorted_datetime(df_news, "timestamp")
    df["date"] = df["timestamp"].dt.normalize()

    title = df["title"].fillna("")
    desc = df["description"].fillna("")
    text = (title + " " + desc).astype(str)
    text_lower = text.str.lower()

    # BTC 관련 여부
    df["is_btc"] = text_lower.apply(
        lambda s: any(kw in s for kw in BTC_KEYWORDS)
    )

    # 도메인 이벤트 점수 (ETF/해킹/규제 등 키워드 기반)
    df["news_event_score"] = text_lower.apply(_event_score)

    # 토픽 태깅
    for topic, kws in TOPIC_KEYWORDS.items():
        col = f"topic_{topic}"
        df[col] = text_lower.apply(
            lambda s, _kws=kws: any(kw in s for kw in _kws)
        )

    # 감성 점수 (VADER 또는 fallback)
    sent_df = pd.DataFrame(list(text.apply(_compute_sentiment)))
    df["sent_compound"] = sent_df["compound"]
    df["sent_pos"] = sent_df["pos"]
    df["sent_neg"] = sent_df["neg"]
    df["sent_neu"] = sent_df["neu"]

    # -------- 일 단위 집계 --------
    grp = df.groupby("date")

    agg = pd.DataFrame(index=grp.size().index)
    agg["news_count"] = grp.size()
    agg["news_btc_count"] = grp["is_btc"].sum()
    agg["news_btc_ratio"] = agg["news_btc_count"] / agg["news_count"]

    # 감성 집계
    agg["news_sent_compound_mean"] = grp["sent_compound"].mean()
    agg["news_sent_compound_std"] = grp["sent_compound"].std()
    agg["news_sent_pos_mean"] = grp["sent_pos"].mean()
    agg["news_sent_neg_mean"] = grp["sent_neg"].mean()

    # 도메인 이벤트 점수
    agg["news_event_score_sum"] = grp["news_event_score"].sum()
    agg["news_event_score_mean"] = grp["news_event_score"].mean()

    # 토픽별 count / ratio
    for topic in TOPIC_KEYWORDS.keys():
        col = f"topic_{topic}"
        cnt_col = f"news_topic_{topic}_count"
        ratio_col = f"news_topic_{topic}_ratio"

        agg[cnt_col] = grp[col].sum()
        agg[ratio_col] = agg[cnt_col] / agg["news_count"]

    agg = agg.reset_index().sort_values("date").reset_index(drop=True)
    return agg
