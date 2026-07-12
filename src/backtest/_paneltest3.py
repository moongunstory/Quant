"""panel — Quant 원자료(심볼별 parquet)를 "date×coin" 표로 바꾼다.

이 파일이 coin 과 Quant 의 가장 큰 차이다. coin 은 자체 ingest/core.panel 을 쓰지만,
Quant 는 src/collector 가 심볼별로 시간단위 parquet 를 저장한다. 여기서 그걸 읽어
일 단위로 리샘플하고, date(행)×coin(열) 패널로 합친다.

핵심 2가지:
  1. 일 단위 리샘플 — 전략은 하루 1회 리밸런싱(일봉). 시간봉/5분봉을 일봉으로 집계.
  2. 시점정확(point-in-time) 유니버스 — universe_snapshots/YYYY-MM.json(그 시점의 top-100
     멤버)로 date×coin 불리언 마스크를 만든다. 그날 실제로 top-100 이었던 코인만 거래 →
     생존편향/미래참조 없음. 이게 Quant 의 존재 이유(라이브 top-100 제약과 백테스트를 일치).

패널 계약(coin 과 동일): index=UTC 자정 타임스탬프, columns=심볼, 빈칸=NaN(미상장/상폐).

FIELD 목록(알파 수식에서 이 이름으로 참조):
  close, open, high, low, volume, quote_volume, trades, taker_buy_ratio  (klines)
  funding_rate                                                            (fundingRate)
  premium                                                                 (premiumIndexKlines)
  open_interest, oi_value, toptrader_ratio, long_short_ratio, taker_ls_ratio (metrics)
"""
from __future__ import annotations

import logging

import pandas as pd

from src.config.backtest_settings import SETTINGS

log = logging.getLogger("quant.panel")

# field 이름 -> (데이터셋 폴더, 원본 시간컬럼, 값컬럼, 일집계 방식)
FIELD_SPECS = {
    "close":         ("klines", "open_time", "close", "last"),
    "open":          ("klines", "open_time", "open", "first"),
    "high":          ("klines", "open_time", "high", "max"),
    "low":           ("klines", "open_time", "low", "min"),
    "volume":        ("klines", "open_time", "volume", "sum"),
    "quote_volume":  ("klines", "open_time", "quote_volume", "sum"),
    "trades":        ("klines", "open_time", "trades", "sum"),
    "taker_buy_volume": ("klines", "open_time", "taker_buy_volume", "sum"),
    "funding_rate":  ("fundingRate", "calc_time", "last_funding_rate", "sum"),
    "premium":       ("premiumIndexKlines", "open_time", "close", "last"),
    "oi_value":      ("metrics", "create_time", "sum_open_interest_value", "last"),
    "open_interest": ("metrics", "create_time", "sum_open_interest", "last"),
    "toptrader_ratio": ("metrics", "create_time", "sum_toptrader_long_short_ratio", "mean"),
    "long_short_ratio": ("metrics", "create_time", "count_long_short_ratio", "mean"),
    "taker_ls_ratio": ("metrics", "create_time", "sum_taker_long_short_vol_ratio", "mean"),
}

# 다른 field 로부터 파생되는 것(원본 컬럼이 아님). build_panel 에서 특별 처리.
DERIVED = {
    "taker_buy_ratio": ("taker_buy_volume", "volume"),  # 매수체결 비중
}


def _symbols_for(dataset: str):
    d = SETTINGS.processed_dir / dataset
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


def _series_at(dataset, time_col, value_col, agg, symbol, offset):
    """한 심볼의 parquet 를 읽어 offset(예 '1D','8h','1h') 시리즈로. 못 읽으면 None."""
    path = SETTINGS.processed_dir / dataset / f"{symbol}.parquet"
    try:
        df = pd.read_parquet(path, columns=[time_col, value_col])
    except Exception:
        return None  # 수집 중이거나 손상된 파일은 조용히 건너뜀
    if df.empty:
        return None
    t = pd.to_datetime(df[time_col], utc=True)
    s = pd.Series(df[value_col].to_numpy(), index=t).sort_index()
    # 중복 타임스탬프 제거(metrics 에 중복 존재) 후 지정 주기로 집계
    s = s[~s.index.duplicated(keep="last")]
    out = s.resample(offset).agg(agg)
    out.name = symbol
    return out


def _daily_series(dataset, time_col, value_col, agg, symbol):
    """하위호환: 일단위(1D) 시리즈."""
    return _series_at(dataset, time_col, value_col, agg, symbol, "1D")


def build_panel_at(field: str, bar: str = "1d") -> pd.DataFrame:
    """한 field 의 date×coin 패널을 bar 주기로 만든다(Phase 3).
    bar='1d' 면 기존 build_panel 과 동일(자정 UTC 정규화)."""
    from src.backtest.timegrid import bar_to_offset
    offset = bar_to_offset(bar)

    if field in DERIVED:
        num_f, den_f = DERIVED[field]
        num = build_panel_at(num_f, bar)
        den = build_panel_at(den_f, bar).replace(0.0, pd.NA)
        return (num / den).astype(float)

    if field not in FIELD_SPECS:
        raise KeyError(f"알 수 없는 field {field!r}; 사용가능: {sorted(FIELD_SPECS)}")

    dataset, time_col, value_col, agg = FIELD_SPECS[field]
    cols = {}
    for sym in _symbols_for(dataset):
        s = _series_at(dataset, time_col, value_col, agg, sym, offset)
        if s is not None and s.notna().any():
            cols[sym] = s
    if not cols:
        raise RuntimeError(f"field {field!r}: 읽을 수 있는 심볼이 없음 ({dataset})")

    panel = pd.DataFrame(cols).sort_index()
    if bar == "1d":
        panel.index = panel.index.normalize()  # 자정 UTC 정규화(하위호환)
    return panel


def build_panel(field: str) -> pd.DataFrame:
    """한 field 의 일단위(1d) date×coin 패널. Phase 3 이전과 동일."""
    return build_panel_at(field, "1d")


def build_universe_mask(index, columns) -> pd.DataFrame:
    """월별 top-100 스냅샷 -> date×coin 불리언 마스크(시점정확).

    각 날짜의 유효 멤버 = 그 날짜 이전(포함) 가장 최근 스냅샷의 members.
    월 1회 리밸런싱이므로 다음 스냅샷 전까지 그 멤버십을 유지한다.
    """
    import json

    snap_dir = SETTINGS.universe_snapshot_dir
    snaps = {}
    for p in sorted(snap_dir.glob("[0-9]*.json")):
        if p.stem.endswith("_diff"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        date = pd.Timestamp(d["rebalance_date"], tz="UTC").normalize()
        snaps[date] = set(d.get("members", []))
    if not snaps:
        # 스냅샷이 없으면 전부 True(유니버스 필터 없음)
        log.warning("universe_snapshots 없음 -> 유니버스 필터 미적용")
        return pd.DataFrame(1.0, index=index, columns=columns)

    snap_dates = sorted(snaps)
    mask = pd.DataFrame(False, index=index, columns=columns)
    for day in index:
        # day 이전(포함) 최근 스냅샷 찾기
        active = None
        for sd in snap_dates:
            if sd <= day:
                active = sd
            else:
                break
        if active is None:
            continue
        members = snaps[active] & set(columns)
        if members:
            mask.loc[day, list(members)] = True
    # float(1.0/0.0)로 반환: 엔진에서 reindex 할 때 bool 다운캐스팅 경고를 피한다.
    return mask.astype(float)


# --------------------------- 캐시 --------------------------- #

def _read_cache(path):
    """캐시 parquet 를 읽는다. 손상/잘린 파일이면 경고 후 삭제하고 None 반환.

    쓰다 만(중단된) parquet 는 푸터 magic bytes 가 없어 read 시 예외가 난다.
    그런 파일은 재빌드가 정답이므로 지우고 None 을 돌려, 호출부가 다시 만들게 한다."""
    try:
        return pd.read_parquet(path)
    except Exception as e:
        log.warning("캐시 손상 감지 %s (%s) → 삭제 후 재빌드", path.name, e)
        try:
            path.unlink()
        except OSError:
            pass
        return None


def panel_path(field):
    return SETTINGS.panel_dir / f"{field}.parquet"


def save_panel(field, panel):
    SETTINGS.panel_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path(field))


def load_panel(field, rebuild=False):
    """캐시가 있으면 로드, 없거나 rebuild=True 면 새로 만들고 저장."""
    path = panel_path(field)
    if path.exists() and not rebuild:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    panel = build_panel(field)
    save_panel(field, panel)
    return panel


def load_panels_for(fields, rebuild=False):
    """수식이 요구하는 field 들 + 항상 필요한 close/funding_rate 를 로드."""
    always = ("close", "funding_rate")
    want = set(fields) | set(always)
    panels = {}
    for f in sorted(want):
        try:
            panels[f] = load_panel(f, rebuild=rebuild)
        except Exception as e:
            if f in always:
                log.warning("항상필요 field %r 로드 실패: %s", f, e)
                continue
            raise
    return panels


# --------------------- bar 주기별 패널 (Phase 3) --------------------- #

def panel_path_at(field, bar):
    """bar='1d' 는 기존 <field>.parquet 재사용, 그 외는 <field>@<bar>.parquet."""
    if bar == "1d":
        return panel_path(field)
    return SETTINGS.panel_dir / f"{field}@{bar}.parquet"


def load_panel_at(field, bar="1d", rebuild=False):
    """field 를 bar 주기 패널로 로드/캐시. bar='1d' 는 load_panel 과 동일."""
    if bar == "1d":
        return load_panel(field, rebuild=rebuild)
    path = panel_path_at(field, bar)
    if path.exists() and not rebuild:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    panel = build_panel_at(field, bar)
    SETTINGS.panel_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)
    return panel


def load_panels_for_bar(fields, bar="1d", rebuild=False):
    """수식 field + 항상필요 close/funding_rate 를 bar 주기로 로드."""
    always = ("close", "funding_rate")
    want = set(fields) | set(always)
    panels = {}
    for f in sorted(want):
        try:
            panels[f] = load_panel_at(f, bar=bar, rebuild=rebuild)
        except Exception as e:
            if f in always:
                log.warning("항상필요 field %r (bar=%s) 로드 실패: %s", f, bar, e)
                continue
            raise
    return panels


# --------------------- 펀딩 이벤트(네이티브 8h) --------------------- #
# 알파 신호용 `funding_rate` 패널은 일단위(FIELD_SPECS)로 집계되지만,
# 펀딩 '비용 회계'는 8h 정산시각(00/08/16 UTC)마다 그 순간 보유 포지션에만
# 부과해야 정확하다(일별 sum 은 손해). 그래서 여기서는 리샘플 없이
# 원본 8h 해상도를 그대로 보존한다. index=UTC 정산시각, columns=심볼.

def build_funding_events() -> pd.DataFrame:
    """모든 심볼의 fundingRate parquet 을 8h 네이티브 해상도로 합쳐
    date-time(정산시각) × coin 패널을 만든다. (일단위 리샘플 안 함)"""
    dataset, time_col, value_col = "fundingRate", "calc_time", "last_funding_rate"
    cols = {}
    for sym in _symbols_for(dataset):
        path = SETTINGS.processed_dir / dataset / f"{sym}.parquet"
        try:
            df = pd.read_parquet(path, columns=[time_col, value_col])
        except Exception:
            continue
        if df.empty:
            continue
        t = pd.to_datetime(df[time_col], unit="ms", utc=True)
        s = pd.Series(df[value_col].to_numpy(), index=t).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if s.notna().any():
            cols[sym] = s
    if not cols:
        raise RuntimeError("funding events: 읽을 수 있는 심볼이 없음 (fundingRate)")
    panel = pd.DataFrame(cols).sort_index()
    return panel


def funding_events_path():
    return SETTINGS.panel_dir / "_funding_events.parquet"


def load_funding_events(rebuild=False):
    path = funding_events_path()
    if path.exists() and not rebuild:
        cached = _read_cache(path)
        if cached is not None:
            return cached
    try:
        panel = build_funding_events()
    except Exception as e:
        log.warning("funding events build failed(0): %s", e)
        return None
    SETTINGS.panel_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)
    return panel
