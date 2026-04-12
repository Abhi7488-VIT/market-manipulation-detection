"""
data_fetch.py
-------------
Fetches all raw data for the Market Manipulation Detection System:
  - Stock OHLCV data         (yfinance)
  - NIFTY index data         (yfinance)
  - India VIX data           (yfinance, with simulated fallback)
  - News headlines           (NewsAPI, with web-scrape + mock fallback)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta

import utils


# ---------------------------------------------------------------------------
# STOCK / INDEX DATA
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker: str, period: str = utils.DEFAULT_PERIOD) -> pd.DataFrame:
    """
    Download OHLCV (Open, High, Low, Close, Volume) data for a single ticker.

    Parameters
    ----------
    ticker : str   e.g. "RELIANCE.NS"
    period : str   yfinance period string e.g. "3mo", "1y"

    Returns
    -------
    pd.DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume
    """
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        # Flatten MultiIndex columns if present (yfinance ≥0.2.x sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        # Drop rows with NaN values (incomplete data for current trading day)
        df = df.dropna(subset=["Close"])
        return df
    except Exception as e:
        print(f"[data_fetch] Warning: Could not fetch {ticker}: {e}. Using simulated data.")
        return _simulate_stock_data(ticker, period)


def fetch_index_data(
    tickers: list = utils.NIFTY_CONSTITUENTS,
    period: str = utils.DEFAULT_PERIOD,
) -> dict:
    """
    Download OHLCV data for multiple tickers (NIFTY constituents).
    Returns a dict mapping ticker -> DataFrame.
    (This function runs sequentially for simplicity; production could use async/multithreading)
    """
    result = {}
    for t in tickers:
        result[t] = fetch_stock_data(t, period)
    return result


def fetch_nifty_data(period: str = utils.DEFAULT_PERIOD) -> pd.DataFrame:
    """Download NIFTY 50 index price data."""
    return fetch_stock_data(utils.NIFTY_INDEX_TICKER, period)


def fetch_major_indices_data(period: str = utils.DEFAULT_PERIOD) -> dict:
    """Download OHLCV data for major broad and sectoral indices (Force Streamlit Reload)"""
    result = {}
    for name, ticker in utils.MAJOR_INDICES.items():
        try:
            df = fetch_stock_data(ticker, period)
            result[name] = df
        except Exception as e:
            print(f"[data_fetch] Could not fetch {name} ({ticker}): {e}")
            result[name] = None
    return result




# ---------------------------------------------------------------------------
# INDIA VIX
# ---------------------------------------------------------------------------

def fetch_vix_data(period: str = utils.DEFAULT_PERIOD) -> pd.DataFrame:
    """
    Download India VIX from Yahoo Finance (^INDIAVIX).
    Falls back to simulated VIX if the ticker is unavailable.

    Returns
    -------
    pd.DataFrame with columns: Close  (VIX value)
    """
    try:
        df = yf.download(utils.VIX_TICKER, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("Empty VIX data")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        # Drop rows with NaN values (incomplete intraday data)
        df = df.dropna(subset=["Close"])
        return df[["Close"]]
    except Exception as e:
        print(f"[data_fetch] VIX fetch failed ({e}). Using simulated VIX.")
        return _simulate_vix_data(period)

# ---------------------------------------------------------------------------
# NEWS HEADLINES
# ---------------------------------------------------------------------------

def fetch_news_headlines(
    query: str = "Indian stock market",
    newsapi_key: str | None = None,
    max_results: int = utils.MAX_HEADLINES,
) -> list[dict]:
    """
    Fetch financial news headlines.

    Tries in order:
      1. NewsAPI  (if api_key provided)
      2. Moneycontrol RSS scrape  (free, no key)
      3. Mock headlines           (always works, for demo purposes)

    Returns
    -------
    list of dicts: [{"title": str, "source": str, "url": str}, ...]
    """
    headlines = []

    # -- Attempt 1: NewsAPI --
    if newsapi_key:
        headlines = _fetch_newsapi(query, newsapi_key, max_results)
        if headlines:
            return headlines

    # -- Attempt 2: Moneycontrol RSS --
    headlines = _fetch_moneycontrol_rss(max_results)
    if headlines:
        return headlines

    # -- Attempt 3: Mock data --
    print("[data_fetch] Using mock news headlines.")
    return _mock_headlines(query, max_results)


def _fetch_newsapi(query: str, api_key: str, max_results: int) -> list[dict]:
    """Query NewsAPI /v2/everything endpoint."""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_results,
            "apiKey": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "ok":
            return []
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", "NewsAPI"),
                "url": a.get("url", ""),
            }
            for a in articles
            if a.get("title")
        ][:max_results]
    except Exception as e:
        print(f"[data_fetch] NewsAPI error: {e}")
        return []


def _fetch_moneycontrol_rss(max_results: int) -> list[dict]:
    """Scrape Moneycontrol markets RSS feed (no API key needed)."""
    try:
        from bs4 import BeautifulSoup
        rss_url = "https://www.moneycontrol.com/rss/marketreports.xml"
        resp = requests.get(rss_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item")[:max_results]
        headlines = []
        for item in items:
            title = item.find("title")
            link  = item.find("link")
            headlines.append({
                "title": title.text.strip() if title else "",
                "source": "Moneycontrol",
                "url": link.text.strip() if link else "",
            })
        return [h for h in headlines if h["title"]]
    except Exception as e:
        print(f"[data_fetch] Moneycontrol RSS error: {e}")
        return []


def _mock_headlines(query: str, max_results: int) -> list[dict]:
    """Return realistic mock headlines for demo / offline use."""
    mocks = [
        {"title": "SEBI tightens surveillance on F&O segment amid unusual activity", "source": "Mock/Demo", "url": ""},
        {"title": "Reliance Industries hits 52-week high on strong earnings beat", "source": "Mock/Demo", "url": ""},
        {"title": "India VIX surges 18% as global uncertainty rattles markets", "source": "Mock/Demo", "url": ""},
        {"title": "NIFTY50 rallies 2% but breadth remains weak; analysts cautious", "source": "Mock/Demo", "url": ""},
        {"title": "Bulk deals worth ₹4,200 Cr spotted in mid-cap space this week", "source": "Mock/Demo", "url": ""},
        {"title": "FIIs turn net buyers in Indian equities; DII selling offsets gains", "source": "Mock/Demo", "url": ""},
        {"title": "NSE flags abnormal volume spike in three small-cap counters", "source": "Mock/Demo", "url": ""},
        {"title": "RBI policy decision supports banking stocks; HDFC Bank up 3%", "source": "Mock/Demo", "url": ""},
        {"title": "Retail investors drive record F&O turnover amid volatile session", "source": "Mock/Demo", "url": ""},
        {"title": "Circuit breakers triggered in four stocks after unusual intraday moves", "source": "Mock/Demo", "url": ""},
    ]
    return mocks[:max_results]


# ---------------------------------------------------------------------------
# SIMULATION FALLBACKS
# ---------------------------------------------------------------------------

def _simulate_stock_data(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data as a fallback."""
    days = _period_to_days(period)
    np.random.seed(abs(hash(ticker)) % (2**31))
    dates = pd.bdate_range(end=datetime.today(), periods=days)
    close = 1000 * np.cumprod(1 + np.random.normal(0.0003, 0.015, days))
    volume = np.random.randint(500_000, 5_000_000, days).astype(float)
    # Inject a couple of volume spikes for demo interest
    spike_idx = np.random.choice(range(days - 5, days), 2, replace=False)
    volume[spike_idx] *= np.random.uniform(3, 6, 2)
    df = pd.DataFrame(
        {
            "Open":   close * np.random.uniform(0.99, 1.00, days),
            "High":   close * np.random.uniform(1.00, 1.02, days),
            "Low":    close * np.random.uniform(0.98, 1.00, days),
            "Close":  close,
            "Volume": volume,
        },
        index=dates,
    )
    return df


def _simulate_vix_data(period: str = "3mo") -> pd.DataFrame:
    """Generate realistic synthetic India VIX data as a fallback."""
    days = _period_to_days(period)
    np.random.seed(42)
    dates = pd.bdate_range(end=datetime.today(), periods=days)
    vix = 14 + np.cumsum(np.random.normal(0, 0.4, days))
    vix = np.clip(vix, 8, 45)
    # Inject a VIX spike near the end for demo interest
    vix[-5:] *= np.linspace(1.0, 1.8, 5)
    vix = np.clip(vix, 8, 45)
    return pd.DataFrame({"Close": vix}, index=dates)


def _period_to_days(period: str) -> int:
    """Convert a yfinance period string to approximate trading days."""
    mapping = {
        "1d": 1, "5d": 5, "1mo": 22, "3mo": 65,
        "6mo": 130, "1y": 252, "2y": 504, "5y": 1260,
    }
    return mapping.get(period, 65)
