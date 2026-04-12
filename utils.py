"""
utils.py
--------
Shared constants, configuration, and utility helpers for the
Multi-Layer Market Manipulation Detection System.
"""

# ---------------------------------------------------------------------------
# TICKERS
# ---------------------------------------------------------------------------
# Primary target stock (user-configurable at runtime via Streamlit sidebar)
DEFAULT_TICKER = "RELIANCE.NS"

# NIFTY 50 index ticker on Yahoo Finance
NIFTY_INDEX_TICKER = "^NSEI"

# Major macro indices for the market overview tab
MAJOR_INDICES = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "NIFTY IT": "^CNXIT",
    "NIFTY AUTO": "^CNXAUTO"
}


# NIFTY 50 full constituent list used for breadth calculation
# Source: NSE India — as of March 2026
NIFTY_CONSTITUENTS = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS",
    "BAJFINANCE.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]

# India VIX ticker on Yahoo Finance  (may not always be available)
VIX_TICKER = "^INDIAVIX"

# ---------------------------------------------------------------------------
# ROLLING WINDOW SIZES
# ---------------------------------------------------------------------------
VOLUME_ROLLING_WINDOW = 20      # days for volume spike baseline
VIX_ROLLING_WINDOW    = 20      # days for VIX spike baseline
PRICE_CHANGE_WINDOW   = 5       # look-back days for price change %

# ---------------------------------------------------------------------------
# ANOMALY THRESHOLDS
# ---------------------------------------------------------------------------
PRICE_CHANGE_THRESHOLD   = 3.0  # % – price change beyond this is flagged
VOLUME_SPIKE_THRESHOLD   = 2.0  # ratio – current vol / rolling avg
VIX_SPIKE_THRESHOLD      = 1.5  # ratio – current VIX / rolling avg
BREADTH_LOW_THRESHOLD    = 0.30 # fraction – <30% stocks moving = low breadth

# ---------------------------------------------------------------------------
# DETECTION WEIGHTS  (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_STOCK      = 0.40
WEIGHT_INDEX      = 0.30
WEIGHT_DERIVATIVE = 0.30

# ---------------------------------------------------------------------------
# NEWS SENTIMENT ADJUSTMENT
# ---------------------------------------------------------------------------
# When sentiment strongly aligns with price move, lower the final score
NEWS_REDUCTION_FACTOR   = 0.25   # reduce score by up to 25%
NEWS_INCREASE_FACTOR    = 0.15   # increase score by up to 15%
NEWS_STRONG_THRESHOLD   = 0.35   # |compound| above this = strong sentiment

# ---------------------------------------------------------------------------
# LABEL THRESHOLDS  (manipulation score 0-100)
# ---------------------------------------------------------------------------
LABEL_NORMAL       = "Normal"
LABEL_NEWS_DRIVEN  = "News-Driven Movement"
LABEL_SUSPICIOUS   = "Suspicious Activity"
LABEL_HIGH_RISK    = "High Manipulation Risk"

SCORE_SUSPICIOUS = 40   # score ≥ this → Suspicious
SCORE_HIGH_RISK  = 65   # score ≥ this → High Risk

# ---------------------------------------------------------------------------
# COLOR MAP – used by Streamlit/Plotly
# ---------------------------------------------------------------------------
LABEL_COLORS = {
    LABEL_NORMAL:      "#2ecc71",   # green
    LABEL_NEWS_DRIVEN: "#3498db",   # blue
    LABEL_SUSPICIOUS:  "#f39c12",   # orange
    LABEL_HIGH_RISK:   "#e74c3c",   # red
}

# ---------------------------------------------------------------------------
# DEFAULT DATE RANGE
# ---------------------------------------------------------------------------
DEFAULT_PERIOD = "3mo"   # passed to yfinance download

# ---------------------------------------------------------------------------
# NEWS
# ---------------------------------------------------------------------------
MAX_HEADLINES = 10  # maximum headlines to fetch / display


def score_color(score: float) -> str:
    """Return a hex color for a manipulation score 0-100."""
    if score >= SCORE_HIGH_RISK:
        return LABEL_COLORS[LABEL_HIGH_RISK]
    elif score >= SCORE_SUSPICIOUS:
        return LABEL_COLORS[LABEL_SUSPICIOUS]
    elif score >= 20:
        return LABEL_COLORS[LABEL_NEWS_DRIVEN]
    else:
        return LABEL_COLORS[LABEL_NORMAL]
