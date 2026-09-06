"""
feature_engineering.py
-----------------------
Computes all signal features used by the detection layer:

  - Price change %          → stock anomaly
  - Volume spike ratio      → stock anomaly
  - Combined stock signal   → 0-100 score
  - Market breadth          → index anomaly (low-breadth flag)
  - VIX spike ratio         → derivative anomaly
"""

import numpy as np
import pandas as pd

import utils


# ---------------------------------------------------------------------------
# STOCK LAYER SIGNALS
# ---------------------------------------------------------------------------

def compute_price_change(df: pd.DataFrame, window: int = utils.PRICE_CHANGE_WINDOW) -> float:
    """
    Percentage price change over the last `window` trading days.

    Uses Close prices.
    Returns a float (can be negative).
    """
    if len(df) < window + 1:
        window = max(1, len(df) - 1)
    close = df["Close"].dropna()
    if len(close) < 2:
        return 0.0
    change = (close.iloc[-1] - close.iloc[-(window + 1)]) / close.iloc[-(window + 1)] * 100
    # Flatten any nested array/Series that yfinance may return
    if hasattr(change, "item"):
        change = change.item()
    return float(change)


def compute_volume_spike(
    df: pd.DataFrame, window: int = utils.VOLUME_ROLLING_WINDOW
) -> float:
    """
    Volume spike ratio = current volume / rolling-average volume.

    A ratio > 1.0 means above-average volume.
    A ratio > 2.0 (VOLUME_SPIKE_THRESHOLD) is flagged as an anomaly.
    """
    vol = df["Volume"].dropna()
    if len(vol) < window + 1:
        window = max(1, len(vol) - 1)
    rolling_avg = vol.iloc[-(window + 1) : -1].mean()
    current_vol = vol.iloc[-1]
    if rolling_avg == 0:
        return 1.0
    ratio = float(current_vol) / float(rolling_avg)
    return round(ratio, 3)


def compute_stock_signal(
    df: pd.DataFrame,
    price_threshold: float = utils.PRICE_CHANGE_THRESHOLD,
    volume_threshold: float = utils.VOLUME_SPIKE_THRESHOLD,
) -> dict:
    """
    Combine price change and volume spike into a stock anomaly score (0-100).

    Scoring logic:
      - Each component contributes up to 50 points.
      - Price score  = min(|price_change| / threshold * 50, 50)
      - Volume score = min((spike_ratio - 1) / (threshold - 1) * 50, 50)

    Returns
    -------
    dict with keys:
        price_change_pct  (float)
        volume_spike      (float)
        price_score       (float, 0-50)
        volume_score      (float, 0-50)
        stock_signal      (float, 0-100)
        anomaly_flags     (list[str])
    """
    price_change = compute_price_change(df)
    volume_spike  = compute_volume_spike(df)

    abs_price = abs(price_change)
    price_score  = min(abs_price / price_threshold * 50, 50)
    volume_score = min(max((volume_spike - 1) / (volume_threshold - 1) * 50, 0), 50)

    stock_signal = price_score + volume_score

    flags = []
    if abs_price >= price_threshold:
        flags.append(f"Price moved {price_change:+.2f}% (≥{price_threshold}% threshold)")
    if volume_spike >= volume_threshold:
        flags.append(f"Volume spike {volume_spike:.2f}x rolling avg (≥{volume_threshold}x threshold)")

    return {
        "price_change_pct": round(price_change, 3),
        "volume_spike":      round(volume_spike, 3),
        "price_score":       round(price_score, 2),
        "volume_score":      round(volume_score, 2),
        "stock_signal":      round(stock_signal, 2),
        "anomaly_flags":     flags,
    }


# ---------------------------------------------------------------------------
# INDEX LAYER SIGNALS
# ---------------------------------------------------------------------------

def compute_market_breadth(
    constituent_dfs: dict,
    index_df: pd.DataFrame | None = None,
    price_threshold: float = utils.PRICE_CHANGE_THRESHOLD,
) -> dict:
    """
    Compute market breadth as the fraction of constituent stocks that moved
    significantly (|price_change| >= threshold) in the latest session.

    Low breadth on its own is NOT suspicious — on a quiet session nothing moves
    and breadth is legitimately near zero. What matters for surveillance is a
    NARROW MOVE: the index itself moved materially while only a handful of
    constituents participated, i.e. the index was dragged by a few names.
    That combination is what this layer scores.

    Parameters
    ----------
    constituent_dfs : dict  {ticker: pd.DataFrame}
    index_df        : pd.DataFrame | None  index OHLCV (NIFTY). When omitted the
                      layer cannot distinguish a quiet market from a narrow move
                      and stays in its low-signal branch.
    price_threshold : float  movement threshold %

    Returns
    -------
    dict with keys:
        breadth_ratio        (float, 0-1)  fraction of stocks moving
        movers               (list[str])   tickers that moved
        non_movers           (list[str])
        low_breadth          (bool)  raw breadth below threshold
        index_move_pct       (float) 1-day index move %
        narrow_move          (bool)  index moved AND breadth was low
        index_signal         (float, 0-100)
        anomaly_flags        (list[str])
    """
    movers, non_movers = [], []

    for ticker, df in constituent_dfs.items():
        if df is None or df.empty:
            continue
        change = abs(compute_price_change(df, window=1))  # 1-day change
        if change >= price_threshold:
            movers.append(ticker)
        else:
            non_movers.append(ticker)

    total = len(movers) + len(non_movers)
    breadth_ratio = len(movers) / total if total > 0 else 0.0
    low_breadth   = breadth_ratio < utils.BREADTH_LOW_THRESHOLD

    # Did the index itself move enough for participation to be meaningful?
    if index_df is not None and not index_df.empty:
        index_move = compute_price_change(index_df, window=1)
    else:
        index_move = 0.0
    index_moved = abs(index_move) >= utils.INDEX_MOVE_THRESHOLD

    narrow_move = index_moved and low_breadth

    if narrow_move:
        # Index moved on thin participation → concentration is the signal.
        index_signal = (1 - breadth_ratio) * 100
    else:
        # Quiet market, or a broad move with healthy participation.
        # Churn contributes a mild baseline; a flat, still market scores ~0.
        index_signal = breadth_ratio * 40

    flags = []
    if narrow_move:
        flags.append(
            f"Narrow index move: NIFTY moved {index_move:+.2f}% but only "
            f"{len(movers)}/{total} stocks moved ≥{price_threshold}% "
            f"(index driven by few stocks)"
        )
    elif low_breadth and not index_moved:
        flags.append(
            f"Quiet session: index flat ({index_move:+.2f}%) and only "
            f"{len(movers)}/{total} stocks moved ≥{price_threshold}% — benign"
        )

    return {
        "breadth_ratio":  round(breadth_ratio, 3),
        "movers":         movers,
        "non_movers":     non_movers,
        "low_breadth":    low_breadth,
        "index_move_pct": round(index_move, 3),
        "narrow_move":    narrow_move,
        "index_signal":   round(index_signal, 2),
        "anomaly_flags":  flags,
    }


# ---------------------------------------------------------------------------
# DERIVATIVE LAYER SIGNALS
# ---------------------------------------------------------------------------

def compute_vix_spike(
    vix_df: pd.DataFrame,
    window: int = utils.VIX_ROLLING_WINDOW,
) -> dict:
    """
    Measure the VIX spike ratio = current VIX / rolling-average VIX.

    Returns
    -------
    dict with keys:
        current_vix    (float)
        rolling_avg    (float)
        vix_spike      (float)  ratio
        spike_flag     (bool)
        vix_signal     (float, 0-100)
        anomaly_flags  (list[str])
    """
    close = vix_df["Close"].dropna()
    if len(close) < window + 1:
        window = max(1, len(close) - 1)

    current_vix = float(close.iloc[-1])
    rolling_avg  = float(close.iloc[-(window + 1): -1].mean())

    if rolling_avg == 0:
        ratio = 1.0
    else:
        ratio = current_vix / rolling_avg

    spike_flag = ratio >= utils.VIX_SPIKE_THRESHOLD

    # Score: ratio of 1.5 → 50 pts; ratio ≥ 2.0 → 100 pts
    vix_signal = min((ratio - 1) / (utils.VIX_SPIKE_THRESHOLD - 1) * 50, 100)
    vix_signal = max(vix_signal, 0)

    flags = []
    if spike_flag:
        flags.append(
            f"India VIX spike: {current_vix:.2f} vs rolling avg {rolling_avg:.2f} "
            f"(ratio {ratio:.2f}x ≥ {utils.VIX_SPIKE_THRESHOLD}x)"
        )

    return {
        "current_vix":   round(current_vix, 2),
        "rolling_avg":   round(rolling_avg, 2),
        "vix_spike":     round(ratio, 3),
        "spike_flag":    spike_flag,
        "vix_signal":    round(vix_signal, 2),
        "anomaly_flags": flags,
    }
