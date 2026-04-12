"""
detection.py
------------
Core detection engine. Combines signals from all layers into a single
Manipulation Score (0-100) and a classification label with explanation.

Weighting:
  Isolation Forest ML Model → 100% Primary Anomaly Classification
  News Context Adjustment   → Secondary False-Positive Downgrader

News Context Adjustment:
  If strong news aligns with price direction → reduce ML score (false positive)
  If no strong news                          → increase ML score (raises suspicion)
"""

import utils


# ---------------------------------------------------------------------------
# MAIN SCORING FUNCTION
# ---------------------------------------------------------------------------

def compute_manipulation_score(
    ml_score: float,
    news_score: float,
    price_change: float,
) -> float:
    """
    Compute the Manipulation Score (0-100) using the Isolation Forest output.

    Parameters
    ----------
    ml_score      : float  0-100  from machine_learning.train_and_predict_iforest
    news_score    : float  -1..1  mean VADER compound from news_analysis
    price_change  : float  %      raw price change (to determine direction)

    Returns
    -------
    float  Manipulation Score clamped to [0, 100]
    """
    raw_score = ml_score

    # Step 2 – News context adjustment
    strong_news = abs(news_score) >= utils.NEWS_STRONG_THRESHOLD
    price_up    = price_change > 0

    if strong_news:
        # Sentiment aligns with price move → likely news-driven, lower suspicion
        sentiment_aligns = (price_up and news_score > 0) or (not price_up and news_score < 0)
        if sentiment_aligns:
            raw_score *= (1 - utils.NEWS_REDUCTION_FACTOR)
        else:
            # Sentiment contradicts move → raise ML suspicion slightly
            raw_score *= (1 + utils.NEWS_INCREASE_FACTOR)
    else:
        # Weak / no news → raise ML suspicion
        raw_score *= (1 + utils.NEWS_INCREASE_FACTOR)

    return round(min(max(raw_score, 0), 100), 2)


# ---------------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------------

def classify_score(score: float, news_score: float = 0.0) -> str:
    """
    Map a Manipulation Score to a label.

    Thresholds defined in utils.py:
      0  – 39  → Normal  (or News-Driven if strong news)
      40 – 64  → Suspicious Activity
      65 – 100 → High Manipulation Risk
    """
    if score >= utils.SCORE_HIGH_RISK:
        return utils.LABEL_HIGH_RISK
    elif score >= utils.SCORE_SUSPICIOUS:
        return utils.LABEL_SUSPICIOUS
    elif abs(news_score) >= utils.NEWS_STRONG_THRESHOLD:
        return utils.LABEL_NEWS_DRIVEN
    else:
        return utils.LABEL_NORMAL


# ---------------------------------------------------------------------------
# EXPLANATION GENERATOR
# ---------------------------------------------------------------------------

def generate_explanation(
    stock_result:  dict,
    index_result:  dict,
    vix_result:    dict,
    news_result:   dict,
    score:         float,
    label:         str,
) -> str:
    """
    Build a human-readable explanation string from all signal results.

    Parameters are the dicts returned by each feature_engineering function
    and news_analysis.analyze_sentiment.
    """
    lines = [f"### Manipulation Score: {score:.1f} / 100 — {label}\n"]

    # --- Stock Layer ---
    lines.append("**📊 Stock Layer**")
    lines.append(f"- Price change: {stock_result['price_change_pct']:+.2f}%")
    lines.append(f"- Volume spike: {stock_result['volume_spike']:.2f}x rolling average")
    if stock_result["anomaly_flags"]:
        for flag in stock_result["anomaly_flags"]:
            lines.append(f"  ⚠ {flag}")
    else:
        lines.append("  ✓ No stock-layer anomalies detected")

    # --- Index Layer ---
    lines.append("\n**📈 Index / Breadth Layer**")
    lines.append(f"- Breadth ratio: {index_result['breadth_ratio']*100:.0f}% stocks moved significantly")
    lines.append(f"- Movers: {', '.join(index_result['movers']) if index_result['movers'] else 'None'}")
    if index_result["anomaly_flags"]:
        for flag in index_result["anomaly_flags"]:
            lines.append(f"  ⚠ {flag}")
    else:
        lines.append("  ✓ Market breadth is healthy")

    # --- Derivative Layer ---
    lines.append("\n**📉 Derivative / VIX Layer**")
    lines.append(f"- India VIX: {vix_result['current_vix']:.2f}  (rolling avg: {vix_result['rolling_avg']:.2f})")
    lines.append(f"- VIX spike ratio: {vix_result['vix_spike']:.2f}x")
    if vix_result["anomaly_flags"]:
        for flag in vix_result["anomaly_flags"]:
            lines.append(f"  ⚠ {flag}")
    else:
        lines.append("  ✓ VIX within normal range")

    # --- News Context ---
    lines.append("\n**📰 News Context**")
    lines.append(f"- Mean sentiment score: {news_result['mean_score']:+.3f}")
    lines.append(f"- Overall sentiment: {news_result['overall_label']}")
    if news_result["strong_signal"]:
        lines.append("  ✓ Strong news signal detected — adjusts confidence")
    else:
        lines.append("  ⚠ Weak/no news support — raises suspicion")

    # --- Final Verdict ---
    lines.append(f"\n**🔍 Final Verdict:** {label}")
    verdict_map = {
        utils.LABEL_NORMAL:      "All signals within normal parameters. No manipulation detected.",
        utils.LABEL_NEWS_DRIVEN: "Significant movement appears supported by news sentiment. Low manipulation risk.",
        utils.LABEL_SUSPICIOUS:  "Multiple anomalous signals detected with limited news justification. Monitor closely.",
        utils.LABEL_HIGH_RISK:   "Strong multi-layer anomaly signals with no news support. High manipulation risk — escalate review.",
    }
    lines.append(verdict_map.get(label, ""))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CONVENIENCE: RUN FULL PIPELINE
# ---------------------------------------------------------------------------

def run_detection_pipeline(
    stock_df,
    nifty_df,
    constituent_dfs: dict,
    vix_df,
    headlines: list[dict],
    demo_mode: bool = False
) -> dict:
    """
    End-to-end helper that calls all feature engineering functions and
    returns a complete result dict ready for the Streamlit dashboard.

    Returns
    -------
    dict with keys:
        stock_result, index_result, vix_result, news_result,
        score, label, explanation
    """
    from feature_engineering import compute_stock_signal, compute_market_breadth, compute_vix_spike
    from news_analysis import analyze_sentiment
    import machine_learning

    stock_result = compute_stock_signal(stock_df)
    index_result = compute_market_breadth(constituent_dfs)
    vix_result   = compute_vix_spike(vix_df)
    news_result  = analyze_sentiment(headlines)

    ml_anomaly_score = machine_learning.train_and_predict_iforest(stock_df, nifty_df, demo_mode)

    score = compute_manipulation_score(
        ml_score      = ml_anomaly_score,
        news_score    = news_result["mean_score"],
        price_change  = stock_result["price_change_pct"],
    )

    label       = classify_score(score, news_result["mean_score"])
    explanation = generate_explanation(
        stock_result, index_result, vix_result, news_result, score, label
    )

    return {
        "stock_result":  stock_result,
        "index_result":  index_result,
        "vix_result":    vix_result,
        "news_result":   news_result,
        "score":         score,
        "label":         label,
        "explanation":   explanation,
    }
