"""
detection.py
------------
Core detection engine. Combines signals from all layers into a single
Manipulation Score (0-100) and a classification label with explanation.

Weighting (weights live in utils.py and must sum to 1.0):
  Stock layer       40%  → Isolation Forest anomaly score  (machine_learning.py)
  Index layer       30%  → market breadth / narrow-move signal
  Derivative layer  30%  → India VIX spike signal
  News layer         —   → context multiplier, not a weighted term

News Context Adjustment:
  If strong news aligns with price direction → reduce fused score (false positive)
  If sentiment contradicts the move          → increase fused score
  If no strong news                          → increase fused score (raises suspicion)
"""

import utils


# ---------------------------------------------------------------------------
# MAIN SCORING FUNCTION
# ---------------------------------------------------------------------------

def compute_manipulation_score(
    ml_score: float,
    index_signal: float = 0.0,
    vix_signal: float = 0.0,
    news_score: float = 0.0,
    price_change: float = 0.0,
) -> dict:
    """
    Fuse the three detection layers into a single Manipulation Score (0-100),
    then apply the news-sentiment context adjustment.

    Step 1 — Weighted fusion:
        base = 0.40 * ml_score + 0.30 * index_signal + 0.30 * vix_signal

    Step 2 — News context multiplier (see module docstring).

    Parameters
    ----------
    ml_score      : float  0-100  from machine_learning.train_and_predict_iforest
    index_signal  : float  0-100  from feature_engineering.compute_market_breadth
    vix_signal    : float  0-100  from feature_engineering.compute_vix_spike
    news_score    : float  -1..1  mean VADER compound from news_analysis
    price_change  : float  %      raw price change (to determine direction)

    Returns
    -------
    dict with keys:
        score            (float, 0-100)  final Manipulation Score
        base_score       (float, 0-100)  pre-news weighted fusion
        contributions    (dict)          weighted points contributed per layer
        news_multiplier  (float)         multiplier applied in step 2
        news_reason      (str)           why that multiplier was applied
    """
    # ---- Step 1: weighted multi-layer fusion --------------------------------
    contributions = {
        "stock":      round(utils.WEIGHT_STOCK      * ml_score,     2),
        "index":      round(utils.WEIGHT_INDEX      * index_signal, 2),
        "derivative": round(utils.WEIGHT_DERIVATIVE * vix_signal,   2),
    }
    base_score = sum(contributions.values())

    # ---- Step 2: news context adjustment ------------------------------------
    strong_news = abs(news_score) >= utils.NEWS_STRONG_THRESHOLD
    price_up    = price_change > 0

    if strong_news:
        # Sentiment aligns with price move → likely news-driven, lower suspicion
        sentiment_aligns = (price_up and news_score > 0) or (not price_up and news_score < 0)
        if sentiment_aligns:
            multiplier = 1 - utils.NEWS_REDUCTION_FACTOR
            reason = "Strong news aligned with price direction — likely news-driven"
        else:
            # Sentiment contradicts move → raise suspicion slightly
            multiplier = 1 + utils.NEWS_INCREASE_FACTOR
            reason = "Strong news contradicts price direction — raises suspicion"
    else:
        # Weak / no news → raise suspicion
        multiplier = 1 + utils.NEWS_INCREASE_FACTOR
        reason = "Weak or no news support for the move — raises suspicion"

    final = round(min(max(base_score * multiplier, 0), 100), 2)

    return {
        "score":           final,
        "base_score":      round(base_score, 2),
        "contributions":   contributions,
        "news_multiplier": round(multiplier, 3),
        "news_reason":     reason,
    }


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
    scoring:       dict | None = None,
    ml_score:      float = 0.0,
) -> str:
    """
    Build a human-readable explanation string from all signal results.

    Parameters are the dicts returned by each feature_engineering function
    and news_analysis.analyze_sentiment. `scoring` is the dict returned by
    compute_manipulation_score() and drives the score-composition section.
    """
    lines = [f"### Manipulation Score: {score:.1f} / 100 — {label}\n"]

    # --- Score Composition (audit trail for the fused score) ---
    if scoring:
        c = scoring["contributions"]
        lines.append("**🧮 Score Composition**")
        lines.append(
            f"- Stock layer (ML anomaly): {ml_score:.1f} × {utils.WEIGHT_STOCK:.0%} "
            f"= **{c['stock']:.1f}** pts"
        )
        lines.append(
            f"- Index layer (breadth): {index_result['index_signal']:.1f} × "
            f"{utils.WEIGHT_INDEX:.0%} = **{c['index']:.1f}** pts"
        )
        lines.append(
            f"- Derivative layer (VIX): {vix_result['vix_signal']:.1f} × "
            f"{utils.WEIGHT_DERIVATIVE:.0%} = **{c['derivative']:.1f}** pts"
        )
        lines.append(f"- Weighted base score: **{scoring['base_score']:.1f}**")
        lines.append(
            f"- News adjustment: × {scoring['news_multiplier']:.2f} "
            f"({scoring['news_reason']})"
        )
        lines.append(f"- Final score: **{score:.1f} / 100**\n")

    # --- Stock Layer ---
    lines.append("**📊 Stock Layer**")
    lines.append(f"- Isolation Forest anomaly score: {ml_score:.1f} / 100")
    lines.append(f"- Price change: {stock_result['price_change_pct']:+.2f}%")
    lines.append(f"- Volume spike: {stock_result['volume_spike']:.2f}x rolling average")
    if stock_result["anomaly_flags"]:
        for flag in stock_result["anomaly_flags"]:
            lines.append(f"  ⚠ {flag}")
    else:
        lines.append("  ✓ No stock-layer anomalies detected")

    # --- Index Layer ---
    lines.append("\n**📈 Index / Breadth Layer**")
    lines.append(f"- NIFTY 1-day move: {index_result.get('index_move_pct', 0.0):+.2f}%")
    lines.append(f"- Breadth ratio: {index_result['breadth_ratio']*100:.0f}% stocks moved significantly")
    lines.append(f"- Movers: {', '.join(index_result['movers']) if index_result['movers'] else 'None'}")
    if index_result["anomaly_flags"]:
        # A quiet-session note is informational; only a narrow move is a warning.
        icon = "⚠" if index_result.get("narrow_move") else "✓"
        for flag in index_result["anomaly_flags"]:
            lines.append(f"  {icon} {flag}")
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
    index_result = compute_market_breadth(constituent_dfs, index_df=nifty_df)
    vix_result   = compute_vix_spike(vix_df)
    news_result  = analyze_sentiment(headlines)

    ml_anomaly_score = machine_learning.train_and_predict_iforest(stock_df, nifty_df, demo_mode)

    scoring = compute_manipulation_score(
        ml_score      = ml_anomaly_score,
        index_signal  = index_result["index_signal"],
        vix_signal    = vix_result["vix_signal"],
        news_score    = news_result["mean_score"],
        price_change  = stock_result["price_change_pct"],
    )
    score = scoring["score"]

    label       = classify_score(score, news_result["mean_score"])
    explanation = generate_explanation(
        stock_result, index_result, vix_result, news_result, score, label,
        scoring=scoring, ml_score=ml_anomaly_score,
    )

    return {
        "stock_result":  stock_result,
        "index_result":  index_result,
        "vix_result":    vix_result,
        "news_result":   news_result,
        "ml_score":      ml_anomaly_score,
        "scoring":       scoring,
        "score":         score,
        "label":         label,
        "explanation":   explanation,
    }
