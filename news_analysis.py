"""
news_analysis.py
----------------
Fetches and analyses news headlines using VADER sentiment analysis.

VADER (Valence Aware Dictionary and sEntiment Reasoner) works well on
short social-media / financial headline text without needing training.
Compound score ranges from -1 (very negative) to +1 (very positive).
"""

import utils

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False
    print("[news_analysis] vaderSentiment not installed. Falling back to TextBlob.")

try:
    from textblob import TextBlob
    _TEXTBLOB_AVAILABLE = True
except ImportError:
    _TEXTBLOB_AVAILABLE = False


# ---------------------------------------------------------------------------
# SENTIMENT ANALYSER
# ---------------------------------------------------------------------------

def analyze_sentiment(headlines: list[dict]) -> dict:
    """
    Run sentiment analysis on a list of headline dicts.

    Each dict must have at least a "title" key.

    Returns
    -------
    dict with keys:
        scores          list[float]   compound score per headline  (-1 to 1)
        mean_score      float         average compound score
        labels          list[str]     'Positive' / 'Negative' / 'Neutral' per headline
        overall_label   str           overall news sentiment label
        strong_signal   bool          True if |mean_score| >= NEWS_STRONG_THRESHOLD
    """
    if not headlines:
        return _empty_sentiment()

    scores = []
    for h in headlines:
        text = h.get("title", "")
        if not text:
            scores.append(0.0)
            continue
        score = _score_text(text)
        scores.append(score)

    mean_score = sum(scores) / len(scores) if scores else 0.0
    labels     = [classify_news_sentiment(s) for s in scores]

    return {
        "scores":        [round(s, 4) for s in scores],
        "mean_score":    round(mean_score, 4),
        "labels":        labels,
        "overall_label": classify_news_sentiment(mean_score),
        "strong_signal": abs(mean_score) >= utils.NEWS_STRONG_THRESHOLD,
    }


def classify_news_sentiment(compound: float) -> str:
    """
    Map a VADER compound score to a human-readable label.

    Thresholds follow VADER's recommended cutoffs:
        compound >= 0.05  → Positive
        compound <= -0.05 → Negative
        else              → Neutral
    """
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _score_text(text: str) -> float:
    """Return a compound sentiment score for a single text string."""
    if _VADER_AVAILABLE:
        analyser = SentimentIntensityAnalyzer()
        return analyser.polarity_scores(text)["compound"]
    elif _TEXTBLOB_AVAILABLE:
        # TextBlob polarity is -1 to 1; treat as a reasonable proxy
        return TextBlob(text).sentiment.polarity
    else:
        # If neither library is available, return neutral
        return 0.0


def _empty_sentiment() -> dict:
    return {
        "scores":        [],
        "mean_score":    0.0,
        "labels":        [],
        "overall_label": "Neutral",
        "strong_signal": False,
    }
