# 🛡️ Market Manipulation Detection Terminal

> **A Multi-Layer Surveillance System for the Indian Equity Market**

Real-time market manipulation detection dashboard that combines **Isolation Forest anomaly detection**, **cross-market signal fusion** across three analytical layers, and **VADER sentiment analysis** to identify suspicious trading activity in NIFTY 50 stocks.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Detection Layers](#-detection-layers)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [License](#-license)

---

## ✨ Features

- **🤖 Unsupervised ML Detection** — Isolation Forest trained on 5 behavioral features; no labeled data required
- **📊 Three-Layer Signal Fusion** — ML anomaly score + Market breadth + VIX volatility combined by weight (40/30/30) into a single manipulation score (0–100)
- **🔥 Universe Anomaly Heatmap** — every NIFTY 50 constituent scored across the trailing N sessions, ranked most-anomalous first, with a watchlist table
- **🧾 Auditable Scoring** — every score shows its per-layer arithmetic (`raw × weight = points`) in both the dashboard and the generated report
- **📰 Sentiment-Aware Scoring** — VADER NLP adjusts scores: −25% for news-supported moves, +15% for unexplained activity
- **🔍 Fuzzy Ticker Search** — Type any company name or partial ticker to find stocks instantly
- **📈 Interactive Charts** — Anomaly heatmap, candlestick OHLCV, VIX trends, breadth analysis, sentiment bars via Plotly
- **🌓 Dual Theme** — Professional dark mode and light mode with full CSS theming
- **⚡ Real-Time Data** — Live market data via yfinance, news from NewsAPI/Moneycontrol RSS
- **🧪 Stress Test Mode** — Inject synthetic anomalies to demonstrate detection capability

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                         │
│  yfinance (OHLCV) │ NIFTY 50 Breadth │ India VIX │ NewsAPI │
└──────────┬──────────────────┬────────────────┬──────────────┘
           │                  │                │
           ▼                  ▼                ▼
┌──────────────────┐ ┌────────────────┐ ┌──────────────────┐
│   STOCK LAYER    │ │  INDEX LAYER   │ │ DERIVATIVE LAYER │
│   Weight: 40%    │ │  Weight: 30%   │ │   Weight: 30%    │
│                  │ │                │ │                  │
│ ISOLATION FOREST │ │ • Breadth %    │ │ • VIX Spike Ratio│
│ 5 features →     │ │ • Index move % │ │ • Rolling Avg    │
│ anomaly score    │ │ • Narrow move? │ │ • VIX Signal     │
│ (unsupervised)   │ │ • Index Signal │ │                  │
└────────┬─────────┘ └───────┬────────┘ └────────┬─────────┘
         │                   │                    │
         └───────────────────┼────────────────────┘
                             ▼
              ┌──────────────────────────┐
              │    WEIGHTED FUSION       │
              │  0.40·ML + 0.30·Index    │
              │       + 0.30·VIX         │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │  NEWS SENTIMENT ADJUST   │
              │  VADER Compound Score    │
              │  Align → −25% │ No → +15%│
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │   MANIPULATION SCORE     │
              │      0 — 100 Gauge       │
              │                          │
              │  0-39  Normal            │
              │  0-39  News-Driven       │
              │ 40-64  Suspicious        │
              │ 65-100 High Risk         │
              └──────────────────────────┘
```

---

## 🔬 Detection Layers

| Layer | Weight | Features | Signal fires when |
|-------|--------|----------|-------------------|
| **Stock** | 40% | Isolation Forest anomaly score over 5 engineered features | Session is isolated vs the stock's own history |
| **Index** | 30% | NIFTY 50 breadth (% stocks moving ≥ 3%) + index 1-day move | **Narrow move**: index moves ≥ 0.75% *and* breadth < 30% |
| **Derivative** | 30% | India VIX spike ratio vs 20d rolling average | VIX ratio ≥ 1.5× |
| **News** | Adjustment | VADER sentiment compound score on headlines | \|score\| ≥ 0.35 |

> **On the Index layer:** low breadth alone is *not* suspicious — on a quiet
> session almost nothing moves and breadth is legitimately near zero. The layer
> only escalates on a **narrow move**: the index itself moved materially while
> few constituents participated, i.e. the index was dragged by a handful of
> names. A flat, still market scores near 0 here.

Price change % (5d) and volume spike ratio remain available as descriptive
stock-layer diagnostics in the dashboard and report, but the **weighted stock
signal is the ML anomaly score** — not the threshold heuristics.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit, Plotly |
| **ML Model** | scikit-learn (Isolation Forest) |
| **NLP** | VADER Sentiment Analysis |
| **Data** | yfinance, NewsAPI, BeautifulSoup |
| **Search** | rapidfuzz (fuzzy matching) |
| **Language** | Python 3.11+ |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/market-manipulation-detection.git
cd market-manipulation-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install additional dependency

```bash
pip install rapidfuzz
```

---

## 💻 Usage

### Run the dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Quick Start

1. **Enter a stock** — Type a ticker (e.g., `RELIANCE.NS`) or company name in the sidebar
2. **Select a match** — Choose from fuzzy search suggestions
3. **Set period** — Choose analysis window (1mo, 3mo, 6mo, 1y)
4. **Run Analysis** — Click the button and explore the seven analysis tabs
5. **Heatmap** — Open the HEATMAP tab to see all 50 constituents scored across recent sessions, with a ranked watchlist
6. **Stress Test** — Enable "Simulate Anomaly" checkbox to inject synthetic anomalies

### Optional: NewsAPI Key

For live news headlines, get a free API key from [newsapi.org](https://newsapi.org/) and enter it in the sidebar.

---

## 📁 Project Structure

```
market-manipulation-detection/
├── app.py                    # Main Streamlit dashboard (UI + charts)
├── detection.py              # Core detection engine & scoring pipeline
├── feature_engineering.py    # Signal computation (stock/index/VIX layers)
├── machine_learning.py       # Isolation Forest training & prediction
├── news_analysis.py          # VADER sentiment analysis module
├── data_fetch.py             # Data acquisition (yfinance, NewsAPI, RSS)
├── stock_utils.py            # Fuzzy ticker search (rapidfuzz)
├── utils.py                  # Constants, thresholds, configuration
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

All detection parameters are centralized in `utils.py`:

```python
# Anomaly Thresholds
PRICE_CHANGE_THRESHOLD   = 3.0   # % price change to flag
VOLUME_SPIKE_THRESHOLD   = 2.0   # volume ratio to flag
VIX_SPIKE_THRESHOLD      = 1.5   # VIX ratio to flag
BREADTH_LOW_THRESHOLD    = 0.30  # breadth fraction (< 30%)
INDEX_MOVE_THRESHOLD     = 0.75  # % index move that makes breadth meaningful

# Detection Weights (must sum to 1.0)
WEIGHT_STOCK      = 0.40
WEIGHT_INDEX      = 0.30
WEIGHT_DERIVATIVE = 0.30

# News Sentiment
NEWS_REDUCTION_FACTOR = 0.25     # reduce score for aligned news
NEWS_INCREASE_FACTOR  = 0.15     # increase score for weak/no news
NEWS_STRONG_THRESHOLD = 0.35     # |compound| above this = strong signal

# Classification
SCORE_SUSPICIOUS = 40            # score ≥ 40 → Suspicious
SCORE_HIGH_RISK  = 65            # score ≥ 65 → High Risk
```

---

## 🧠 How It Works

### Isolation Forest ML Pipeline

1. **Feature Engineering** — 5 features extracted from joined stock + NIFTY data:
   - Daily Return (%)
   - Relative Return (Stock − Index)
   - Volume Spike Ratio (vs 20-day SMA)
   - Rolling Volatility (5-day StdDev)
   - RSI-14 (Momentum)

2. **Standardization** — Features scaled to zero mean, unit variance via `StandardScaler`

3. **Training** — Isolation Forest fitted with `contamination=0.05`, `n_estimators=100`

4. **Scoring** — `decision_function` output normalized: negative scores → 0-100 manipulation risk

5. **Layer Fusion** — the ML score is combined with the index and VIX signals by weight

6. **News Adjustment** — VADER sentiment modifies the fused score based on news alignment

### Scoring Formula

```
base = 0.40 · ML_anomaly + 0.30 · index_signal + 0.30 · vix_signal

If strong news aligns with price direction:
    Final Score = base × 0.75   (−25%)

If sentiment contradicts, or no strong news:
    Final Score = base × 1.15   (+15%)
```

Every score is auditable: the dashboard and the generated report both print the
per-layer arithmetic (`raw × weight = points`), the weighted base, and the news
multiplier with the reason it was applied.

### Universe Anomaly Heatmap

`machine_learning.build_anomaly_matrix()` fits one Isolation Forest per stock
over its own feature history, then scores every session in the window — giving
a **50 stocks × N sessions** matrix rendered as a heatmap and a ranked watchlist.

```python
from machine_learning import build_anomaly_matrix
matrix = build_anomaly_matrix(constituent_dfs, nifty_df, n_days=30)
```

Scores are comparable **along a row** (same stock, same fitted model). Across
rows they rank relative isolation — they are **not** calibrated probabilities of
manipulation.

### Interpreting Scores

This is a **triage and prioritization tool, not a validated classifier.** There is
no public labelled ground truth for market manipulation, so the system reports no
precision/recall figures. A high score means "this session looks statistically
unusual against this stock's own recent behaviour and lacks news support" — it is
a prompt for human review, not a determination of wrongdoing.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built for market transparency and investor protection 🛡️</b>
</p>
