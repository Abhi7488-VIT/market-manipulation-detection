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
- **📊 Three-Layer Signal Fusion** — Stock anomalies + Market breadth + VIX volatility combined into a single manipulation score (0–100)
- **📰 Sentiment-Aware Scoring** — VADER NLP adjusts scores: −25% for news-supported moves, +15% for unexplained activity
- **🔍 Fuzzy Ticker Search** — Type any company name or partial ticker to find stocks instantly
- **📈 Interactive Charts** — Candlestick OHLCV, VIX trends, breadth analysis, sentiment bars via Plotly
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
│ • Price Δ (5d)   │ │ • Breadth %    │ │ • VIX Spike Ratio│
│ • Volume Spike   │ │ • Movers Count │ │ • Rolling Avg    │
│ • Stock Signal   │ │ • Index Signal │ │ • VIX Signal     │
└────────┬─────────┘ └───────┬────────┘ └────────┬─────────┘
         │                   │                    │
         └───────────────────┼────────────────────┘
                             ▼
              ┌──────────────────────────┐
              │    ISOLATION FOREST      │
              │   5 Features → Score     │
              │  (Unsupervised ML)       │
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

| Layer | Weight | Features | Threshold |
|-------|--------|----------|-----------|
| **Stock** | 40% | Price change % (5d), Volume spike ratio (20d avg) | Price ≥ 3%, Volume ≥ 2× |
| **Index** | 30% | NIFTY 50 market breadth (% stocks moving ≥ 3%) | Breadth < 30% |
| **Derivative** | 30% | India VIX spike ratio vs 20d rolling average | VIX ratio ≥ 1.5× |
| **News** | Adjustment | VADER sentiment compound score on headlines | \|score\| ≥ 0.35 |

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
4. **Run Analysis** — Click the button and explore the six analysis tabs
5. **Stress Test** — Enable "Simulate Anomaly" checkbox to inject synthetic anomalies

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

5. **News Adjustment** — VADER sentiment modifies the ML score based on news alignment

### Scoring Formula

```
If strong news aligns with price direction:
    Final Score = ML Score × 0.75  (−25%)

If no strong news or sentiment contradicts:
    Final Score = ML Score × 1.15  (+15%)
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>Built for market transparency and investor protection 🛡️</b>
</p>
