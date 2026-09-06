# PROJECT.md — Interview Preparation Guide

**Market Manipulation Detection Terminal (MLMMDS)**

This document explains the project from scratch: what it does, how it is built, why
each decision was made, and what the honest weaknesses are. It is written for
**explaining out loud in an interview**, not for reading as API documentation.

Everything here is based on the actual code in this repository. Where a number or
behaviour is stated, it comes from a real file — those files are named so you can
go look.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture Design](#2-architecture-design)
3. [System Design Decisions](#3-system-design-decisions)
4. [Backend (Theory)](#4-backend-theory)
5. [Frontend (Theory)](#5-frontend-theory)
6. [Tech Stack Summary Table](#6-tech-stack-summary-table)
7. [Data Flow & Request Lifecycle](#7-data-flow--request-lifecycle)
8. [Known Limitations / Future Improvements](#8-known-limitations--future-improvements)
9. [Top 20 Interview Questions & Answers](#9-top-20-interview-questions--answers)

---

## 1. Overview

### What the project does

This is a **market surveillance tool** for the Indian stock market. Think of it as a
security camera pointed at the NIFTY 50 — the 50 biggest companies listed on India's
National Stock Exchange. It watches how those stocks trade every day and flags the
days that look statistically strange.

"Strange" here means: this stock behaved unlike its own normal behaviour, and there
is no news story that explains why. A stock jumping 8% on a day it announced record
profits is not suspicious — that is just news. A stock jumping 8% on ten times its
usual trading volume with total silence in the press is the kind of pattern a human
investigator would want to look at. The tool's whole job is to separate those two
cases and rank what deserves a human's attention first.

### The core problem it solves

Real market manipulation — pump-and-dump schemes, circular trading, insider-driven
moves — is rare, hidden, and **not labelled**. Nobody publishes a dataset saying
"these 400 trading days were manipulation." Regulators like SEBI investigate cases
privately and the outcomes take years. That single fact drives almost every technical
decision in this project: you cannot train a normal machine learning model that
learns from labelled examples, because the labels do not exist.

So instead of asking *"does this look like known manipulation?"*, the system asks a
question it can actually answer from public data: *"is today unusual for this
particular stock, compared to how this stock normally behaves?"* That is an
**anomaly detection** problem rather than a classification problem, and it can be
solved without labels.

### Who it is for

The intended user is a **surveillance analyst** — one person sitting at a desk who
needs to decide which of 50 stocks to investigate this morning. That user shapes the
design more than anything else. There is no login system, no multi-user database, no
alert emails, because there is exactly one analyst looking at read-only public market
data and nothing is being saved between sessions.

The honest framing, and the one you should use in an interview: **this is a triage
and prioritisation tool, not a verdict machine.** It never claims a stock *was*
manipulated. It produces a 0–100 score, a plain-English report explaining which
signals fired, and a ranked list — so a human can decide where to spend their time.
Saying this clearly is a strength, not a weakness. It shows you understand the limits
of what your data can support.

---

## 2. Architecture Design

### The narrative

The system is a **single Python program** that runs as one process. There is no
separate backend server, no database, no message queue, and no API that a browser
calls. When you run `streamlit run app.py`, you get one process that fetches data,
runs the maths, and draws the screen.

It is useful to picture the code in four stages, in order:

**Stage 1 — Get the data.** `data_fetch.py` downloads everything from the outside
world: daily price and volume history for the target stock, for all 50 NIFTY
constituents, for the NIFTY index itself, for five major indices, plus the India VIX
(a "fear index" that measures how much volatility traders expect). It also pulls
recent news headlines. Every one of these downloads has a fallback, which we will
come back to.

**Stage 2 — Turn raw numbers into signals.** `feature_engineering.py` and
`machine_learning.py` convert raw prices into meaningful measurements. The machine
learning module fits an Isolation Forest — an anomaly detection model — and outputs
an anomaly score from 0 to 100. The feature engineering module computes the market
breadth signal and the VIX signal. `news_analysis.py` scores headlines for sentiment.

**Stage 3 — Combine everything into one number.** `detection.py` is the decision
engine. It takes the three signals, combines them using fixed weights, then adjusts
the result up or down based on news, and finally attaches a label like "Suspicious
Activity". It also writes the human-readable report.

**Stage 4 — Draw it.** `app.py` renders everything: seven tabs of charts, a gauge, a
heatmap, and the report.

The important thing to understand — and a very common interview trip-up — is that
the word **"layer"** in this project means an **analytical layer** (a category of
evidence), *not* an architectural tier. The "three layers" are stock, index, and
volatility evidence. They are not three servers.

### Diagram

```
                        ┌──────────────────────────────────┐
                        │        OUTSIDE WORLD             │
                        │  Yahoo Finance  ·  NewsAPI       │
                        │  Moneycontrol RSS                │
                        └───────────────┬──────────────────┘
                                        │  (each call has a fallback)
                                        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  STAGE 1 — DATA ACQUISITION            data_fetch.py               │
   │  target stock OHLCV · 50 constituents · NIFTY · VIX · headlines    │
   │  Cached in memory for 120 seconds                                  │
   └───────────────┬────────────────────────────────────────────────────┘
                   │
                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  STAGE 2 — SIGNAL GENERATION                                       │
   │                                                                    │
   │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐   │
   │  │  STOCK LAYER     │  │  INDEX LAYER     │  │ DERIVATIVE      │   │
   │  │  machine_        │  │  feature_        │  │ LAYER           │   │
   │  │  learning.py     │  │  engineering.py  │  │ feature_eng.py  │   │
   │  │                  │  │                  │  │                 │   │
   │  │ Isolation Forest │  │ How many of the  │  │ Is the VIX      │   │
   │  │ over 5 features  │  │ 50 stocks moved? │  │ spiking vs its  │   │
   │  │ → 0–100          │  │ Did the index    │  │ 20-day average? │   │
   │  │                  │  │ move on thin     │  │ → 0–100         │   │
   │  │                  │  │ participation?   │  │                 │   │
   │  │                  │  │ → 0–100          │  │                 │   │
   │  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘   │
   │           │  weight 40%         │  weight 30%         │  30%       │
   └───────────┼─────────────────────┼─────────────────────┼────────────┘
               └─────────────────────┼─────────────────────┘
                                     ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  STAGE 3 — DECISION ENGINE               detection.py              │
   │                                                                    │
   │   base = 0.40·stock + 0.30·index + 0.30·vix                        │
   │                          │                                         │
   │                          ▼                                         │
   │   NEWS ADJUSTMENT  (news_analysis.py → VADER sentiment)            │
   │   news explains the move   → × 0.75   (less suspicious)            │
   │   news contradicts / silent → × 1.15  (more suspicious)            │
   │                          │                                         │
   │                          ▼                                         │
   │   FINAL SCORE 0–100  +  LABEL  +  written report                   │
   │   0–39 Normal / News-Driven · 40–64 Suspicious · 65–100 High Risk  │
   └───────────────┬────────────────────────────────────────────────────┘
                   │
                   ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │  STAGE 4 — PRESENTATION                   app.py                   │
   │  7 tabs: MARKETS · HEATMAP · STOCK · INDEX · VIX · NEWS · SUMMARY  │
   │  Gauge · Anomaly heatmap (50 stocks × N days) · Ranked watchlist   │
   └────────────────────────────────────────────────────────────────────┘
```

### Walking through the diagram in plain English

Data comes in at the top from three outside sources. It gets cached for two minutes
so that clicking around the interface does not re-download 58 files every time.

That data then splits into three parallel streams of evidence. The **stock layer**
is the machine learning part — it looks at the one stock you selected and asks how
unusual today is compared to that stock's own recent history. The **index layer**
looks at the whole market: if the index moved a lot but almost none of the 50 stocks
participated, that means a handful of names dragged the whole index, which is worth
noticing. The **derivative layer** watches the VIX for a sudden fear spike.

Those three numbers, each on a 0–100 scale, get merged with fixed weights — 40% to
the machine learning score, 30% each to the other two. Then news acts as a **referee**
rather than a fourth vote: if the headlines explain the move, the score is cut by 25%;
if there is no supporting news, it is raised by 15%.

The result is one number, one label, and a written report that shows its own
arithmetic. Finally `app.py` draws it across seven tabs.

---

## 3. System Design Decisions

For each decision: what was chosen, what else was realistic, why this won, and what
was given up.

### 3.1 One single process, not a client–server application

**Chosen:** A Streamlit monolith. One Python process does data fetching, maths, and
rendering. `app.py` is the only entry point.

**Alternatives:** A FastAPI or Flask backend with a React frontend calling it over
HTTP. Or Plotly Dash. Or a scheduled batch job writing to a dashboard.

**Why this won:** The user count is one. There are no user accounts, nothing is
written or saved, and every piece of data is public and read-only. A client–server
split exists to solve problems this project does not have — multiple concurrent
users, independent scaling of frontend and backend, separate deployment cycles.
Splitting it would have meant writing and maintaining an HTTP layer, serialisation,
CORS handling, and a second codebase in a second language, all to serve one analyst
looking at public data. That work buys nothing here.

**Trade-off accepted:** The analytical code and the display code live in the same
process, so you cannot scale them independently, and you cannot reuse the detection
engine from another application without importing the Python modules directly. That
is acceptable because the detection logic is deliberately kept in plain modules
(`detection.py`, `machine_learning.py`, `feature_engineering.py`) that have **no
Streamlit imports at all**. They are importable from a script, a notebook, or a
future API without touching a line. The monolith is a packaging choice, not a
tangling of concerns.

### 3.2 Unsupervised learning, not supervised

**Chosen:** Isolation Forest, an unsupervised anomaly detection algorithm — it learns
what "normal" looks like without being told which days are bad.

**Alternatives:** A supervised classifier (logistic regression, random forest,
gradient boosting) trained on labelled manipulation cases.

**Why this won:** There is no labelled dataset. SEBI does not publish "these were
manipulated days." Building a supervised model would have required inventing labels,
and any labels invented by the developer would just encode the developer's own
assumptions — the model would then "confirm" those assumptions and look impressive
while proving nothing circular.

**Trade-off accepted:** Without labels you cannot compute accuracy, precision, or
recall, so the system can never state how often it is right. This is the single
biggest honest limitation of the project, and it is why the tool is positioned as
triage rather than judgement. The upside is that it works on day one for any stock
with price history, with no data collection effort.

### 3.3 Isolation Forest specifically, over other anomaly detectors

**Chosen:** `IsolationForest` from scikit-learn, with `contamination=0.05` (an
assumption that roughly 5% of trading days are unusual), 100 trees, and a fixed
random seed for reproducibility.

**Alternatives:** Simple statistical rules (flag anything more than 3 standard
deviations from the mean), Local Outlier Factor, One-Class SVM, or an autoencoder
neural network.

**Why this won:**
- **Versus plain standard-deviation rules:** those look at one number at a time. A
  stock can have an unremarkable price move *and* unremarkable volume, but the
  *combination* of the two can still be strange. Isolation Forest looks at all five
  features together, so it catches combinations that single-column rules miss.
- **Versus One-Class SVM:** SVMs are sensitive to parameter tuning and scale poorly
  as data grows. The heatmap fits 50 separate models on every run, so speed matters.
- **Versus an autoencoder:** far more data and tuning than ~65 trading days per stock
  can support, and much harder to explain to a non-technical reviewer.
- **The mechanism fits the problem.** Isolation Forest works by randomly splitting
  the data and seeing how quickly a point gets separated from the rest. Genuinely odd
  points get isolated in very few splits. That is a natural match for "this day
  stands out from this stock's normal behaviour," and it is easy to explain in one
  sentence — which matters for a tool whose output a human has to trust.

**Trade-off accepted:** `contamination=0.05` is an assumption, not something learned
from data. If a stock genuinely had no anomalous days, the model would still label
roughly 5% of them as the most isolated. The score should therefore be read as a
*ranking* — "these are the strangest days for this stock" — not as an absolute
probability.

### 3.4 A fixed weighted formula for combining signals, not a learned model

**Chosen:** The three layer scores are combined with hard-coded weights (40/30/30)
defined in `utils.py`.

**Alternatives:** Learn the weights from data, or feed all signals into a second
model that outputs the final score.

**Why this won:** Learning the weights requires labels, and there are none — same
wall as before. More importantly, this output is meant to justify a human
investigation. A fixed formula means the tool can show its work: the dashboard and
the written report both print `raw score × weight = points` for every layer, then the
weighted base, then the news multiplier and the reason it was applied. An analyst can
audit exactly why a stock scored 71. A learned combiner would be a black box wrapped
around another black box.

**Trade-off accepted:** The weights are a judgement call, not an optimum. They could
be wrong. But they are *visible and adjustable* in one place in `utils.py`, which is
better than being wrong invisibly.

### 3.5 One model per stock, not one model for all stocks

**Chosen:** Each stock gets its own Isolation Forest, fitted only on its own history.
The heatmap fits 50 models on every run.

**Alternatives:** One global model trained on all 50 stocks pooled together.

**Why this won:** Stocks have wildly different normal behaviour. A 4% daily move is
routine for a volatile small-cap and extraordinary for a large stable name like HDFC
Bank. A pooled model would learn one average definition of "normal" and would then
constantly flag the naturally volatile stocks while missing genuinely unusual days in
the calm ones. Per-stock models mean each stock is judged against **itself**.

**Trade-off accepted:** Scores are properly comparable *along one row* of the heatmap
(same stock, same fitted model) but only roughly comparable *across* rows, because
each row came from a different model. The dashboard says this explicitly underneath
the heatmap rather than hiding it.

### 3.6 In-memory caching with a short expiry, not a database

**Chosen:** Streamlit's built-in `cache_data` with a 120-second expiry, wrapping both
the data download and the 50-model heatmap computation.

**Alternatives:** A real database (SQLite, Postgres) storing downloaded history, or
Redis, or no caching at all.

**Why this won:** A cold run makes **58 separate downloads** from Yahoo Finance — the
target stock, the NIFTY index, 50 constituents, the VIX, and 5 major indices — one
after another. Without caching, every single click in the interface would repeat all
58. The cache turns interface interaction from unusable into instant. Two minutes is
chosen deliberately: long enough that clicking between tabs is free, short enough that
an analyst refreshing during market hours gets fresh prices. There is also a manual
**WIPE CACHE** button for forcing a fresh download.

A database was rejected because nothing needs to *persist*. The tool answers "what
does today look like?" using data that is re-downloadable at any moment. A database
would add schema management, migrations, and a staleness problem, in exchange for
storing a copy of data that Yahoo already stores.

**Trade-off accepted:** Everything is lost when the process restarts, and there is no
history of past scores — you cannot ask "what did this stock score last Tuesday?"
That is a genuine limitation and it is the first thing worth adding (see Section 8).

### 3.7 Graceful degradation on data failure, not fail-fast

**Chosen:** Every external call has a fallback chain. If Yahoo Finance fails for a
ticker, `data_fetch.py` generates realistic **synthetic** price data. If the VIX is
unavailable, it simulates one. News tries NewsAPI first, then scrapes Moneycontrol's
RSS feed, then falls back to a fixed list of ten realistic mock headlines.

**Alternatives:** Fail loudly and refuse to render.

**Why this won:** The app depends on free public endpoints that are genuinely
unreliable — during testing, Yahoo returned a 404 for `TATAMOTORS.NS` on a normal
day. If one bad ticker out of 50 could crash the whole dashboard, the tool would be
unusable and undemonstrable.

**Trade-off accepted, and this one is important to state honestly:** the fallback is
only announced on the **console**, not in the user interface. An analyst looking at
the screen cannot tell that one of the 50 stocks is showing simulated numbers. For a
demo that is fine; for real surveillance work it is a correctness problem, because a
tool that silently invents data is worse than one that admits it is broken. Fixing
this is the highest-priority improvement in Section 8.

### 3.8 Sequential downloads, not parallel

**Chosen:** The 58 downloads happen one after another in a plain loop.

**Alternatives:** Threads, `asyncio`, or yfinance's built-in multi-ticker batch call.

**Why this won:** Honestly — simplicity, and the cache makes it a one-time cost per
two-minute window. The code comment in `data_fetch.py` says as much: it runs
"sequentially for simplicity; production could use async/multithreading."

**Trade-off accepted:** The first load is slow, roughly a minute. This is the clearest
known performance weakness and the most obvious thing to improve. Being able to name
it precisely, and say what you would do instead, is worth more in an interview than
pretending it is optimal.

---

## 4. Backend (Theory)

### Language and ecosystem

**Python**, because every part of this problem already has a mature Python library:
scikit-learn for the model, pandas for time-series manipulation, yfinance for market
data, VADER for sentiment. Doing this in Java or Node would have meant either
reimplementing Isolation Forest or calling out to Python anyway.

### How the code is organised, and what each part is responsible for

The design rule is that **each module does one job and does not know about the
others' internals**. Data flows one direction: fetch → measure → decide → display.

| Module | Responsibility | Deliberately does NOT |
|---|---|---|
| `utils.py` | All constants and thresholds in one place — weights, cut-offs, labels, colours | Contain logic |
| `data_fetch.py` | Talk to the outside world; handle every failure | Interpret data |
| `feature_engineering.py` | Turn raw prices into the index and VIX signals | Know about the ML model |
| `machine_learning.py` | Build features, fit Isolation Forests, produce anomaly scores | Know about news or weights |
| `news_analysis.py` | Score headline sentiment with VADER | Know about prices |
| `detection.py` | Combine signals, apply news, classify, write the report | Fetch data or draw anything |
| `stock_utils.py` | Fuzzy-match what the user typed to a real ticker | Anything else |
| `app.py` | Everything visual | Contain detection logic |

The payoff of that separation is concrete: **none of the analysis modules import
Streamlit.** You can `import detection` from a script, a scheduled job, or a future
web API and get identical results with no user interface anywhere. That is why the
smoke test can exercise the whole pipeline headlessly.

`utils.py` deserves a specific mention. Every tunable number — the 40/30/30 weights,
the 3% price threshold, the 2× volume threshold, the 1.5× VIX threshold, the 30%
breadth threshold, the 0.75% index-move threshold, the 40 and 65 score cut-offs —
lives there and nowhere else. Changing the system's behaviour means editing one file,
and the dashboard reads those same constants when it displays the weighting, so the
interface can never drift out of sync with the maths. (It previously did drift, which
is exactly why this matters — see Q13.)

### How data is modelled

There is **no database and no ORM**. Data is modelled in two shapes:

1. **pandas DataFrames** for anything that is a time series — a table with one row
   per trading day and columns for Open, High, Low, Close, Volume. This is the
   natural shape for market data, and it makes rolling calculations (like a 20-day
   average volume) a single operation instead of a manual loop.
2. **Plain Python dictionaries** for results passed between modules. Every signal
   function returns a dictionary containing both the number *and* the evidence behind
   it — for example the breadth function returns the ratio, the list of stocks that
   moved, the index's own move, whether it counts as a "narrow move", the 0–100
   signal, and a list of human-readable warning strings.

That second choice is deliberate and worth explaining: because each result carries
its own explanation, the report generator does not need to recompute anything or
guess why a signal fired. It just reads what each layer already said about itself.

### "API" design

There is no HTTP API, so the relevant contract is the **function interface between
modules**. Two conventions hold throughout:

- **Every signal is normalised to 0–100** before it reaches the decision engine.
  Raw inputs are on wildly different scales — a percentage return, a volume ratio,
  an RSI value — and you cannot meaningfully take a weighted average of things
  measured in different units. Converting everything to a common 0–100 scale first is
  what makes the weighted formula legitimate.
- **Functions return dictionaries, not bare numbers**, so a caller can always get at
  the reasoning. The scoring function returns the final score, the pre-news base, the
  per-layer point contributions, the news multiplier, and a sentence explaining that
  multiplier.

### Error handling, conceptually

Three different strategies, chosen by how bad the failure is:

- **Outside-world failures degrade.** A failed download produces synthetic data and a
  console warning rather than an exception (Section 3.7).
- **Not-enough-data returns a safe neutral value.** If a stock has fewer than 10 clean
  rows of history, the model returns 0 rather than fitting on almost nothing. Fitting
  an anomaly detector to 6 data points would produce confident nonsense; returning
  "no signal" is the honest answer.
- **User error stops the app cleanly.** If the typed symbol matches nothing, the app
  shows an explanatory message and halts instead of continuing with an empty ticker.

### Third-party services and why they were chosen

- **Yahoo Finance (via `yfinance`)** — free, needs no API key, covers NSE tickers with
  the `.NS` suffix, and gives daily OHLCV history. Paid feeds (Bloomberg, Refinitiv,
  or Indian brokers' APIs) offer better reliability and tick-level depth, but cost
  money and usually require an account and approval. For a daily-resolution project
  the free source is sufficient; the reliability cost is exactly what the fallback
  system absorbs.
- **NewsAPI, with a Moneycontrol RSS fallback** — NewsAPI gives structured search but
  needs a key and rate-limits the free tier. Moneycontrol's RSS feed needs no key at
  all and is India-specific, which matters because the stocks are Indian. Trying the
  better source first and quietly falling back to the always-available one gets the
  best of both. The key is typed into a password field at runtime and never stored in
  the repository.
- **VADER for sentiment** — see the table in Section 6 and Q10.

---

## 5. Frontend (Theory)

### The framework, and why

**Streamlit.** The thing to understand about Streamlit is its execution model, which
is unusual: **the entire script runs from top to bottom every time anything changes.**
Click a button, move a slider, switch a theme — `app.py` re-executes from line 1. There
is no event handler and no component tree that updates in place.

That sounds wasteful, and it would be, except that it removes an entire category of
bugs. In a traditional frontend you must manually keep the screen in sync with the
data. Here the screen is *rebuilt from the data* every time, so it cannot fall out of
sync. For a data dashboard, where almost every interaction changes what should be on
screen anyway, this is a very good trade.

**Alternatives:** React with a separate API would give full control over layout and
interaction, but needs a second language, a build step, and an HTTP layer — for a
single-user internal tool that is a large cost for little gain. Plotly Dash is the
closest competitor and is more flexible for complex callbacks, but it requires you to
wire up explicit callback functions, which is more machinery than this app needs.
Jupyter notebooks were rejected because they cannot easily be handed to a
non-technical user.

### How state is managed

Three different mechanisms, each for a different lifetime:

1. **Widget values** (which ticker, which period, heatmap window size) exist for the
   duration of one script run. Because the script re-runs on every interaction, simply
   reading the widget gives the current value — no state store needed.
2. **`session_state`** for things that must survive re-runs. In practice this is the
   dark/light theme toggle. If it were a normal variable it would reset to default on
   every re-run, and the theme would flip back the moment you touched anything else.
3. **`cache_data`** for expensive results that should survive across re-runs *and* be
   shared. This is what stops the 58 downloads and 50 model fits from repeating on
   every click.

The caching has one subtlety worth mentioning if asked. Streamlit normally inspects a
function's arguments to decide whether a cached result can be reused. Inspecting a
dictionary of 50 DataFrames would itself be slow. So those arguments are marked with
a leading underscore to tell Streamlit to skip inspecting them, and a short text key —
built from the ticker, period, last date, and window size — is passed alongside as the
thing to actually compare. Same idea as a cache key in any other system: compare
something cheap that changes when the expensive thing changes.

### Component structure and key patterns

The interface is one page with a sidebar for controls and **seven tabs** for content:
MARKETS, HEATMAP, STOCK, INDEX, VIX, NEWS, SUMMARY. The tabs mirror the analytical
layers, so someone who understands the detection method can find the corresponding
evidence immediately — the INDEX tab shows breadth, the VIX tab shows volatility, the
SUMMARY tab shows the combination and the report.

Two patterns keep `app.py` manageable despite its size:

- **Chart builders are pure functions.** Each `chart_*` function takes data and
  returns a Plotly figure. It draws nothing and touches no global state. That makes
  each chart independently testable and keeps layout separate from rendering.
- **Small HTML helpers for repeated elements.** Rather than repeating markup, helpers
  produce the coloured status pills, the KPI cards, the section headers, and the alert
  strip.

There is also a deliberate **gate**: the app shows an explanatory landing page and
stops until the user presses RUN ANALYSIS. Since a full run costs 58 downloads, doing
that automatically on page load would punish anyone who merely opened the URL.

### Styling

Custom CSS injected as two complete themes (light and dark), plus a set of theme
variables (`TH_TEXT`, `TH_BORDER`, `TH_BG_CARD`, and so on) that get set to different
values depending on the active theme. Charts then reference those variables instead of
hard-coded colours, so both the page and the charts change together.

The visual target is a professional trading terminal — dark background, monospaced
fonts for numbers, dense information, restrained colour used only to mean something
(red for risk, amber for caution, teal for normal). This is a judgement about the
audience: a surveillance analyst reads dense numeric screens all day, and the design
matches the tools they already use.

---

## 6. Tech Stack Summary Table

| Technology | Why this | Why not the closest alternative |
|---|---|---|
| **Python** | Whole ML and data ecosystem already exists here | **Java/Node** — would mean reimplementing Isolation Forest or shelling out to Python anyway |
| **Streamlit** | Data UI with no separate frontend, no build step, no API layer | **React + FastAPI** — second language and HTTP layer for a single-user read-only tool; **Dash** — needs explicit callback wiring this app doesn't need |
| **scikit-learn** | Battle-tested Isolation Forest, consistent API, fast on small data | **TensorFlow/PyTorch** — an autoencoder needs far more data than ~65 days per stock and is much harder to explain |
| **Isolation Forest** | Finds odd *combinations* across 5 features; needs no labels; easy to explain | **One-Class SVM** — tuning-sensitive, slower across 50 models; **z-score rules** — one column at a time, misses combinations |
| **pandas** | Time-series indexing and rolling windows are one-liners | **Raw Python lists** — rolling averages and date alignment become manual, error-prone loops |
| **yfinance** | Free, no key, covers NSE `.NS` tickers, daily OHLCV | **Paid feeds (Bloomberg/Refinitiv/broker APIs)** — better reliability and depth, but cost and account approval, unnecessary at daily resolution |
| **VADER** | Rule-based, no training, tuned for short punchy text like headlines | **FinBERT** — genuinely better on financial language but needs a heavy model download and GPU-ish compute; **TextBlob** — kept only as fallback, weaker on negation and emphasis |
| **Plotly** | Interactive hover/zoom, native heatmap and gauge, works inside Streamlit | **Matplotlib** — static images, no hover; the heatmap needs per-cell inspection to be useful |
| **rapidfuzz** | Typo-tolerant matching so "Tata" or "relance" still finds the right ticker | **Exact string match** — forces users to memorise exact symbols like `BAJAJ-AUTO.NS` |
| **NewsAPI + Moneycontrol RSS** | Structured search first, keyless India-specific feed as backup | **Either one alone** — NewsAPI alone breaks without a key; RSS alone loses per-stock search |
| **BeautifulSoup** | Parses the RSS/XML feed reliably | **Regex on XML** — fragile and a classic source of silent bugs |
| **In-memory TTL cache** | Nothing needs to persist; kills the 58-download repeat cost | **SQLite/Postgres** — schema, migrations, and staleness management to store data Yahoo already stores |

---

## 7. Data Flow & Request Lifecycle

Let us trace one real action all the way through: **an analyst types "Tata Motors",
selects it, and presses RUN ANALYSIS.**

**1 — Typing resolves to a real ticker.** The typed text goes to `stock_utils.py`,
which fuzzy-matches it against 57 supported stocks (the 50 NIFTY constituents plus
seven deliberately volatile names kept for demonstrating the detector). "Tata Motors"
scores highest against `TATAMOTORS.NS`, and the top five matches appear in a dropdown.
If nothing matched at all, the app would explain and stop here.

**2 — Pressing the button re-runs the whole script.** This is the Streamlit model
from Section 5. `app.py` executes from the top; this time the run-button condition is
true, so execution continues past the landing page instead of stopping.

**3 — Data acquisition, or a cache hit.** The cached fetch function is called with the
ticker, period, and news query. If those exact arguments were used in the last 120
seconds, the stored result is returned instantly and steps 3a–3c are skipped
entirely. Otherwise:

- 3a. Daily OHLCV for `TATAMOTORS.NS` is downloaded (default period: 3 months).
- 3b. The same is downloaded for the NIFTY index, all 50 constituents, the India VIX,
  and five major indices — 58 downloads in total, sequentially.
- 3c. News headlines are fetched: NewsAPI if a key was supplied, otherwise
  Moneycontrol's RSS feed, otherwise the built-in mock list.
- Any individual failure here silently substitutes synthetic data.

**4 — The detection pipeline runs.** `detection.py` orchestrates four things:

- **Stock layer.** `machine_learning.py` joins the stock's history to the NIFTY's,
  then builds five features per day: the daily return, the return *relative* to the
  index (which strips out days when the whole market moved), volume compared to its
  own 20-day average, 5-day volatility, and RSI-14 (a standard momentum measure).
  Rows with gaps are dropped. The five columns are then standardised — rescaled so
  each has the same spread — because otherwise a volume ratio in the thousands would
  drown out a return measured in decimals. An Isolation Forest is fitted on the
  result, every day is scored, and the most recent day's score is converted onto the
  0–100 scale.
- **Index layer.** `feature_engineering.py` checks all 50 constituents to count how
  many moved at least 3% today, and separately checks how much the NIFTY index itself
  moved. Both matter — see Q14 for why.
- **Derivative layer.** Today's India VIX is compared with its own 20-day average.
- **News layer.** `news_analysis.py` runs each headline through VADER, producing a
  score from −1 (very negative) to +1 (very positive), and averages them.

**5 — Combination and classification.** The three layer scores are multiplied by
their weights and summed into a base score. Then news adjusts it: if the average
sentiment is strong (absolute value at least 0.35) *and* points the same direction as
the price move, the score is multiplied by 0.75, because the move looks explained. If
sentiment contradicts the move, or there is no strong news at all, it is multiplied by
1.15. The result is clamped to 0–100 and labelled: below 40 is Normal (or "News-Driven"
if strong news is present), 40–64 is Suspicious Activity, 65 and above is High
Manipulation Risk.

**6 — The report is written.** Still in `detection.py`, a plain-text report is
assembled from what each layer already recorded about itself. It opens with the score
arithmetic — each layer's raw score, its weight, and the points it contributed — then
the weighted base, then the news multiplier with the reason it was applied, then a
section per layer with its specific findings, then a final verdict sentence.

**7 — The 50-stock heatmap is computed.** Separately cached, this repeats the machine
learning step for *every* constituent, fitting one Isolation Forest per stock and
scoring each of the last 30 sessions (adjustable from 10 to 60). The result is a table
of 50 rows by 30 columns, sorted so the most anomalous stocks appear first.

**8 — The screen is drawn.** An alert strip states the verdict; six KPI cards show
score, price change, volume spike, VIX, breadth and sentiment; then the seven tabs
render. The HEATMAP tab shows the matrix as a colour grid — green for normal, amber
around 40, red at 65-plus — with a ranked watchlist table beneath it. The SUMMARY tab
shows the gauge, the score-composition breakdown, and the full report.

**9 — Everything after this is free.** Clicking between tabs does not re-run the
analysis; the content is already computed and Streamlit just displays a different tab.
Changing the ticker or pressing RUN ANALYSIS again starts over at step 2 — and if it
is within the two-minute window, even that skips the downloads.

---

## 8. Known Limitations / Future Improvements

Being able to list these clearly is one of the most valuable things you can do in an
interview. It shows you understand your own system's boundaries.

### Limitations that exist today

**1. No way to measure whether it works.** No labelled data means no precision, no
recall, no false-positive rate. The system can only be evaluated qualitatively —
does it flag days that look plausible on inspection? This is the fundamental limit and
it is a property of the problem, not a coding oversight.

**2. Fallback data is invisible in the interface.** When Yahoo fails, synthetic data
is substituted with only a console message. On screen it is indistinguishable from
real data. This is the most important thing to fix, because a surveillance tool that
silently invents numbers is actively misleading. The fix is small: have the fetch
layer report which tickers were substituted and show a banner.

**3. Slow first load.** 58 sequential downloads take roughly a minute cold. Batching
them or fetching in parallel would cut this dramatically.

**4. No history.** Nothing is stored, so you cannot ask what a stock scored last week,
or whether scores are trending, or how often a stock has been flagged this quarter.
Every run is a fresh snapshot.

**5. The scale-mapping constants are hand-tuned.** Converting the model's raw output
to 0–100 saturates at a fixed cut-off chosen by inspection, and `contamination=0.05`
is an assumption. Both are display and modelling choices, not learned values.

**6. VADER is not a finance model.** It is a general-purpose sentiment tool. Phrases
that matter enormously in markets — "beats estimates", "downgrade", "block deal",
"cut guidance" — carry no particular weight in its dictionary.

**7. Daily data only.** Real manipulation often happens *within* a day — spoofing,
layering, marking the close. Daily bars cannot see any of it.

**8. The VIX layer is market-wide, not per-stock.** On a high-VIX day every stock
receives the same 30% contribution, which can lift scores across the board for reasons
that have nothing to do with any individual stock.

**9. Small housekeeping issue.** A section comment in `app.py` describes the cache as
"10 min" while the actual expiry is 120 seconds. The code is right; the comment is
stale.

### What the ideal long-term version looks like

- **Store results over time**, so scores become a time series. That immediately
  enables trend detection ("this stock has drifted upward in anomaly score for two
  weeks") and, crucially, creates the beginnings of a feedback dataset.
- **Capture analyst decisions.** Every time a human reviews a flag and marks it
  genuine or a false alarm, that is a label. Enough of those and the project can
  finally move from unsupervised ranking to a supervised model with real accuracy
  numbers — closing the loop on Limitation 1.
- **Use intraday data** to catch within-day patterns that daily bars hide.
- **Swap VADER for a finance-tuned language model** such as FinBERT, and expand
  beyond headlines to article bodies and regulatory filings.
- **Add relationship-based detection.** Manipulation frequently involves coordinated
  groups of accounts trading among themselves. Individual-stock anomaly detection
  cannot see coordination; a graph-based view of who trades with whom could.
- **Run detection on a schedule** rather than on a button press, so results are ready
  before the analyst arrives, with the dashboard becoming a viewer over stored output.

---

## 9. Top 20 Interview Questions & Answers

**Q1. Tell me about this project in a couple of sentences.**
It is a market surveillance tool for the NIFTY 50. It watches daily trading data and
flags days where a stock behaved unusually compared to its own history and there is no
news that explains the move. It combines a machine learning anomaly score with two
market-context signals into a single 0–100 score, plus a written report explaining why
that score was given. The point is triage — telling an analyst which of 50 stocks to
look at first.

**Q2. Why unsupervised learning instead of a normal classifier?**
Because there are no labels. Nobody publishes a dataset of confirmed manipulation
days — SEBI investigates privately and cases take years. If I had invented labels
myself, the model would just have learned my own assumptions and then "confirmed"
them, which proves nothing. Unsupervised anomaly detection lets me ask a question I
can actually answer from public data: is today unusual for this stock?

**Q3. Why Isolation Forest specifically?**
Two reasons. First, it looks at all five features together, so it catches odd
*combinations* — a price move that is unremarkable on its own and a volume level that
is unremarkable on its own, but strange together. Simple standard-deviation rules look
at one column at a time and miss that. Second, the mechanism is easy to explain: it
randomly splits the data and sees how quickly a point gets separated from everything
else. Genuinely odd points get isolated in very few splits. For a tool whose output a
human has to trust, being explainable matters.

**Q4. How accurate is it?**
I cannot give you an accuracy number, and I would be suspicious of anyone who could on
this problem. There is no labelled ground truth, so there is no precision or recall to
report. What I can say is that it is a ranking and triage tool: it surfaces the most
statistically unusual sessions for review. I validated it qualitatively — checking that
flagged days correspond to real volatility events — and by asserting that each layer
measurably moves the final score. Not claiming accuracy I cannot support is a
deliberate choice.

**Q5. You fit the model on the same data you score. Isn't that data leakage?**
For unsupervised outlier detection this is the standard setup — there is no label to
hold out, and the model's job is to describe the distribution of the window and report
how isolated each point is within it. That said, there is a real weakness: the
anomalous day contributes to the mean and spread used to standardise the features, so
it slightly dampens its own score. The cleaner approach would be to fit the scaler on
a trailing window that excludes the day being scored.

**Q6. Why one model per stock instead of one model for everything?**
Because "normal" is completely different per stock. A 4% daily move is routine for a
volatile small-cap and remarkable for HDFC Bank. A single pooled model would learn one
average notion of normal, then constantly flag volatile stocks and miss genuinely odd
days in calm ones. Per-stock models judge each stock against itself. The cost is that
scores are properly comparable along one heatmap row but only roughly across rows,
and the dashboard says so explicitly.

**Q7. Where did the 40/30/30 weights come from?**
They are a judgement call, not an optimum — I could not learn them because I have no
labels. The reasoning is that the machine learning signal is the only stock-specific
evidence, so it gets the largest share, while breadth and VIX are market-wide context
that should influence but not dominate. What matters is that they live as constants in
one file and the interface displays the arithmetic, so an analyst can audit the score
and I can change the weights in one place.

**Q8. Why not learn the weights or use a model to combine the signals?**
Same labelling problem, plus explainability. The output exists to justify a human
investigation, so it has to show its work — the report literally prints each layer's
raw score, its weight, and the points contributed. A learned combiner would be a black
box wrapped around another black box, and an analyst could not audit why a stock
scored 71.

**Q9. What does `contamination=0.05` mean and how did you choose it?**
It tells the model to expect roughly 5% of sessions to be anomalous. I chose it as a
reasonable prior, not by tuning — without labels there is nothing to tune against.
The honest implication is that if a stock genuinely had no unusual days, the model
would still mark its most isolated 5% as the strangest. That is why the output should
be read as a ranking rather than an absolute probability.

**Q10. Why VADER for sentiment, and what are its weaknesses here?**
VADER is rule-based, needs no training, and is designed for short punchy text, which
is exactly what headlines are — it also handles negation and emphasis well. The real
weakness is that it is a general-purpose tool, not a financial one: phrases like "beats
estimates", "downgrade", or "cut guidance" carry enormous meaning in markets and
nothing in VADER's dictionary. FinBERT would be the upgrade; I did not use it because
it needs a heavy model download and much more compute for what is a secondary signal
here.

**Q11. Why is news a multiplier instead of a fourth weighted layer?**
Because news plays a different role. The other three answer "how unusual is this?"
News answers "is there already an innocent explanation?" A large move with matching
news is not suspicious at all, so news should *scale* the suspicion rather than add to
it. Modelling it as a multiplier — 0.75 when news explains the move, 1.15 when it does
not — captures that relationship. As a fourth additive term, strong news would
paradoxically increase the score.

**Q12. Why Streamlit rather than a proper frontend and API?**
The user count is one, nothing is written or saved, and all the data is public and
read-only. A client–server split solves problems this project does not have —
concurrent users, independent scaling, separate deploy cycles — at the cost of an HTTP
layer and a second codebase. Importantly, I kept the analysis modules free of any
Streamlit imports, so the detection engine can be imported by a script or a future API
without changing a line. The monolith is a packaging choice, not tangled code.

**Q13. Tell me about a bug you found in this project.**
A good one. The system advertised a 40/30/30 weighting in the README and on screen,
but the scoring function only ever used the machine learning score and the news
adjustment — the breadth and VIX signals were computed, displayed, and then discarded.
When I wired them in properly, I found a second bug hiding behind the first: the
breadth layer scored a *quiet* market as maximally suspicious. On a calm day almost
nothing moves, so breadth is near zero, and the old formula turned that into a signal
of about 100. It was harmless only because the value was unused — the moment it was
weighted in, it would have added around 30 points to every stock on every calm day.

**Q14. How did you fix that breadth bug?**
By recognising that low participation only means something if the index itself moved.
The layer now escalates only on what I call a narrow move: the index moved at least
0.75% *and* fewer than 30% of constituents took part — meaning a handful of names
dragged the whole index, which is genuinely worth investigating. A flat, still market
now scores near zero. On a live run with the index up 0.10% and one stock out of 50
moving, the signal came out at 0.8 where the old formula gave about 98.

**Q15. What happens if Yahoo Finance is down?**
Every external call has a fallback. A failed ticker download produces realistic
synthetic price data, the VIX has a simulated fallback, and news falls back from
NewsAPI to a Moneycontrol RSS scrape to a built-in mock list. This matters in practice
— during testing Yahoo returned a 404 for one NIFTY constituent on an ordinary day, and
without fallbacks one bad ticker out of 50 would take down the whole dashboard. The
honest problem is that the substitution is only announced on the console, so on screen
you cannot tell. That is the first thing I would fix, because a surveillance tool that
silently invents data is worse than one that admits it is broken.

**Q16. Why no database?**
Nothing needs to persist. The tool answers "what does today look like?" from data that
can be re-downloaded at any time, so a database would mean schema management,
migrations, and staleness handling in order to store a copy of what Yahoo already
stores. I used a two-minute in-memory cache instead, which solves the real problem —
a cold run makes 58 downloads, and without caching every click would repeat all of
them. The genuine cost is that there is no history, and storing results over time is
the first feature I would add, because it also creates the path to real labels.

**Q17. This takes about a minute to load. How would you speed it up?**
The bottleneck is 58 sequential downloads in a plain loop. The straightforward fix is
to batch or parallelise them — yfinance can fetch multiple tickers in one call, and
the requests are I/O-bound so threads would help a lot. Beyond that, most constituent
history does not change intraday, so it could be fetched once daily and cached
persistently, leaving only the current day to refresh. I left it sequential because
the cache makes it a one-time cost per two-minute window, but I would not defend it as
optimal.

**Q18. How would this scale from 50 stocks to 5,000?**
Two things break. First, data acquisition — 5,000 sequential downloads is impossible,
so it would need parallel fetching and a persistent store refreshed on a schedule
rather than on demand. Second, model fitting — the heatmap currently fits one
Isolation Forest per stock per run, so 5,000 fits per click does not work. I would move
detection into a scheduled batch job that runs after market close, stores the scores,
and turns the dashboard into a viewer over stored results rather than a live compute
engine. That is a natural evolution: the analysis modules already have no interface
dependencies, so they could be run by a scheduler without modification.

**Q19. Isn't flagging stocks as "manipulated" risky? What about false positives?**
Yes, and that is why the tool is deliberately positioned as triage rather than
judgement. It never says manipulation occurred — it says a session is statistically
unusual and lacks news support, which is a prompt for a human to look. The labels
reflect that: the top band is "High Manipulation Risk", not "Manipulation". Every score
comes with a report showing exactly which signals fired, so a reviewer can dismiss it
in seconds if the reasoning is weak. With no ground truth, keeping a human in the loop
is not a limitation to apologise for, it is the correct design.

**Q20. What would you do differently if you started again?**
Three things. I would design for stored results from day one, because that unlocks
trend detection and, more importantly, lets analyst decisions accumulate as labels —
which is the only realistic path from unsupervised ranking to a model with real
accuracy numbers. I would make degraded data visible in the interface from the start
rather than logging it to a console. And I would be more careful about a class of bug
I hit here: the README and the interface described a weighting the code did not
implement, and it survived because nothing checked that documentation matched
behaviour. Now the interface reads the same constants the maths uses, so they cannot
drift apart again.
