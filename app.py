"""
app.py  —  Market Surveillance Terminal
Bloomberg / TradingView-style professional dark UI
Run:  python -m streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

import utils
import stock_utils
import data_fetch
import detection
import machine_learning
import importlib
importlib.reload(utils)
importlib.reload(stock_utils)
importlib.reload(data_fetch)
importlib.reload(machine_learning)
importlib.reload(detection)

from data_fetch import (
    fetch_stock_data, fetch_index_data,
    fetch_nifty_data, fetch_vix_data, fetch_news_headlines,
    fetch_major_indices_data
)

from detection import run_detection_pipeline
# Force streamlit reload to pick up data_fetch.py changes

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Surveillance Terminal",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM  —  Bloomberg Terminal Dark
# ─────────────────────────────────────────────────────────────────────────────
#   bg-primary   #0b0e11   | near-black canvas
#   bg-panel     #131722   | card / panel background
#   bg-border    #0f172a   | subtle dividers
#   accent-teal  #059669   | primary accent (up / positive)
#   accent-red   #dc2626   | negative / risk
#   accent-amber #d97706   | warning / suspicious
#   accent-blue  #3b82f6   | neutral / informational
#   text-primary #0f172a
#   text-muted   #64748b
#   text-dim     #94a3b8


if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

CSS_LIGHT = """
<style>
/* ── Google Fonts ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & Canvas ───────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* ── Hide Streamlit chrome ────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem !important; max-width: 100% !important; }

/* ── Global text overrides for light mode ─────── */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stMarkdown div, .stText, div[data-testid="stText"] {
    color: #1e293b !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #0f172a !important;
}

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #f1f5f9 !important;
    border-right: 1px solid #cbd5e1 !important;
    width: 270px !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #334155 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.03em;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox select {
    background: #ffffff !important;
    border: 1px solid #94a3b8 !important;
    color: #0f172a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 4px !important;
}
section[data-testid="stSidebar"] hr { border-color: #cbd5e1 !important; }

/* ── Primary Button ───────────────────────────── */
.stButton > button[kind="primary"] {
    background: #ea580c !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #c2410c !important;
    box-shadow: 0 0 16px rgba(234,88,12,0.35) !important;
}

/* ── All buttons text fix ────────────────────── */
.stButton > button {
    color: #1e293b !important;
}
.stButton > button[kind="primary"] {
    color: #ffffff !important;
}

/* ── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #cbd5e1 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #475569 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #059669 !important;
    border-bottom: 2px solid #059669 !important;
    background: transparent !important;
}

/* ── Expander ─────────────────────────────────── */
.streamlit-expander {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 4px !important;
}
.streamlit-expander summary span {
    color: #1e293b !important;
}

/* ── Dataframe / Table ────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #cbd5e1 !important; border-radius: 4px !important; }

/* ── Spinners / alerts ────────────────────────── */
.stSpinner > div { border-top-color: #059669 !important; }
.stInfo  { background: rgba(59,130,246,0.10) !important; border-left-color: #3b82f6 !important; color:#334155 !important; }

/* ── KPI Card ─────────────────────────────────── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.70rem;
    color: #475569;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Alert Strip ──────────────────────────────── */
.alert-strip {
    padding: 10px 18px;
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
}
.alert-normal  { background: rgba(5,150,105,0.08); border-left: 3px solid #059669; color: #047857; }
.alert-news    { background: rgba(59,130,246,0.08); border-left: 3px solid #3b82f6; color: #2563eb; }
.alert-susp    { background: rgba(217,119,6,0.08); border-left: 3px solid #d97706; color: #b45309; }
.alert-high    { background: rgba(220,38,38,0.10);  border-left: 3px solid #dc2626; color: #b91c1c; }

/* ── Status pill ──────────────────────────────── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border-radius: 3px;
    padding: 3px 10px;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.pill-normal { background: rgba(5,150,105,0.12); color: #047857; border: 1px solid rgba(5,150,105,0.25); }
.pill-news   { background: rgba(59,130,246,0.12); color: #2563eb; border: 1px solid rgba(59,130,246,0.25); }
.pill-susp   { background: rgba(217,119,6,0.12); color: #b45309; border: 1px solid rgba(217,119,6,0.25); }
.pill-high   { background: rgba(220,38,38,0.12);  color: #b91c1c; border: 1px solid rgba(220,38,38,0.30); }
.pill-pos    { background: rgba(5,150,105,0.10); color: #047857; border: 1px solid rgba(5,150,105,0.20); }
.pill-neg    { background: rgba(220,38,38,0.10);  color: #b91c1c; border: 1px solid rgba(220,38,38,0.20); }
.pill-neu    { background: rgba(100,116,139,0.12); color: #475569; border: 1px solid rgba(100,116,139,0.20); }

/* ── Section header ───────────────────────────── */
.section-hdr {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #334155;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 6px;
    margin-bottom: 14px;
}

/* ── Terminal header ──────────────────────────── */
.term-header {
    background: #f1f5f9;
    border-bottom: 1px solid #cbd5e1;
    padding: 12px 0 10px 0;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.term-brand {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.90rem;
    font-weight: 600;
    color: #059669;
    letter-spacing: 0.06em;
}
.term-clock {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
}

/* ── News row ─────────────────────────────────── */
.news-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid #e2e8f0;
    font-size: 0.82rem;
}
.news-title { color: #0f172a; line-height: 1.4; }
.news-meta  { font-size: 0.70rem; color: #475569; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }

/* ── Explanation panel ────────────────────────── */
.expl-panel {
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 20px 24px;
    font-size: 0.95rem;
    line-height: 2.2;
    color: #334155;
}
.expl-panel h3 { color: #0f172a; font-size: 0.92rem; margin-bottom: 12px; }
.expl-panel strong { color: #0f172a; }

/* ── Toggle (Dark Mode switch) ───────────────── */
/* Toggle label text */
[data-testid="stToggle"] label span,
[data-testid="stToggle"] label p,
[data-testid="stToggle"] p,
label[data-testid="stToggle"] span,
label[data-baseweb="checkbox"] p,
label[data-baseweb="checkbox"] span {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}
/* Toggle track (the slider background) - unchecked state */
[data-baseweb="checkbox-toggle"],
[data-testid="stToggle"] [data-baseweb="checkbox-toggle"],
label[data-baseweb="checkbox"] > div:first-child,
[data-testid="stToggle"] [role="checkbox"],
[data-testid="stToggle"] > div > div[role="checkbox"],
[data-testid="stToggle"] label > div:first-child,
label[data-testid="stToggle"] > div:first-child,
[data-testid="stToggle"] div[data-baseweb="toggle"] {
    background-color: #64748b !important;
    border: 2px solid #475569 !important;
    border-radius: 999px !important;
    min-width: 42px !important;
    min-height: 22px !important;
}
/* Toggle track - checked state */
[data-baseweb="checkbox-toggle"][aria-checked="true"],
[data-testid="stToggle"] [data-baseweb="checkbox-toggle"][aria-checked="true"],
label[data-baseweb="checkbox"] > div:first-child[aria-checked="true"],
[data-testid="stToggle"] [role="checkbox"][aria-checked="true"],
[data-testid="stToggle"] > div > div[role="checkbox"][aria-checked="true"],
[data-testid="stToggle"] label > div:first-child[aria-checked="true"],
label[data-testid="stToggle"] > div:first-child[aria-checked="true"] {
    background-color: #059669 !important;
    border-color: #047857 !important;
}
/* Toggle thumb (the circle that slides) */
[data-baseweb="checkbox-toggle"] > div,
[data-testid="stToggle"] [data-baseweb="checkbox-toggle"] > div,
label[data-baseweb="checkbox"] > div:first-child > div,
[data-testid="stToggle"] [role="checkbox"] > div,
[data-testid="stToggle"] > div > div[role="checkbox"] > div,
[data-testid="stToggle"] label > div:first-child > div,
label[data-testid="stToggle"] > div:first-child > div {
    background-color: #ffffff !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
    border-radius: 50% !important;
    width: 18px !important;
    height: 18px !important;
}

/* ── Selectbox / Dropdown — fix dark backgrounds ─ */
[data-baseweb="select"],
[data-baseweb="select"] > div,
[data-baseweb="select"] > div > div,
.stSelectbox > div > div,
.stSelectbox [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: #94a3b8 !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="select"] input {
    color: #0f172a !important;
}
/* Dropdown menu items */
[data-baseweb="menu"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"] li,
ul[role="listbox"],
ul[role="listbox"] li {
    background-color: #ffffff !important;
    color: #0f172a !important;
}
ul[role="listbox"] li:hover,
[data-baseweb="menu"] li:hover {
    background-color: #f1f5f9 !important;
}
/* Selectbox arrow icon */
[data-baseweb="select"] svg {
    fill: #475569 !important;
}

/* ── Labels for all form widgets ───────────────── */
.stSelectbox label, .stTextInput label, .stCheckbox label,
.stSelectbox > label, .stTextInput > label, .stCheckbox > label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stCheckbox label {
    color: #334155 !important;
}

/* ── Checkbox — fix invisible text ───────────── */
.stCheckbox label span,
.stCheckbox label p,
.stCheckbox span,
[data-testid="stCheckbox"] label span,
[data-testid="stCheckbox"] span {
    color: #1e293b !important;
    font-weight: 500 !important;
}

/* ── Toggle label — fix invisible text ───────── */
[data-testid="stToggle"] label,
[data-testid="stToggle"] label span,
[data-testid="stToggle"] label p,
[data-testid="stToggle"] > label > span,
[data-testid="stToggle"] span:not([role="checkbox"] span),
label[data-testid="stToggle"],
label[data-testid="stToggle"] span {
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
}

/* ── Text input — fix styling ────────────────── */
.stTextInput input,
section[data-testid="stSidebar"] .stTextInput input,
[data-testid="stTextInput"] input {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-color: #94a3b8 !important;
}

/* ── Generic Streamlit widget text ─────────────── */
.stMarkdown p, .stMarkdown span, .stMarkdown div,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span {
    color: #1e293b !important;
}

/* ── Help tooltip icon ───────────────────────── */
.stTooltipIcon svg {
    fill: #64748b !important;
}
</style>
"""

CSS_DARK = """
<style>
/* ── Google Fonts ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & Canvas ───────────────────────────── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #0b0e11 !important;
    color: #e2e8f0 !important;
}

/* ── Hide Streamlit chrome ────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ──────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2730 !important;
    width: 270px !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.03em;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox select {
    background: #131722 !important;
    border: 1px solid #1e2730 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    border-radius: 4px !important;
}
section[data-testid="stSidebar"] hr { border-color: #1e2730 !important; }

/* ── Primary Button ───────────────────────────── */
.stButton > button[kind="primary"] {
    background: #00d4aa !important;
    color: #0b0e11 !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #00b894 !important;
    box-shadow: 0 0 16px rgba(0,212,170,0.35) !important;
}

/* ── Tabs ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1e2730 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748b !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4aa !important;
    border-bottom: 2px solid #00d4aa !important;
    background: transparent !important;
}

/* ── Expander ─────────────────────────────────── */
.streamlit-expander {
    background: #131722 !important;
    border: 1px solid #1e2730 !important;
    border-radius: 4px !important;
}

/* ── Dataframe / Table ────────────────────────── */
[data-testid="stDataFrame"] { border: 1px solid #1e2730 !important; border-radius: 4px !important; }

/* ── Spinners / alerts ────────────────────────── */
.stSpinner > div { border-top-color: #00d4aa !important; }
.stInfo  { background: rgba(59,130,246,0.10) !important; border-left-color: #3b82f6 !important; color:#94a3b8 !important; }

/* ── KPI Card ─────────────────────────────────── */
.kpi-card {
    background: #131722;
    border: 1px solid #1e2730;
    border-radius: 6px;
    padding: 16px 20px;
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.70rem;
    color: #475569;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Alert Strip ──────────────────────────────── */
.alert-strip {
    padding: 10px 18px;
    border-radius: 4px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
}
.alert-normal  { background: rgba(0,212,170,0.08); border-left: 3px solid #00d4aa; color: #00d4aa; }
.alert-news    { background: rgba(59,130,246,0.08); border-left: 3px solid #3b82f6; color: #3b82f6; }
.alert-susp    { background: rgba(245,158,11,0.08); border-left: 3px solid #f59e0b; color: #f59e0b; }
.alert-high    { background: rgba(239,68,68,0.10);  border-left: 3px solid #ef4444; color: #ef4444; }

/* ── Status pill ──────────────────────────────── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    border-radius: 3px;
    padding: 3px 10px;
    font-size: 0.73rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.pill-normal { background: rgba(0,212,170,0.12); color: #00d4aa; border: 1px solid rgba(0,212,170,0.25); }
.pill-news   { background: rgba(59,130,246,0.12); color: #3b82f6; border: 1px solid rgba(59,130,246,0.25); }
.pill-susp   { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.25); }
.pill-high   { background: rgba(239,68,68,0.12);  color: #ef4444; border: 1px solid rgba(239,68,68,0.30); }
.pill-pos    { background: rgba(0,212,170,0.10); color: #00d4aa; border: 1px solid rgba(0,212,170,0.20); }
.pill-neg    { background: rgba(239,68,68,0.10);  color: #ef4444; border: 1px solid rgba(239,68,68,0.20); }
.pill-neu    { background: rgba(100,116,139,0.12); color: #64748b; border: 1px solid rgba(100,116,139,0.20); }

/* ── Section header ───────────────────────────── */
.section-hdr {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #334155;
    border-bottom: 1px solid #1e2730;
    padding-bottom: 6px;
    margin-bottom: 14px;
}

/* ── Terminal header ──────────────────────────── */
.term-header {
    background: #0d1117;
    border-bottom: 1px solid #1e2730;
    padding: 12px 0 10px 0;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.term-brand {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.90rem;
    font-weight: 600;
    color: #00d4aa;
    letter-spacing: 0.06em;
}
.term-clock {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #334155;
}

/* ── News row ─────────────────────────────────── */
.news-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid #1e2730;
    font-size: 0.82rem;
}
.news-title { color: #cbd5e1; line-height: 1.4; }
.news-meta  { font-size: 0.70rem; color: #475569; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }

/* ── Explanation panel ────────────────────────── */
.expl-panel {
    background: #131722;
    border: 1px solid #1e2730;
    border-radius: 6px;
    padding: 20px 24px;
    font-size: 0.84rem;
    line-height: 1.7;
    color: #94a3b8;
}
.expl-panel h3 { color: #e2e8f0; font-size: 0.92rem; margin-bottom: 12px; }
.expl-panel strong { color: #cbd5e1; }

/* ── Toggle (Dark Mode switch) ───────────────── */
[data-testid="stToggle"] label span {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
[data-testid="stToggle"] [role="checkbox"] {
    background-color: #334155 !important;
    border: 1px solid #475569 !important;
}
[data-testid="stToggle"] [role="checkbox"][aria-checked="true"] {
    background-color: #00d4aa !important;
    border-color: #00d4aa !important;
}
[data-testid="stToggle"] [role="checkbox"] > div {
    background-color: #ffffff !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4) !important;
}
</style>
"""

if st.session_state.dark_mode:
    st.markdown(CSS_DARK, unsafe_allow_html=True)

    PLOT_LAYOUT = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1318",
        font=dict(family="Inter", color="#64748b", size=11),
        margin=dict(l=12, r=12, t=36, b=12),
        xaxis=dict(
            gridcolor="#1e2730", gridwidth=1,
            showline=False, zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=10, color="#475569"),
        ),
        yaxis=dict(
            gridcolor="#1e2730", gridwidth=1,
            showline=False, zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=10, color="#475569"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#64748b"),
            orientation="h", yanchor="bottom", y=1.02,
        ),
    )

    COLOR_UP   = "#00d4aa"
    COLOR_DOWN = "#ef4444"
    COLOR_DIM  = "#334155"
    COLOR_AMB  = "#f59e0b"

else:
    st.markdown(CSS_LIGHT, unsafe_allow_html=True)

    PLOT_LAYOUT = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter", color="#64748b", size=11),
        margin=dict(l=12, r=12, t=36, b=12),
        xaxis=dict(
            gridcolor="#0f172a", gridwidth=1,
            showline=False, zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=10, color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="#0f172a", gridwidth=1,
            showline=False, zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=10, color="#64748b"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#64748b"),
            orientation="h", yanchor="bottom", y=1.02,
        ),
    )

    COLOR_UP   = "#059669"   # teal  — positive / up
    COLOR_DOWN = "#dc2626"   # red   — negative / down
    COLOR_DIM  = "#94a3b8"   # dim   — neutral, grid
    COLOR_AMB  = "#d97706"   # amber — warning

# ── Theme-aware inline style tokens ──
if st.session_state.dark_mode:
    TH_BG_CARD    = "#131722"
    TH_BORDER     = "#1e2730"
    TH_TEXT       = "#e2e8f0"
    TH_TEXT_MUTED = "#94a3b8"
    TH_TEXT_DIM   = "#64748b"
    TH_BADGE_BG   = "rgba(255,255,255,0.04)"
    TH_CTA_BG     = "rgba(255,255,255,0.02)"
    TH_BRAND_CLR  = "#00d4aa"
else:
    TH_BG_CARD    = "#ffffff"
    TH_BORDER     = "#e2e8f0"
    TH_TEXT       = "#0f172a"
    TH_TEXT_MUTED = "#475569"
    TH_TEXT_DIM   = "#64748b"
    TH_BADGE_BG   = "rgba(0,0,0,0.04)"
    TH_CTA_BG     = "rgba(0,0,0,0.02)"
    TH_BRAND_CLR  = "#059669"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _pill(label: str) -> str:
    cls = {
        utils.LABEL_NORMAL:      "pill-normal",
        utils.LABEL_NEWS_DRIVEN: "pill-news",
        utils.LABEL_SUSPICIOUS:  "pill-susp",
        utils.LABEL_HIGH_RISK:   "pill-high",
    }.get(label, "pill-normal")
    dots = {"pill-normal":"●", "pill-news":"●", "pill-susp":"●", "pill-high":"●"}
    return f'<span class="pill {cls}">{dots.get(cls,"●")} {label}</span>'

def _sent_pill(s: str) -> str:
    cls = {"Positive":"pill-pos","Negative":"pill-neg","Neutral":"pill-neu"}.get(s,"pill-neu")
    return f'<span class="pill {cls}">{s}</span>'

def _alert(label: str, msg: str):
    cls = {
        utils.LABEL_NORMAL:      "alert-normal",
        utils.LABEL_NEWS_DRIVEN: "alert-news",
        utils.LABEL_SUSPICIOUS:  "alert-susp",
        utils.LABEL_HIGH_RISK:   "alert-high",
    }.get(label, "alert-normal")
    icon = {
        utils.LABEL_NORMAL:      "●",
        utils.LABEL_NEWS_DRIVEN: "◆",
        utils.LABEL_SUSPICIOUS:  "▲",
        utils.LABEL_HIGH_RISK:   "⬥",
    }.get(label, "●")
    st.markdown(f'<div class="alert-strip {cls}"><span>{icon}</span><span>{msg}</span></div>',
                unsafe_allow_html=True)

def _kpi(label: str, value: str, sub: str = "", color: str = "#0f172a"):
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def _section(title: str):
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)

def _score_color(s: float) -> str:
    if s >= 65: return COLOR_DOWN
    if s >= 40: return COLOR_AMB
    return COLOR_UP

# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def chart_ohlcv(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick + volume — TradingView style."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.68, 0.32], vertical_spacing=0.02,
    )
    # ── Candlestick ──
    up   = df["Close"] >= df["Open"]
    down = ~up
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        increasing=dict(line=dict(color=COLOR_UP,   width=1), fillcolor=COLOR_UP),
        decreasing=dict(line=dict(color=COLOR_DOWN, width=1), fillcolor=COLOR_DOWN),
        name=ticker, showlegend=False,
        whiskerwidth=0.3,
    ), row=1, col=1)
    # ── 20-day EMA ──
    ema = df["Close"].ewm(span=20).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=ema, mode="lines",
        line=dict(color="#3b82f6", width=1.2, dash="dot"),
        name="EMA-20",
    ), row=1, col=1)
    # ── Volume ──
    vol_colors = np.where(up, COLOR_UP, COLOR_DOWN)
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vol_colors, marker_opacity=0.55,
        name="Volume",
    ), row=2, col=1)
    # ── Volume rolling avg ──
    vol_avg = df["Volume"].rolling(utils.VOLUME_ROLLING_WINDOW).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_avg, mode="lines",
        line=dict(color=COLOR_AMB, width=1, dash="dot"),
        name=f"Vol MA{utils.VOLUME_ROLLING_WINDOW}",
    ), row=2, col=1)
    fig.update_layout(
        **PLOT_LAYOUT,
        height=440,
        title=dict(text=f"{ticker}  ·  OHLCV", font=dict(size=11, color="#64748b"), x=0),
        xaxis_rangeslider_visible=False,
    )
    # Override legend position separately (PLOT_LAYOUT already defines legend, so must be separate)
    fig.update_layout(legend=dict(
        orientation="h",
        x=1, xanchor="right",
        y=1.0, yanchor="top",
        bgcolor="rgba(15,19,24,0.80)",
        bordercolor="#0f172a",
        borderwidth=1,
        font=dict(size=9, color="#64748b"),
    ))
    fig.update_yaxes(title_text="₹", row=1, col=1,
                     title_font=dict(size=10, color="#94a3b8"),
                     tickprefix="₹", tickfont=dict(family="JetBrains Mono", size=10))
    fig.update_yaxes(title_text="Vol", row=2, col=1,
                     title_font=dict(size=10, color="#94a3b8"),
                     tickfont=dict(family="JetBrains Mono", size=10))
    return fig


def chart_nifty(nifty_df: pd.DataFrame, breadth: dict) -> go.Figure:
    """NIFTY line + breadth donut side by side."""
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.70, 0.30],
        specs=[[{"type":"xy"},{"type":"domain"}]],
        subplot_titles=["NIFTY 50", f"Breadth  [{len(breadth['movers'])}/{len(breadth['movers'])+len(breadth['non_movers'])} moved]"],
    )
    fig.add_trace(go.Scatter(
        x=nifty_df.index, y=nifty_df["Close"],
        mode="lines", name="NIFTY 50",
        line=dict(color="#3b82f6", width=1.5),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.05)",
    ), row=1, col=1)
    # Donut
    moved      = len(breadth["movers"])
    not_moved  = len(breadth["non_movers"])
    fig.add_trace(go.Pie(
        labels=["Moved ≥3%", "Stable"],
        values=[moved, not_moved],
        hole=0.60,
        marker=dict(colors=[COLOR_DOWN if breadth["low_breadth"] else COLOR_UP, "#0f172a"]),
        textfont=dict(size=10, color="#94a3b8"),
        showlegend=False,
    ), row=1, col=2)
    fig.update_layout(**PLOT_LAYOUT, height=380,
                      title=dict(text="Index Layer  ·  NIFTY 50 + Market Breadth",
                                 font=dict(size=12, color="#64748b"), x=0, y=0.98))
    fig.update_layout(margin=dict(l=12, r=12, t=70, b=12))
    # Push subplot titles down so they don't collide with main title
    for ann in fig.layout.annotations:
        ann.update(y=ann.y - 0.08, font=dict(size=10, color="#94a3b8"))
    return fig


def chart_vix(vix_df: pd.DataFrame, vix_res: dict) -> go.Figure:
    """VIX line with rolling avg and spike threshold."""
    close   = vix_df["Close"]
    rolling = close.rolling(utils.VIX_ROLLING_WINDOW).mean()
    thresh  = rolling.dropna().iloc[-1] * utils.VIX_SPIKE_THRESHOLD if not rolling.dropna().empty else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vix_df.index, y=close,
        mode="lines", name="India VIX",
        line=dict(color=COLOR_AMB, width=1.5),
        fill="tozeroy", fillcolor="rgba(217,119,6,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=vix_df.index, y=rolling,
        mode="lines", name=f"MA-{utils.VIX_ROLLING_WINDOW}",
        line=dict(color="#64748b", width=1, dash="dot"),
    ))
    if thresh:
        fig.add_shape(
            type="line", x0=vix_df.index[0], x1=vix_df.index[-1],
            y0=thresh, y1=thresh,
            line=dict(color=COLOR_DOWN, width=1, dash="dash"),
        )
        fig.add_annotation(
            x=vix_df.index[-1], y=thresh, xanchor="right",
            yanchor="bottom", yshift=5,
            text=f"Spike threshold  {thresh:.1f}",
            showarrow=False, font=dict(size=9, color=COLOR_DOWN),
            bgcolor="rgba(220,38,38,0.08)", borderpad=3,
        )
    fig.update_layout(**PLOT_LAYOUT, height=340,
                      title=dict(text="Derivative Layer  ·  India VIX",
                                 font=dict(size=12, color="#64748b"), x=0, y=0.98))
    fig.update_layout(margin=dict(l=12, r=12, t=70, b=12))
    return fig


def chart_gauge(score: float, label: str) -> go.Figure:
    color = _score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(
            font=dict(size=52, family="JetBrains Mono", color=color),
            suffix="",
        ),
        title=dict(text="MANIPULATION SCORE", font=dict(size=9, color="#94a3b8",
                   family="Inter"), align="center"),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickvals=[0, 20, 40, 65, 100],
                ticktext=["0", "20", "40", "65", "100"],
                tickfont=dict(size=9, family="JetBrains Mono", color="#94a3b8"),
                tickcolor="#94a3b8",
            ),
            bar=dict(color=color, thickness=0.20),
            bgcolor="#0f1318",
            borderwidth=0,
            steps=[
                dict(range=[0,  40], color="rgba(5,150,105,0.06)"),
                dict(range=[40, 65], color="rgba(217,119,6,0.06)"),
                dict(range=[65,100], color="rgba(220,38,38,0.08)"),
            ],
            threshold=dict(
                line=dict(color=color, width=2),
                thickness=0.80, value=score,
            ),
        ),
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        margin=dict(l=24, r=24, t=40, b=0),
    )
    return fig


def chart_layer_bars(s_sig: float, i_sig: float, v_sig: float) -> go.Figure:
    layers = ["VIX Layer", "Index Layer", "Stock Layer"]
    vals   = [v_sig, i_sig, s_sig]
    colors = [_score_color(v) for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=layers,
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"  {v:.1f}" for v in vals],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color="#64748b"),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=190,
        title=dict(text="LAYER SIGNAL STRENGTH  ·  0 – 100",
                   font=dict(size=9, color="#94a3b8", family="Inter"), x=0),
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 120], gridcolor="#0f172a")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def chart_sentiment_bar(scores: list) -> go.Figure:
    """Horizontal bar of per-headline sentiment scores."""
    if not scores:
        return go.Figure()
    colors = [COLOR_UP if s > 0.05 else COLOR_DOWN if s < -0.05 else COLOR_DIM for s in scores]
    fig = go.Figure(go.Bar(
        x=scores,
        y=[f"H{i+1}" for i in range(len(scores))],
        orientation="h",
        marker=dict(color=colors, opacity=0.8),
        text=[f"{s:+.3f}" for s in scores],
        textfont=dict(family="JetBrains Mono", size=9, color="#64748b"),
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#94a3b8", line_width=1)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter", color="#64748b", size=11),
        margin=dict(l=12, r=12, t=36, b=12),
        height=max(200, len(scores) * 28),
        title=dict(text="HEADLINE SENTIMENT SCORES",
                   font=dict(size=9, color="#94a3b8"), x=0),
        xaxis=dict(
            range=[-1.1, 1.1], gridcolor="#0f172a",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1", "−0.5", "0", "+0.5", "+1"],
            showline=False, zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=10, color="#64748b"),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)", showline=False, zeroline=False,
            tickfont=dict(family="JetBrains Mono", size=10, color="#64748b"),
        ),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:16px 0 14px;border-bottom:1px solid {TH_BORDER};margin-bottom:16px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;color:{TH_BRAND_CLR};
                  letter-spacing:0.12em;font-weight:600;">📡 MARKET SURVEILLANCE</div>
      <div style="font-size:0.66rem;color:{TH_TEXT_MUTED};margin-top:3px;letter-spacing:0.06em;">
        MANIPULATION DETECTION TERMINAL v2.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    search_input = st.text_input(
        "SYMBOL OR COMPANY", value=utils.DEFAULT_TICKER,
        help="Search top 50 NSE stocks (e.g., 'TCS' or 'Tata')"
    ).strip()

    suggestions = stock_utils.get_stock_suggestions(search_input)
    
    if not suggestions:
        st.error("Invalid input. No matching stocks found in the NIFTY 50 universe.")
        
        # Project an error onto the main page as well so it is highly visible
        st.markdown(
            """<div style="padding:40px;text-align:center;">
                 <h3 style="color:#ef4444;font-family:'JetBrains Mono',monospace;">⚠️ INVALID TICKER</h3>
                 <p style="color:#94a3b8;">The stock you entered could not be found via fuzzy search.</p>
                 <p style="color:#94a3b8;">Please enter a valid company name (e.g. <b>Tata</b>) or ticker (e.g. <b>RELIANCE</b>) in the sidebar.</p>
               </div>""", unsafe_allow_html=True
        )
        st.stop()
        
    options = [f"{s['ticker']} - {s['name']}" for s in suggestions]
    selected_option = st.selectbox("SELECT MATCH", options=options, index=0)
    ticker = selected_option.split(" - ")[0].strip()

    period = st.selectbox(
        "PERIOD", options=["1mo", "3mo", "6mo", "1y"], index=1,
    )

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # Angel One Integration Removed (Phase 9 Rollback)

    newsapi_key = st.text_input(
        "NEWSAPI KEY (OPTIONAL)", type="password",
        help="Free key at newsapi.org — leave blank for Moneycontrol RSS",
    )
    
    news_query = st.text_input(
        "NEWS QUERY",
        value=f"{ticker.replace('.NS','').replace('.BO','')} stock India",
    )

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    demo_mode = st.checkbox("Simulate Anomaly (Stress Test ML)", value=False, help="Forces a massively volatile pseudo-day to trigger the Isolation Forest.")
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    
    run_btn = st.button("▶  RUN ANALYSIS", use_container_width=True, type="primary")

    if st.button("🔄 WIPE CACHE", use_container_width=True, help="Force a brand new data download, bypassing the 2-minute memory cache."):
        st.cache_data.clear()
        st.rerun()

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    with st.expander(f"NIFTY 50 BREADTH UNIVERSE  [{len(utils.NIFTY_CONSTITUENTS)} stocks]"):
        for c in utils.NIFTY_CONSTITUENTS:
            st.markdown(
                f'<div style="font-family:JetBrains Mono,monospace;font-size:0.70rem;'
                f'color:{TH_TEXT_DIM};padding:2px 0;">{c}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="margin-top:16px;font-size:0.66rem;color:{TH_TEXT_MUTED};'
        f'border-top:1px solid {TH_BORDER};padding-top:10px;">'
        'DATA: Yahoo Finance · Moneycontrol · NewsAPI<br>'
        'ANALYSIS: VADER · yfinance · Plotly'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL HEADER  (top bar)
# ─────────────────────────────────────────────────────────────────────────────

col_top1, col_top2 = st.columns([5, 1])
with col_top2:
    if st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode, key="theme_toggle"):
        if not st.session_state.dark_mode:
            st.session_state.dark_mode = True
            st.rerun()
    else:
        if st.session_state.dark_mode:
            st.session_state.dark_mode = False
            st.rerun()

now_str = datetime.now().strftime("%d %b %Y  %H:%M:%S IST")
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            border-bottom:1px solid {TH_BORDER};padding:14px 0 10px;margin-bottom:22px;">
  <div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:1.05rem;
                 font-weight:700;color:{TH_BRAND_CLR};letter-spacing:0.06em;">
      MARKET MANIPULATION DETECTOR
    </span>
    <span style="font-size:0.72rem;color:{TH_TEXT_MUTED};margin-left:14px;letter-spacing:0.04em;">
      MULTI-LAYER SURVEILLANCE SYSTEM
    </span>
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:{TH_TEXT_MUTED};">
    {now_str}
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LANDING STATE
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn:
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown(f"""
        <div style="padding:28px 0;">
          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;
                      color:{TH_TEXT_MUTED};text-transform:uppercase;margin-bottom:18px;">
            Detection Architecture
          </div>
        """, unsafe_allow_html=True)

        rows = [
            ("📊", "STOCK LAYER",      "40 %", "Price change % + Volume spike ratio",          "#059669"),
            ("📈", "INDEX LAYER",      "30 %", "NIFTY 50 breadth — % of stocks moving",        "#3b82f6"),
            ("📉", "VIX LAYER",        "30 %", "India VIX spike vs 20-day rolling average",    "#d97706"),
            ("📰", "NEWS CONTEXT",     "ADJ",  "VADER sentiment — reduces / raises confidence","#8b5cf6"),
        ]
        for icon, name, wt, desc, col in rows:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:14px;
                        padding:12px 16px;background:{TH_BG_CARD};border:1px solid {TH_BORDER};
                        border-radius:5px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,0.06);">
              <div style="font-size:1.2rem">{icon}</div>
              <div style="flex:1">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">
                  <span style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;
                                font-weight:700;color:{col};">{name}</span>
                  <span style="background:{TH_BADGE_BG};border:1px solid {TH_BORDER};
                               padding:1px 7px;border-radius:3px;font-size:0.65rem;
                               color:{TH_TEXT_DIM};font-family:'JetBrains Mono',monospace;">
                    {wt}</span>
                </div>
                <div style="font-size:0.78rem;color:{TH_TEXT_DIM};">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div style="padding:28px 0 0;">
          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;
                      color:{TH_TEXT_MUTED};text-transform:uppercase;margin-bottom:18px;">
            Score Classification
          </div>
        """, unsafe_allow_html=True)
        levels = [
            ("0 – 39",  "NORMAL",               "#059669", "All signals within expected range"),
            ("0 – 39",  "NEWS-DRIVEN MOVEMENT",  "#3b82f6", "Movement supported by news sentiment"),
            ("40 – 64", "SUSPICIOUS ACTIVITY",   "#d97706", "Multi-layer anomalies detected"),
            ("65 – 100","HIGH MANIPULATION RISK", "#dc2626", "Strong signals, no news justification"),
        ]
        for rng, lbl, col, desc in levels:
            st.markdown(f"""
            <div style="display:flex;gap:14px;align-items:flex-start;
                        padding:10px 14px;margin-bottom:8px;
                        border-left:3px solid {col};background:{TH_CTA_BG};border-radius:0 4px 4px 0;">
              <div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                             color:{col};font-weight:700;">{rng}</div>
                <div style="font-size:0.73rem;font-weight:700;color:{TH_TEXT_MUTED};
                             margin:2px 0;">{lbl}</div>
                <div style="font-size:0.72rem;color:{TH_TEXT_DIM};">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:30px;text-align:center;padding:18px;
                border:1px dashed {TH_BORDER};border-radius:6px;color:{TH_TEXT_MUTED};font-size:0.80rem;">
      Enter a ticker symbol in the sidebar and press
      <span style="color:{TH_BRAND_CLR};font-family:'JetBrains Mono',monospace;font-weight:700;">
        ▶ RUN ANALYSIS</span>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH  (cached 10 min)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=120)
def _cached_fetch(ticker, period, news_query, newsapi_key):
    stock_df        = data_fetch.fetch_stock_data(ticker, period)
    nifty_df        = data_fetch.fetch_nifty_data(period)
    constituent_dfs = data_fetch.fetch_index_data(utils.NIFTY_CONSTITUENTS, period)
    vix_df          = data_fetch.fetch_vix_data(period)
    headlines       = data_fetch.fetch_news_headlines(news_query, newsapi_key or None)
    indices_dfs     = data_fetch.fetch_major_indices_data(period)
    return stock_df, nifty_df, constituent_dfs, vix_df, headlines, indices_dfs

with st.spinner(f"Fetching  {ticker}  [{period}]  ·  NIFTY-50 breadth…"):
    stock_df, nifty_df, constituent_dfs, vix_df, headlines, indices_dfs = _cached_fetch(
        ticker, period, news_query, newsapi_key or ""
    )

if stock_df is None or stock_df.empty:
    st.error(f"No data returned for **{ticker}**. Verify the symbol (e.g. RELIANCE.NS).")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DETECTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Running detection pipeline…"):
    result = run_detection_pipeline(stock_df, nifty_df, constituent_dfs, vix_df, headlines, demo_mode=demo_mode)

score        = result["score"]
label        = result["label"]
explanation  = result["explanation"]
sr           = result["stock_result"]
ir           = result["index_result"]
vr           = result["vix_result"]
nr           = result["news_result"]

# ─────────────────────────────────────────────────────────────────────────────
# ALERT STRIP
# ─────────────────────────────────────────────────────────────────────────────
msg_map = {
    utils.LABEL_NORMAL:
        f"{ticker}  —  All signals within normal parameters. No manipulation detected.",
    utils.LABEL_NEWS_DRIVEN:
        f"{ticker}  —  Price movement corroborated by news sentiment. Low manipulation risk.",
    utils.LABEL_SUSPICIOUS:
        f"{ticker}  —  Suspicious multi-layer anomalies detected. Recommend enhanced monitoring.",
    utils.LABEL_HIGH_RISK:
        f"{ticker}  —  ELEVATED RISK: Strong multi-layer anomaly signals with no news justification.",
}
_alert(label, msg_map[label])

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    _kpi("Manip. Score", f"{score:.1f}",
         sub=f"/ 100", color=_score_color(score))
with k2:
    c = COLOR_DOWN if abs(sr["price_change_pct"]) >= utils.PRICE_CHANGE_THRESHOLD else COLOR_UP
    _kpi("Price Δ", f"{sr['price_change_pct']:+.2f}%",
         sub=f"{utils.PRICE_CHANGE_WINDOW}d change", color=c)
with k3:
    c = COLOR_DOWN if sr["volume_spike"] >= utils.VOLUME_SPIKE_THRESHOLD else COLOR_UP
    _kpi("Vol. Spike", f"{sr['volume_spike']:.2f}×",
         sub=f"vs {utils.VOLUME_ROLLING_WINDOW}d avg", color=c)
with k4:
    c = COLOR_DOWN if vr["spike_flag"] else COLOR_UP
    _kpi("India VIX", f"{vr['current_vix']:.2f}",
         sub=f"avg {vr['rolling_avg']:.2f}  ×{vr['vix_spike']:.2f}", color=c)
with k5:
    bpct = ir["breadth_ratio"] * 100
    c = COLOR_DOWN if ir["low_breadth"] else COLOR_UP
    _kpi("Breadth", f"{bpct:.0f}%",
         sub=f"{len(ir['movers'])} / {len(ir['movers'])+len(ir['non_movers'])} moved", color=c)
with k6:
    sc = {"Positive": COLOR_UP, "Negative": COLOR_DOWN, "Neutral": COLOR_DIM}.get(
        nr["overall_label"], COLOR_DIM
    )
    _kpi("News Sent.", nr["overall_label"],
         sub=f"score {nr['mean_score']:+.3f}", color=sc)

st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t_markets, t_stock, t_index, t_vix, t_news, t_summary = st.tabs([
    "MARKETS", "STOCK", "INDEX", "VIX", "NEWS", "SUMMARY",
])

# ── MARKETS ───────────────────────────────────────────────────────────────
with t_markets:
    _section("MAJOR INDICES OVERVIEW")
    cols = st.columns(len(utils.MAJOR_INDICES))
    for i, (name, df_idx) in enumerate(indices_dfs.items()):
        with cols[i]:
            if df_idx is not None and not df_idx.empty:
                curr = df_idx["Close"].iloc[-1]
                prev = df_idx["Close"].iloc[-2] if len(df_idx) > 1 else curr
                pct = ((curr - prev) / prev) * 100
                c = COLOR_UP if pct >= 0 else COLOR_DOWN
                _kpi(name, f"{curr:,.2f}", sub=f"{pct:+.2f}%", color=c)
            else:
                _kpi(name, "N/A", sub="-")

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    
    fig_idx = go.Figure()
    for name, df_idx in indices_dfs.items():
        if df_idx is not None and not df_idx.empty:
            norm = df_idx["Close"] / df_idx["Close"].iloc[0] * 100
            fig_idx.add_trace(go.Scatter(
                x=df_idx.index, y=norm, mode="lines", name=name, line=dict(width=1.5)
            ))
    fig_idx.update_layout(
         **PLOT_LAYOUT, 
         height=400, 
         title=dict(text="RELATIVE PERFORMANCE (BASE 100)", font=dict(size=11, color="#64748b"), x=0, y=0.98),
    )
    fig_idx.update_layout(margin=dict(l=12, r=12, t=70, b=12))
    fig_idx.update_layout(legend=dict(orientation="h", x=0, y=1.12, yanchor="bottom", font=dict(size=10, color="#64748b")))

    # Clear redundant axes if needed, similar to layer_bars
    fig_idx.update_xaxes(showline=False, zeroline=False, gridcolor=TH_BORDER)
    fig_idx.update_yaxes(showline=False, zeroline=False, gridcolor=TH_BORDER)
    st.plotly_chart(fig_idx, use_container_width=True)

# ── STOCK ─────────────────────────────────────────────────────────────────
with t_stock:
    st.plotly_chart(chart_ohlcv(stock_df, ticker), use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        _section("STOCK LAYER  ·  SIGNAL BREAKDOWN")
        rows_s = [
            ("Price Score",   f"{sr['price_score']:.1f}", "/ 50"),
            ("Volume Score",  f"{sr['volume_score']:.1f}", "/ 50"),
            ("Stock Signal",  f"{sr['stock_signal']:.1f}", "/ 100"),
        ]
        for lbl, val, sub in rows_s:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'padding:8px 0;border-bottom:1px solid {TH_BORDER};">'
                f'<span style="font-size:0.78rem;color:{TH_TEXT_DIM};">{lbl}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:1.0rem;color:{TH_TEXT};">'
                f'{val}<span style="font-size:0.70rem;color:{TH_TEXT_MUTED};margin-left:4px;">{sub}</span></span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        if sr["anomaly_flags"]:
            for f in sr["anomaly_flags"]:
                st.markdown(
                    f'<div style="background:rgba(217,119,6,0.07);border-left:2px solid #d97706;'
                    f'padding:8px 12px;border-radius:0 4px 4px 0;font-size:0.78rem;color:#d97706;'
                    f'margin-bottom:6px;">▲ {f}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#059669;font-size:0.78rem;padding:8px 0;">● No anomalies detected</div>',
                unsafe_allow_html=True,
            )
    with cb:
        _section("RECENT OHLCV")
        d = stock_df[["Open","High","Low","Close","Volume"]].tail(12).copy()
        d.index = d.index.strftime("%d %b")
        st.dataframe(
            d.style
            .format({"Open":"₹{:.2f}","High":"₹{:.2f}","Low":"₹{:.2f}",
                     "Close":"₹{:.2f}","Volume":"{:,.0f}"})
            .set_properties(**{"font-family":"JetBrains Mono","font-size":"0.78rem"}),
            use_container_width=True, height=340,
        )


# ── INDEX ─────────────────────────────────────────────────────────────────
with t_index:
    st.plotly_chart(chart_nifty(nifty_df, ir), use_container_width=True)

    cc, cd = st.columns(2)
    with cc:
        _section("BREADTH  ·  METRICS")
        rows_i = [
            ("Breadth Ratio",    f"{ir['breadth_ratio']*100:.0f}%",  "pct of NIFTY 50 moved"),
            ("Stocks Moved",     f"{len(ir['movers'])}",              f"of {len(ir['movers'])+len(ir['non_movers'])} total"),
            ("Index Signal",     f"{ir['index_signal']:.1f}",         "/ 100"),
            ("Low Breadth Flag", "YES" if ir["low_breadth"] else "NO","< 30% threshold"),
        ]
        for lbl, val, sub in rows_i:
            c = COLOR_DOWN if (lbl == "Low Breadth Flag" and val == "YES") else TH_TEXT
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'padding:8px 0;border-bottom:1px solid {TH_BORDER};">'
                f'<span style="font-size:0.78rem;color:{TH_TEXT_DIM};">{lbl}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:1.0rem;color:{c};">'
                f'{val}<span style="font-size:0.70rem;color:{TH_TEXT_MUTED};margin-left:6px;">{sub}</span></span>'
                f'</div>', unsafe_allow_html=True,
            )
        if ir["anomaly_flags"]:
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            for f in ir["anomaly_flags"]:
                st.markdown(
                    f'<div style="background:rgba(217,119,6,0.07);border-left:2px solid #d97706;'
                    f'padding:8px 12px;border-radius:0 4px 4px 0;font-size:0.78rem;color:#d97706;'
                    f'margin-bottom:6px;">▲ {f}</div>',
                    unsafe_allow_html=True,
                )

    with cd:
        _section("CONSTITUENT DETAIL")
        from feature_engineering import compute_price_change
        rows_c = []
        for t_sym, df_c in constituent_dfs.items():
            if df_c is not None and not df_c.empty:
                chg   = compute_price_change(df_c, window=1)
                moved = abs(chg) >= utils.PRICE_CHANGE_THRESHOLD
                rows_c.append({
                    "Ticker":  t_sym.replace(".NS",""),
                    "1d Chg":  f"{chg:+.2f}%",
                    "Moved":   "✓" if moved else "–",
                })
        st.dataframe(
            pd.DataFrame(rows_c),
            use_container_width=True, hide_index=True, height=380,
        )


# ── VIX ──────────────────────────────────────────────────────────────────
with t_vix:
    st.plotly_chart(chart_vix(vix_df, vr), use_container_width=True)

    ce, cf = st.columns(2)
    with ce:
        _section("VIX  ·  METRICS")
        rows_v = [
            ("Current VIX",  f"{vr['current_vix']:.2f}", ""),
            (f"{utils.VIX_ROLLING_WINDOW}d Rolling Avg", f"{vr['rolling_avg']:.2f}", ""),
            ("Spike Ratio",  f"{vr['vix_spike']:.2f}×",  f"threshold {utils.VIX_SPIKE_THRESHOLD}×"),
            ("VIX Signal",   f"{vr['vix_signal']:.1f}",  "/ 100"),
            ("Spike Flag",   "YES" if vr["spike_flag"] else "NO", ""),
        ]
        for lbl, val, sub in rows_v:
            c = COLOR_DOWN if (lbl == "Spike Flag" and val == "YES") else TH_TEXT
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'padding:8px 0;border-bottom:1px solid {TH_BORDER};">'
                f'<span style="font-size:0.78rem;color:{TH_TEXT_DIM};">{lbl}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-size:1.0rem;color:{c};">'
                f'{val}<span style="font-size:0.70rem;color:{TH_TEXT_MUTED};margin-left:6px;">{sub}</span></span>'
                f'</div>', unsafe_allow_html=True,
            )
        if vr["spike_flag"]:
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:rgba(220,38,38,0.07);border-left:2px solid #dc2626;'
                f'padding:8px 12px;border-radius:0 4px 4px 0;font-size:0.78rem;color:#dc2626;">'
                f'● VIX spike confirmed — elevated systemic volatility</div>',
                unsafe_allow_html=True,
            )
    with cf:
        _section("RECENT VIX VALUES")
        vd = vix_df.tail(15).copy()
        vd.index = vd.index.strftime("%d %b")
        vd.columns = ["VIX"]
        st.dataframe(
            vd.style.format({"VIX":"{:.2f}"})
              .set_properties(**{"font-family":"JetBrains Mono","font-size":"0.78rem"}),
            use_container_width=True, height=340,
        )


# ── NEWS ─────────────────────────────────────────────────────────────────
with t_news:
    st.markdown(
        f'<div style="display:flex;gap:16px;align-items:center;margin-bottom:16px;">'
        f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.14em;'
        f'color:{TH_TEXT_MUTED};text-transform:uppercase;">News Context</div>'
        f'<div>{_sent_pill(nr["overall_label"])}</div>'
        f'<div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:{TH_TEXT_DIM};">'
        f'mean {nr["mean_score"]:+.3f}'
        f'&nbsp;&nbsp;strong signal: {"Yes" if nr["strong_signal"] else "No"}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    ca2, cb2 = st.columns([1.3, 1])
    with ca2:
        for i, hl in enumerate(headlines):
            sent  = nr["labels"][i] if i < len(nr["labels"]) else "Neutral"
            sc_v  = nr["scores"][i] if i < len(nr["scores"]) else 0.0
            title = hl.get("title","")
            src   = hl.get("source","")
            url   = hl.get("url","")
            link  = f'<a href="{url}" target="_blank" style="color:{TH_TEXT_MUTED};text-decoration:none;">{title}</a>' if url else f'<span style="color:{TH_TEXT_MUTED};">{title}</span>'
            st.markdown(
                f'<div class="news-item">'
                f'{_sent_pill(sent)}'
                f'<div><div class="news-title">{link}</div>'
                f'<div class="news-meta">{src}&nbsp;&nbsp;|&nbsp;&nbsp;sentiment {sc_v:+.3f}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    with cb2:
        st.plotly_chart(chart_sentiment_bar(nr["scores"]), use_container_width=True)


# ── SUMMARY ──────────────────────────────────────────────────────────────
with t_summary:
    col_g, col_b, col_e = st.columns([1, 1, 1.4])

    with col_g:
        st.plotly_chart(chart_gauge(score, label), use_container_width=True)
        st.markdown(
            f'<div style="text-align:center;margin-top:-8px;">{_pill(label)}</div>',
            unsafe_allow_html=True,
        )

    with col_b:
        st.plotly_chart(chart_layer_bars(
            sr["stock_signal"], ir["index_signal"], vr["vix_signal"]
        ), use_container_width=True)

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        _section("WEIGHT SCHEDULE")
        for ly, wt in [("Stock Layer","40 %"),("Index Layer","30 %"),("VIX Layer","30 %")]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                f'border-bottom:1px solid {TH_BORDER};font-size:0.76rem;">'
                f'<span style="color:{TH_TEXT_DIM};">{ly}</span>'
                f'<span style="font-family:JetBrains Mono,monospace;color:{TH_TEXT_MUTED};">{wt}</span>'
                f'</div>', unsafe_allow_html=True,
            )
        news_adj = "−25%  (aligned news)" if nr["strong_signal"] and nr["mean_score"] > 0 and sr["price_change_pct"] > 0 else "+15%  (weak/no news)"
        st.markdown(
            f'<div style="padding:5px 0;font-size:0.72rem;">'
            f'<span style="color:{TH_TEXT_MUTED};">News adjustment:</span>'
            f'<span style="font-family:JetBrains Mono,monospace;color:{TH_TEXT_DIM};margin-left:6px;">'
            f'{news_adj}</span></div>',
            unsafe_allow_html=True,
        )

    with col_e:
        _section("DETECTION REPORT")
        st.markdown(
            f'<div class="expl-panel">{explanation.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="margin-top:16px;font-family:JetBrains Mono,monospace;'
        f'font-size:0.68rem;color:{TH_TEXT_MUTED};text-align:right;">'
        f'GENERATED  {datetime.now().strftime("%Y-%m-%d  %H:%M:%S")} IST'
        f'  ·  MARKET SURVEILLANCE TERMINAL v2.0'
        f'</div>',
        unsafe_allow_html=True,
    )
