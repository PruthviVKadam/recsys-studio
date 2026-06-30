"""Shared brand polish for the Streamlit portfolio apps.

Accent-agnostic: the per-app `.streamlit/config.toml` sets `primaryColor` (Streamlit applies it
to buttons / sliders / tabs / metric deltas automatically); this just hides Streamlit's default
chrome, loads the display font, and rounds + tightens the layout so the app reads like a product
rather than a default Streamlit page. Call `apply_brand()` once, right after `set_page_config`.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&display=swap');

/* hide Streamlit's default chrome (menu, toolbar/deploy button, footer, header bar) */
#MainMenu, footer {visibility: hidden;}
[data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {display: none !important;}
header[data-testid="stHeader"] {background: transparent;}

/* display type */
h1, h2, h3, h4 {font-family: 'Space Grotesk', ui-sans-serif, system-ui, sans-serif; letter-spacing: -0.02em;}

/* tighter, centered content column */
.block-container {padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px;}

/* metric cards: bordered, rounded, subtle surface */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.09);
  border-radius: 14px; padding: 14px 16px;
}
[data-testid="stMetricValue"] {font-family: 'Space Grotesk', sans-serif; letter-spacing: -0.02em;}

/* pill buttons */
.stButton > button, .stDownloadButton > button {border-radius: 999px; font-weight: 600;}

/* tabs */
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {border-radius: 10px 10px 0 0; padding: 8px 14px;}

/* rounded tables / dataframes */
[data-testid="stTable"], .stDataFrame, [data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}

/* expanders + alerts a touch rounder */
[data-testid="stExpander"], [data-testid="stAlert"] {border-radius: 12px;}
</style>
"""


def apply_brand() -> None:
    """Inject the shared brand CSS. Safe to call once per run after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)
