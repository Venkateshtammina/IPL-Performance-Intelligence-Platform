import os
import sys
import snowflake.connector
import pandas as pd
import streamlit as st
import pickle
import numpy as np
import altair as alt
from dotenv import load_dotenv
from src import analytics as analytics_module
from src.feature_context import (
    build_event_feature_frame,
    build_win_feature_frame,
    canonicalize_venue_name,
)
from src.probability import (
    blend_mean_with_matchup_evidence,
    blend_model_with_matchup_evidence,
    estimate_high_pressure_from_history,
    estimate_chase_win_percentage,
    matchup_confidence_label,
)
from src.strategy import (
    optimize_bowling_plan,
    simulate_dynamic_over,
    simulate_over_outcomes,
)
from src.model_monitoring import assess_model_drift
from src.plan_cache import load_cached_plan, store_cached_plan
from src.plan_export import build_plan_csv, build_plan_pdf

# Models trained by older script runs referenced this module as ``analytics``.
sys.modules.setdefault("analytics", analytics_module)

# Initialize application configuration mapping
load_dotenv()

# Set premium dashboard layout configuration
st.set_page_config(
    page_title="Elite IPL Strategy Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Design tokens: "Dark Operations Center" — fully adapted for a premium
# dark slate canvas. Replaces warm light colors with deep muted panel tones
# (#222A3B) and sharp high-contrast text for maximum legibility in low light.
# ---------------------------------------------------------------------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg-canvas: #F3F7F2; /* Soft sage app canvas */
        --bg-panel: rgba(251, 253, 248, 0.88);  /* Frosted warm panel */
        --bg-panel-2: rgba(244, 248, 239, 0.94); /* Elevated soft surface */
        --bg-panel-3: #FBFDF8; /* Input surface */
        --border-hair: rgba(15, 23, 42, 0.10); /* Soft light border */
        --border-hair-soft: rgba(15, 23, 42, 0.06);
        --text-primary: #0F172A; /* Slate ink */
        --text-secondary: #334155; /* Slate metadata */
        --text-muted: #64748B;
        --floodlight: #6D5DFB;
        --floodlight-dim: rgba(109, 93, 251, 0.12);
        --boundary: #0891B2;
        --boundary-dim: rgba(8, 145, 178, 0.12);
        --wicket: #F43F5E;
        --wicket-dim: rgba(244, 63, 94, 0.12);
        --shadow-card: 0 16px 42px rgba(15, 23, 42, 0.08);
        --shadow-card-hover: 0 22px 60px rgba(15, 23, 42, 0.14);
        --amber: #D97706;
        --amber-dim: rgba(217, 119, 6, 0.12);
    }

    html, body, .stApp {
        background: var(--bg-canvas) !important;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        --primary-color: #0F766E !important;
        --primary-color-hover: #0E7490 !important;
        --primary-color-active: #115E59 !important;
        --secondary-background-color: #FBFDF8 !important;
        --text-color: #0F172A !important;
    }

    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #22D3EE 0%, #7C3AED 48%, #FB7185 100%);
        z-index: 999;
    }

    .stApp {
        background:
            radial-gradient(circle at 14% 8%, rgba(109,93,251,0.16), transparent 30%),
            radial-gradient(circle at 84% 10%, rgba(8,145,178,0.13), transparent 28%),
            radial-gradient(circle at 74% 82%, rgba(244,63,94,0.08), transparent 32%),
            linear-gradient(135deg, #FBFDF8 0%, var(--bg-canvas) 50%, #EAF4EC 100%) !important;
    }

    /* Numeric / scoreboard figures */
    .stat-mono { font-family: 'JetBrains Mono', monospace; }

    /* ---------- Sidebar ---------- */
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(251,253,248,0.96), rgba(237,247,232,0.96));
        border-right: 1px solid var(--border-hair);
    }

    /* Add breathing room around the whole app */
    .block-container { padding-top: 1.6rem; }

    /* ---------- Hero header ---------- */
    .hero-wrap {
        position: relative;
        background:
            linear-gradient(135deg, rgba(251,253,248,0.95), rgba(235,247,239,0.90)),
            radial-gradient(circle at 18% 18%, rgba(8,145,178,0.14), transparent 32%),
            radial-gradient(circle at 82% 20%, rgba(109,93,251,0.20), transparent 38%);
        border: 1px solid var(--border-hair);
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 26px;
        box-shadow: var(--shadow-card);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 20px;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: "";
        position: absolute;
        top: 0; right: 0;
        width: 340px; height: 100%;
        background: radial-gradient(ellipse at top right, rgba(34, 211, 238, 0.16) 0%, rgba(34, 211, 238, 0) 70%);
        pointer-events: none;
    }
    .hero-wrap::after {
        content: "";
        position: absolute;
        inset: auto -40px -72px auto;
        width: 260px;
        height: 180px;
        border-radius: 50%;
        border: 1px solid rgba(109,93,251,0.14);
        box-shadow: inset 0 0 45px rgba(109,93,251,0.16);
        transform: rotate(-12deg);
        pointer-events: none;
    }
    .hero-left { display: flex; align-items: center; gap: 16px; position: relative; z-index: 1; }
    .hero-badge {
        width: 52px; height: 52px;
        border-radius: 14px;
        background: linear-gradient(145deg, #22D3EE 0%, #7C3AED 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 24px;
        color: #FFFFFF;
        box-shadow: 0 12px 30px rgba(109,93,251,0.22);
        flex-shrink: 0;
    }
    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 24px;
        letter-spacing: -0.01em;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.25;
    }
    .hero-sub {
        color: var(--text-secondary);
        font-size: 13.5px;
        margin: 3px 0 0 0;
        letter-spacing: 0.01em;
    }
    .hero-right { display: flex; align-items: center; gap: 10px; position: relative; z-index: 1; flex-shrink: 0; }
    .session-chip {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        font-size: 11px;
        color: var(--text-muted);
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-weight: 600;
        padding-right: 14px;
        border-right: 1px solid var(--border-hair);
    }
    .session-chip span { color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-size: 13px; letter-spacing: 0; text-transform: none; margin-top: 2px; }
    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #FB7185;
        background: var(--wicket-dim);
        border: 1px solid rgba(251,113,133,0.32);
        padding: 6px 12px 6px 9px;
        border-radius: 999px;
    }
    .live-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #FB7185;
        box-shadow: 0 0 0 0 rgba(251,113,133,0.45);
        animation: pulse-dot 1.8s infinite;
    }
    @keyframes pulse-dot {
        0%   { box-shadow: 0 0 0 0 rgba(251,113,133,0.4); }
        70%  { box-shadow: 0 0 0 7px rgba(251,113,133,0); }
        100% { box-shadow: 0 0 0 0 rgba(251,113,133,0); }
    }
    .hr-fade {
        height: 1px;
        border: none;
        margin: 18px 0 22px 0;
        background: linear-gradient(90deg, var(--border-hair) 0%, var(--border-hair-soft) 60%, transparent 100%);
    }

    /* ---------- Section labels ---------- */
    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 13px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-secondary);
        margin: 4px 0 14px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label .tick { width: 3px; height: 14px; border-radius: 2px; background: var(--floodlight); display: inline-block; }

    /* ---------- Config panel ---------- */
    .config-panel {
        background: var(--bg-panel);
        border: 1px solid var(--border-hair);
        border-radius: 16px;
        padding: 18px 18px 4px 18px;
        box-shadow: var(--shadow-card);
    }
    .config-panel [data-testid="stVerticalBlock"] {
        gap: 0.65rem !important;
    }

    .mini-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 18px;
    }
    .mini-kpi {
        background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(248,250,252,0.92));
        border: 1px solid var(--border-hair);
        border-radius: 14px;
        padding: 13px 14px;
        box-shadow: var(--shadow-card);
    }
    .mini-kpi span {
        display: block;
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    .mini-kpi b {
        display: block;
        margin-top: 5px;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-primary);
        font-size: 18px;
    }

    /* ---------- Metric cards ---------- */
    .stat-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-hair);
        border-left-width: 3px;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 16px;
        box-shadow: var(--shadow-card);
        transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-card-hover); }
    .stat-card.tone-floodlight { border-left-color: var(--floodlight); }
    .stat-card.tone-boundary   { border-left-color: var(--boundary); }
    .stat-card.tone-wicket     { border-left-color: var(--wicket); }
    .stat-card.tone-amber      { border-left-color: var(--amber); }

    .stat-card-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }
    .stat-card-title {
        font-size: 12.5px;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .stat-card-badge {
        width: 26px; height: 26px;
        display: flex; align-items: center; justify-content: center;
        font-size: 12px;
        border-radius: 8px;
        flex-shrink: 0;
    }
    .stat-card.tone-floodlight .stat-card-badge { background: var(--floodlight-dim); }
    .stat-card.tone-wicket .stat-card-badge { background: var(--wicket-dim); }
    .stat-card.tone-amber .stat-card-badge { background: var(--amber-dim); }
    
    .stat-card-value {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 26px;
        color: var(--text-primary);
        margin: 2px 0 6px 0;
    }
    .stat-card-foot {
        font-size: 12.5px;
        color: var(--text-muted);
        letter-spacing: 0.01em;
    }
    .stat-card-foot b { color: var(--text-secondary); font-family: 'JetBrains Mono', monospace; font-weight: 600; }

    /* Forecast tiles */
    .forecast-tile {
        background: var(--bg-panel);
        border: 1px solid var(--border-hair);
        border-left-width: 3px;
        border-radius: 14px;
        padding: 16px 18px;
        text-align: left;
        box-shadow: var(--shadow-card);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .forecast-tile:hover { transform: translateY(-2px); box-shadow: var(--shadow-card-hover); }
    .forecast-tile .fhead { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
    .forecast-tile .fdot { width: 8px; height: 8px; border-radius: 50%; }
    .forecast-tile .flabel { font-size: 12px; font-weight: 500; color: var(--text-secondary); letter-spacing: 0.02em; }
    .forecast-tile .fvalue {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 26px;
    }
    .forecast-tile.f-neutral  { border-left-color: var(--floodlight); }
    .forecast-tile.f-boundary { border-left-color: var(--boundary); }
    .forecast-tile.f-wicket   { border-left-color: var(--wicket); }
    .forecast-tile.f-neutral .fdot   { background: var(--floodlight); }
    .forecast-tile.f-boundary .fdot  { background: var(--boundary); }
    .forecast-tile.f-wicket .fdot    { background: var(--wicket); }
    .forecast-tile.f-neutral .fvalue  { color: var(--floodlight); }
    .forecast-tile.f-boundary .fvalue { color: var(--boundary); }
    .forecast-tile.f-wicket .fvalue   { color: var(--wicket); }

    .chart-shell {
        background: var(--bg-panel);
        border: 1px solid var(--border-hair);
        border-radius: 16px;
        padding: 16px;
        margin: 6px 0 18px 0;
        box-shadow: var(--shadow-card);
    }

    /* ---------- Win probability bar ---------- */
    .wp-wrap {
        margin-top: 4px; margin-bottom: 22px;
        background: var(--bg-panel);
        border: 1px solid var(--border-hair);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: var(--shadow-card);
    }
    .wp-labels { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }
    .wp-labels .side { display: flex; align-items: center; gap: 7px; font-size: 12.5px; font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase; }
    .wp-labels .dotmark { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .wp-labels .chasing { color: var(--boundary); }
    .wp-labels .chasing .dotmark { background: var(--boundary); }
    .wp-labels .defending { color: var(--wicket); }
    .wp-labels .defending .dotmark { background: var(--wicket); }
    .wp-labels .val { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 700; color: var(--text-primary); margin-left: 6px; }
    .wp-track {
        position: relative;
        height: 10px;
        border-radius: 999px;
        background: var(--bg-panel-3);
        overflow: hidden;
    }
    .wp-fill {
        position: absolute; left: 0; top: 0; bottom: 0;
        background: linear-gradient(90deg, #22D3EE 0%, #A78BFA 52%, #FB7185 100%);
        border-radius: 999px;
    }

    /* ---------- Streamlit input overrides ---------- */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSlider"] { color: var(--text-primary) !important; font-weight: 500; }
    
    /* Force readable text inside light selection controls */
    div[data-testid="stSelectbox"] * {
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
    }
    
    /* Tabs Overrides */
    .stTabs [data-baseweb="tab-list"] button,
    .stTabs [data-baseweb="tab-list"] button *,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] div,
    .stTabs [data-baseweb="tab"] span {
        color: var(--text-secondary) !important;
        -webkit-text-fill-color: var(--text-secondary) !important;
        font-weight: 600 !important;
    }

    /* Popover menu options contrast alignments */
    div[data-baseweb="popover"] *,
    ul[role="listbox"] * {
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
        background-color: var(--bg-panel-2) !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: var(--bg-panel-3) !important;
        border: 1px solid var(--border-hair) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
        min-height: 44px !important;
        padding-right: 76px !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06) !important;
    }
    div[data-testid="stNumberInput"] > div {
        position: relative !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: rgba(44,191,174,0.42) !important;
        box-shadow: 0 0 0 4px rgba(44,191,174,0.12), 0 10px 26px rgba(15,23,42,0.06) !important;
    }
    /* Both increment and decrement buttons, styled as matching pill controls */
    div[data-testid="stNumberInputStepUp"],
    div[data-testid="stNumberInputStepDown"] {
        display: flex !important;
        align-items: center;
        justify-content: center;
        position: absolute !important;
        top: 50%;
        transform: translateY(-50%);
        width: 24px;
        height: 24px;
        border-radius: 999px !important;
        background: linear-gradient(135deg, #EAF8F3, #EAF2FF) !important;
        border: 1px solid rgba(44,191,174,0.20) !important;
        box-shadow: 0 6px 14px rgba(15,23,42,0.06);
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        z-index: 2;
    }
    div[data-testid="stNumberInputStepUp"] {
        right: 14px;
    }
    div[data-testid="stNumberInputStepDown"] {
        right: 44px;
    }
    div[data-testid="stNumberInputStepUp"]:hover,
    div[data-testid="stNumberInputStepDown"]:hover {
        background: linear-gradient(135deg, #DDF7EE, #DBEAFE) !important;
        transform: translateY(-50%) scale(1.06);
    }
    div[data-testid="stNumberInputStepUp"] svg,
    div[data-testid="stNumberInputStepDown"] svg {
        fill: #0F766E !important;
        width: 14px;
        height: 14px;
    }
    div[data-testid="stSelectbox"] > div > div {
        background-color: var(--bg-panel-3) !important;
        border: 1px solid var(--border-hair) !important;
        border-radius: 12px !important;
        min-height: 44px !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06) !important;
    }
    div[data-testid="stSelectbox"] > div > div:focus-within {
        border-color: var(--floodlight) !important;
        box-shadow: 0 0 0 3px rgba(109,93,251,0.16) !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background: linear-gradient(135deg, #0EA5E9, #6D5DFB) !important;
        border: 2px solid #FBFDF8 !important;
        box-shadow: 0 6px 16px rgba(109,93,251,0.24) !important;
    }
    div[data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(90deg, #0EA5E9, #6D5DFB) !important;
    }
    div[data-testid="stSlider"] div[data-testid="stTickBar"] {
        display: none !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div {
        background-color: #E2E8F0 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 0.45rem !important;
        padding-bottom: 1rem !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-size: 12px !important;
        color: #1E293B !important;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-weight: 700;
    }

    /* ---------- Dataframe ---------- */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border-hair);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-card);
    }

    /* ---------- Typography ---------- */
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.02em; color: var(--text-primary) !important; }

    /* ---------- Tabs (segmented controls layout framework) ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(251, 253, 248, 0.82);
        border: 1px solid var(--border-hair);
        border-radius: 12px;
        padding: 4px;
        width: fit-content;
        margin-bottom: 8px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
    }
    .stTabs [data-baseweb="tab-list"] button {
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(251, 253, 248, 0.88) !important;
        border-radius: 9px;
        padding: 9px 20px;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
        border-bottom: 1px solid rgba(15, 23, 42, 0.08) !important;
        font-size: 13.5px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="false"] *,
    .stTabs [data-baseweb="tab"][aria-selected="false"] p,
    .stTabs [data-baseweb="tab"][aria-selected="false"] span,
    .stTabs [data-baseweb="tab"][aria-selected="false"] div,
    .stTabs button[aria-selected="false"] *,
    .stTabs button[aria-selected="false"] p,
    .stTabs button[aria-selected="false"] span,
    .stTabs button[aria-selected="false"] div {
        color: #1E293B !important;
        -webkit-text-fill-color: #1E293B !important;
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(224, 242, 254, 0.95) !important;
        border-color: rgba(14, 165, 233, 0.32) !important;
        color: #075985 !important;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab"]:hover *,
    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span,
    .stTabs [data-baseweb="tab"]:hover div {
        color: #075985 !important;
        -webkit-text-fill-color: #075985 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0EA5E9 0%, #6D5DFB 100%) !important;
        border-color: rgba(109, 93, 251, 0.48) !important;
        border-bottom-color: rgba(109, 93, 251, 0.48) !important;
        box-shadow: 0 10px 24px rgba(109, 93, 251, 0.26);
    }
    .stTabs button[aria-selected="true"] {
        background: linear-gradient(135deg, #0EA5E9 0%, #6D5DFB 100%) !important;
        border-color: rgba(109, 93, 251, 0.48) !important;
        border-bottom-color: rgba(109, 93, 251, 0.48) !important;
    }
    .stTabs button[aria-selected="false"] {
        background-color: rgba(251, 253, 248, 0.92) !important;
        border-color: rgba(15, 23, 42, 0.08) !important;
        border-bottom-color: rgba(15, 23, 42, 0.08) !important;
    }
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(135deg, #0284C7 0%, #5B4BEA 100%) !important;
        border-color: rgba(91, 75, 234, 0.55) !important;
    }
    .stTabs [aria-selected="true"] *,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [aria-selected="true"] div,
    .stTabs button[aria-selected="true"] *,
    .stTabs button[aria-selected="true"] p,
    .stTabs button[aria-selected="true"] span,
    .stTabs button[aria-selected="true"] div {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ---------- Final tab + slider color overrides ---------- */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(251, 253, 248, 0.86) !important;
        border-color: rgba(15, 23, 42, 0.08) !important;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06) !important;
    }
    .stTabs button[aria-selected="false"],
    .stTabs [data-baseweb="tab"][aria-selected="false"] {
        background: rgba(251, 253, 248, 0.94) !important;
        border-color: rgba(15, 23, 42, 0.08) !important;
    }
    .stTabs button[aria-selected="false"] *,
    .stTabs [data-baseweb="tab"][aria-selected="false"] * {
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
        opacity: 1 !important;
    }
    .stTabs button[aria-selected="false"]:hover,
    .stTabs [data-baseweb="tab"][aria-selected="false"]:hover {
        background: #ECFEFF !important;
        border-color: rgba(14, 116, 144, 0.24) !important;
    }
    .stTabs button[aria-selected="false"]:hover *,
    .stTabs [data-baseweb="tab"][aria-selected="false"]:hover * {
        color: #0E7490 !important;
        -webkit-text-fill-color: #0E7490 !important;
    }
    .stTabs button[aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%) !important;
        border-color: rgba(15, 118, 110, 0.34) !important;
        border-bottom-color: rgba(15, 118, 110, 0.34) !important;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.20) !important;
    }
    .stTabs button[aria-selected="true"] *,
    .stTabs [data-baseweb="tab"][aria-selected="true"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 0.6rem !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div {
        background: #E2E8F0 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: #E2E8F0 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #14B8A6 0%, #2563EB 100%) !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background: #0F766E !important;
        border: 3px solid #FBFDF8 !important;
        box-shadow: 0 6px 16px rgba(15, 118, 110, 0.28) !important;
    }
    div[data-testid="stSlider"] [data-testid="stThumbValue"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [class*="ThumbValue"] {
        background: #E0F2F1 !important;
        border: 1px solid rgba(15, 118, 110, 0.18) !important;
        box-shadow: 0 6px 14px rgba(15, 118, 110, 0.10) !important;
        color: #0F766E !important;
        -webkit-text-fill-color: #0F766E !important;
        border-radius: 7px !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [class*="TickBar"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] [class*="InnerThumb"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] [class*="Thumb"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] div {
        -webkit-text-fill-color: inherit;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] span,
    div[data-testid="stSlider"] [data-baseweb="slider"] p {
        color: #475569 !important;
        -webkit-text-fill-color: #475569 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="background-color: rgb(0, 104, 201)"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="background: rgb(0, 104, 201)"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(0, 104, 201)"] {
        background: #E0F2F1 !important;
        background-color: #E0F2F1 !important;
        border: 1px solid rgba(15, 118, 110, 0.18) !important;
        color: #0F766E !important;
        -webkit-text-fill-color: #0F766E !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="background-color: rgb(0, 104, 201)"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="background: rgb(0, 104, 201)"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(0, 104, 201)"] * {
        color: #0F766E !important;
        -webkit-text-fill-color: #0F766E !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(255, 75, 75)"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="#ff4b4b"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="255, 75, 75"] {
        background: #E0F2F1 !important;
        background-color: #E0F2F1 !important;
        border: 1px solid rgba(15, 118, 110, 0.18) !important;
        color: #0F766E !important;
        -webkit-text-fill-color: #0F766E !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(255, 75, 75)"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="#ff4b4b"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="255, 75, 75"] * {
        color: #0F766E !important;
        -webkit-text-fill-color: #0F766E !important;
    }
    div[data-testid="stSlider"] [role="slider"]::before {
        background: #E0F2F1 !important;
        color: #0F766E !important;
    }
    div[data-testid="stSlider"] [role="slider"]::after {
        background: #E0F2F1 !important;
        color: #0F766E !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [aria-valuenow] {
        color: #0F766E !important;
        -webkit-text-fill-color: #0F766E !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [aria-valuenow] * {
        color: #0F766E !important;
        -webkit-text-fill-color: #0F766E !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="background-color"] {
        border-radius: 999px !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #14B8A6 0%, #2563EB 100%) !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div > div {
        background: linear-gradient(90deg, #14B8A6 0%, #2563EB 100%) !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background: #0F766E !important;
        border-color: #FBFDF8 !important;
    }
    div[data-testid="stSlider"] [role="slider"] * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] span,
    div[data-testid="stSlider"] [data-baseweb="slider"] p,
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="transform"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="position: absolute"] {
        color: transparent !important;
        -webkit-text-fill-color: transparent !important;
    }
    div[data-testid="stTabs"] button,
    div[data-testid="stTabs"] [role="tab"] {
        opacity: 1 !important;
        color: #334155 !important;
        -webkit-text-fill-color: #334155 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] *,
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    .slider-readout {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin: -8px 0 12px 0;
        padding: 7px 11px;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(236,253,245,0.96), rgba(219,234,254,0.92));
        border: 1px solid rgba(15,118,110,0.14);
        box-shadow: 0 10px 24px rgba(15,23,42,0.06);
        color: #0F766E;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 700;
    }
    .slider-readout span {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: linear-gradient(135deg, #14B8A6, #2563EB);
        box-shadow: 0 0 0 4px rgba(20,184,166,0.12);
    }

    /* ---------- Modern product-dashboard polish ---------- */
    .block-container {
        max-width: 1380px;
        padding-top: 1.35rem;
        padding-left: 2.25rem;
        padding-right: 2.25rem;
    }
    .hero-wrap,
    .config-panel,
    .stat-card,
    .forecast-tile,
    .wp-wrap,
    .chart-shell,
    .mini-kpi,
    div[data-testid="stDataFrame"] {
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border-color: rgba(15, 23, 42, 0.08) !important;
    }
    .hero-wrap {
        border-radius: 24px;
        padding: 26px 30px;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.78), rgba(236,253,245,0.78)),
            radial-gradient(circle at 82% 16%, rgba(37,99,235,0.16), transparent 34%),
            radial-gradient(circle at 18% 24%, rgba(20,184,166,0.16), transparent 32%);
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.10);
    }
    .hero-title {
        font-size: 27px;
        letter-spacing: -0.035em;
    }
    .hero-sub {
        color: #475569;
        font-size: 14px;
    }
    .hero-badge {
        border-radius: 18px;
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%);
        box-shadow: 0 16px 34px rgba(37, 99, 235, 0.22);
    }
    .live-pill {
        color: #0F766E;
        background: rgba(20, 184, 166, 0.12);
        border-color: rgba(15, 118, 110, 0.18);
    }
    .live-dot {
        background: #0F766E;
        box-shadow: 0 0 0 0 rgba(15, 118, 110, 0.38);
    }
    .section-label {
        color: #1E293B;
        font-size: 12px;
        margin-bottom: 12px;
    }
    .section-label .tick {
        width: 4px;
        height: 16px;
        background: linear-gradient(180deg, #0F766E, #2563EB);
    }
    .config-panel,
    .stat-card,
    .forecast-tile,
    .wp-wrap,
    .chart-shell {
        background: rgba(251, 253, 248, 0.76) !important;
        border-radius: 20px;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.075);
    }
    .stat-card,
    .forecast-tile {
        border-left-width: 0;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before,
    .forecast-tile::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 5px;
        background: linear-gradient(180deg, #0F766E, #2563EB);
    }
    .stat-card.tone-wicket::before,
    .forecast-tile.f-wicket::before {
        background: linear-gradient(180deg, #F43F5E, #FB7185);
    }
    .stat-card.tone-amber::before {
        background: linear-gradient(180deg, #D97706, #FBBF24);
    }
    .forecast-tile.f-boundary::before {
        background: linear-gradient(180deg, #0891B2, #14B8A6);
    }
    .stat-card:hover,
    .forecast-tile:hover,
    .mini-kpi:hover {
        transform: translateY(-3px);
        box-shadow: 0 24px 60px rgba(15, 23, 42, 0.12);
    }
    .stat-card-badge {
        border-radius: 10px;
        background: rgba(15, 118, 110, 0.10) !important;
    }
    .stat-card-value,
    .forecast-tile .fvalue {
        letter-spacing: -0.03em;
    }
    .mini-kpi {
        background: linear-gradient(145deg, rgba(255,255,255,0.88), rgba(236,253,245,0.74));
        border-radius: 18px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .mini-kpi b {
        color: #0F766E;
    }
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.78) !important;
        border-color: rgba(15, 23, 42, 0.10) !important;
        border-radius: 14px !important;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05) !important;
    }
    label[data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-size: 11.5px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        border-radius: 999px !important;
        padding: 5px !important;
        background: rgba(255,255,255,0.58) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }
    .stTabs button,
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 9px 18px !important;
    }
    .wp-track {
        height: 12px;
        background: #E2E8F0;
    }
    .wp-fill {
        background: linear-gradient(90deg, #0F766E 0%, #14B8A6 45%, #2563EB 100%);
    }
    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        box-shadow: 0 18px 48px rgba(15, 23, 42, 0.075);
    }

    /* ---------- Vibrant light theme finish ---------- */
    .stApp {
        background:
            radial-gradient(circle at 8% 10%, rgba(20,184,166,0.18), transparent 28%),
            radial-gradient(circle at 92% 8%, rgba(37,99,235,0.16), transparent 28%),
            radial-gradient(circle at 72% 86%, rgba(168,85,247,0.12), transparent 30%),
            linear-gradient(135deg, #FFFDF5 0%, #F1FAF4 46%, #EEF6FF 100%) !important;
    }
    .hero-wrap {
        background:
            linear-gradient(135deg, rgba(255,255,255,0.74), rgba(236,253,245,0.80)),
            radial-gradient(circle at 14% 20%, rgba(20,184,166,0.20), transparent 30%),
            radial-gradient(circle at 82% 18%, rgba(37,99,235,0.18), transparent 36%),
            radial-gradient(circle at 66% 92%, rgba(168,85,247,0.10), transparent 34%);
    }
    .hero-wrap::before {
        background: radial-gradient(ellipse at top right, rgba(14,165,233,0.20) 0%, rgba(14,165,233,0) 68%);
    }
    .hero-wrap::after {
        border-color: rgba(37,99,235,0.13);
        box-shadow: inset 0 0 52px rgba(20,184,166,0.16);
    }
    .config-panel,
    .stat-card,
    .forecast-tile,
    .wp-wrap,
    .chart-shell {
        background:
            linear-gradient(145deg, rgba(255,255,255,0.82), rgba(246,253,247,0.74)) !important;
        border-color: rgba(15,118,110,0.10) !important;
    }
    .mini-kpi {
        background:
            linear-gradient(145deg, rgba(255,255,255,0.88), rgba(239,246,255,0.78)) !important;
        border-color: rgba(37,99,235,0.10) !important;
    }
    .stat-card.tone-floodlight::before,
    .forecast-tile.f-neutral::before {
        background: linear-gradient(180deg, #2563EB, #A855F7);
    }
    .section-label .tick {
        background: linear-gradient(180deg, #14B8A6, #2563EB, #A855F7);
    }
    .stTabs button[aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    div[data-testid="stTabs"] button[aria-selected="true"],
    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #14B8A6 0%, #2563EB 58%, #7C3AED 100%) !important;
        box-shadow: 0 12px 28px rgba(37,99,235,0.22) !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div,
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: #DDEFE1 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div,
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div > div {
        background: linear-gradient(90deg, #8DDCCB 0%, #7DB8F5 100%) !important;
    }
    div[data-testid="stSlider"] [role="slider"] {
        background: #2CBFAE !important;
        border-color: #F8FCF5 !important;
        box-shadow: 0 8px 18px rgba(44,191,174,0.20) !important;
    }
    div[data-testid="stSlider"] [data-testid="stThumbValue"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [class*="ThumbValue"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="background-color: rgb(0, 104, 201)"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(0, 104, 201)"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="background-color: rgb(20, 184, 166)"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(20, 184, 166)"] {
        background: #EAF8F3 !important;
        background-color: #EAF8F3 !important;
        border: 1px solid rgba(44,191,174,0.24) !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        box-shadow: 0 6px 14px rgba(15,23,42,0.08) !important;
    }
    div[data-testid="stSlider"] [data-testid="stThumbValue"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] [class*="ThumbValue"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(0, 104, 201)"] *,
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(20, 184, 166)"] * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBar"],
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"],
    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        opacity: 1 !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBar"] *,
    div[data-testid="stSlider"] [data-testid="stTickBarMin"] *,
    div[data-testid="stSlider"] [data-testid="stTickBarMax"] *,
    div[data-testid="stSlider"] [data-testid="stThumbValue"] * {
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
        opacity: 1 !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        padding: 1px 5px !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.10) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBar"],
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"],
    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        visibility: hidden !important;
    }
    .slider-readout {
        background: linear-gradient(135deg, rgba(250,255,250,0.98), rgba(234,248,243,0.96));
        color: #0F172A;
        border-color: rgba(44,191,174,0.18);
    }
    .slider-readout span {
        background: linear-gradient(135deg, #8DDCCB, #7DB8F5);
    }
    .wp-fill {
        background: linear-gradient(90deg, #14B8A6 0%, #2563EB 60%, #7C3AED 100%);
    }

    /* ---------- Final UX refinements ---------- */
    * {
        scrollbar-width: thin;
        scrollbar-color: rgba(15,118,110,0.36) transparent;
    }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, rgba(20,184,166,0.42), rgba(37,99,235,0.32));
        border-radius: 999px;
        border: 3px solid transparent;
        background-clip: padding-box;
    }
    .block-container > div {
        animation: fade-slide-in 0.42s ease both;
    }
    @keyframes fade-slide-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    button,
    [role="button"],
    div[data-testid="stSelectbox"],
    div[data-testid="stNumberInput"] {
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }
    div[data-testid="stSelectbox"]:hover,
    div[data-testid="stNumberInput"]:hover {
        transform: translateY(-1px);
    }
    div[data-testid="stNumberInput"]:hover div[data-testid="stNumberInputStepUp"],
    div[data-testid="stNumberInput"]:hover div[data-testid="stNumberInputStepDown"] {
        background: linear-gradient(135deg, #DDF7EE, #DBEAFE) !important;
        transform: translateY(-50%) scale(1.04);
    }
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.72);
        border-radius: 18px;
    }
    div[data-testid="stAlert"] {
        box-shadow: 0 14px 38px rgba(15,23,42,0.075);
    }
    div[data-testid="stMarkdownContainer"] a {
        color: #0F766E !important;
        text-decoration-color: rgba(15,118,110,0.32) !important;
        text-underline-offset: 3px;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 8px;
        z-index: 20;
    }
    .forecast-tile .fhead,
    .stat-card-head {
        min-height: 28px;
    }
    .chart-shell {
        padding: 20px;
    }
    .hr-fade {
        margin: 22px 0 24px 0;
        background: linear-gradient(90deg, transparent 0%, rgba(15,118,110,0.18) 18%, rgba(37,99,235,0.18) 70%, transparent 100%);
    }
    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .hero-wrap {
            flex-direction: column;
            align-items: flex-start;
            padding: 22px;
        }
        .hero-right {
            width: 100%;
            justify-content: space-between;
        }
        .mini-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        .stTabs [data-baseweb="tab-list"] {
            width: 100%;
            overflow-x: auto;
        }
    }
    @media (max-width: 560px) {
        .mini-strip {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 22px;
        }
        .session-chip {
            align-items: flex-start;
        }
    }

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"] { border-radius: 12px; background-color: var(--bg-panel); border: 1px solid var(--border-hair); }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join("data", "2_processed", "matchup_model.pkl")
ENCODER_PATH = os.path.join("data", "2_processed", "encoders.pkl")
WIN_MODEL_PATH = os.path.join("data", "2_processed", "win_probability_model.pkl")
PLAYER_METADATA_PATH = os.path.join("data", "player_metadata.csv")
MODEL_METRICS_PATH = os.path.join("data", "2_processed", "model_metrics.pkl")
PLAN_CACHE_PATH = os.path.join("data", "2_processed", "bowling_plan_cache.pkl")

def get_snowflake_connection():
    """Dynamically establishes a connection using local .env or Streamlit Cloud Secrets"""
    try:
        if "SF_USER" in st.secrets and st.secrets["SF_USER"]:
            return snowflake.connector.connect(
                user=st.secrets["SF_USER"],
                password=st.secrets["SF_PASSWORD"],
                account=st.secrets["SF_ACCOUNT"],
                warehouse=st.secrets["SF_WAREHOUSE"],
                database=st.secrets["SF_DATABASE"],
                schema=st.secrets["SF_SCHEMA"]
            )
    except Exception:
        pass

    return snowflake.connector.connect(
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        account=os.getenv("SF_ACCOUNT"),
        warehouse=os.getenv("SF_WAREHOUSE"),
        database=os.getenv("SF_DATABASE"),
        schema=os.getenv("SF_SCHEMA")
    )

@st.cache_resource
def load_ml_objects():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        with open(MODEL_PATH, "rb") as m_f:
            model = pickle.load(m_f)
        with open(ENCODER_PATH, "rb") as e_f:
            encoders = pickle.load(e_f)
        win_model = None
        if os.path.exists(WIN_MODEL_PATH):
            with open(WIN_MODEL_PATH, "rb") as w_f:
                win_model = pickle.load(w_f)
        return model, encoders, win_model
    return None, None, None


@st.cache_data
def load_player_metadata():
    if not os.path.exists(PLAYER_METADATA_PATH):
        return {}, {}
    metadata = pd.read_csv(PLAYER_METADATA_PATH).fillna("Unknown")
    batting_hands = dict(zip(metadata["PLAYER"], metadata["BATTING_HAND"]))
    bowling_types = dict(zip(metadata["PLAYER"], metadata["BOWLING_TYPE"]))
    return batting_hands, bowling_types


@st.cache_data
def load_model_metrics():
    if not os.path.exists(MODEL_METRICS_PATH):
        return {}
    with open(MODEL_METRICS_PATH, "rb") as metrics_file:
        return pickle.load(metrics_file)


@st.cache_data(ttl=3600)
def fetch_feature_freshness():
    try:
        ctx = get_snowflake_connection()
        freshness = pd.read_sql(
            """
            SELECT COUNT(*) AS DELIVERY_ROWS,
                   MAX(TRY_TO_NUMBER(SEASON)) AS LATEST_SEASON,
                   MAX(TRY_TO_DATE(START_DATE)) AS LATEST_MATCH_DATE
            FROM ANALYTICAL_MATCHUP_FEATURES
            """,
            ctx,
        ).iloc[0]
        ctx.close()
        return {
            "delivery_rows": int(freshness["DELIVERY_ROWS"]),
            "latest_season": str(freshness["LATEST_SEASON"]),
            "latest_match_date": freshness["LATEST_MATCH_DATE"],
        }
    except Exception:
        return {}


@st.cache_data(ttl=7200)
def get_team_rosters():
    ctx = get_snowflake_connection()
    active_season_cte = """
        WITH latest_season AS (
            SELECT SEASON
            FROM RAW_DELIVERIES
            WHERE SEASON IS NOT NULL
            GROUP BY SEASON
            ORDER BY COALESCE(TRY_TO_NUMBER(SEASON), 0) DESC, SEASON DESC
            LIMIT 1
        ),
        established_batters AS (
            SELECT STRIKER
            FROM RAW_DELIVERIES
            WHERE STRIKER IS NOT NULL
            GROUP BY STRIKER
            HAVING COUNT(*) >= 60
        )
    """
    batting_query = active_season_cte + """
        SELECT
            d.BATTING_TEAM AS TEAM,
            d.STRIKER AS PLAYER,
            IFF(b.STRIKER IS NOT NULL, TRUE, FALSE) AS ANALYTICS_ELIGIBLE
        FROM RAW_DELIVERIES d
        JOIN latest_season s ON d.SEASON = s.SEASON
        LEFT JOIN established_batters b ON d.STRIKER = b.STRIKER
        WHERE d.STRIKER IS NOT NULL
          AND TRIM(d.STRIKER) <> ''
        GROUP BY d.BATTING_TEAM, d.STRIKER, b.STRIKER
    """
    bowling_query = active_season_cte + """
        SELECT DISTINCT
            d.BOWLING_TEAM AS TEAM,
            d.BOWLER AS PLAYER,
            IFF(b.STRIKER IS NOT NULL, TRUE, FALSE) AS ANALYTICS_ELIGIBLE
        FROM RAW_DELIVERIES d
        JOIN latest_season s ON d.SEASON = s.SEASON
        LEFT JOIN established_batters b ON d.BOWLER = b.STRIKER
        WHERE d.BOWLER IS NOT NULL
          AND TRIM(d.BOWLER) <> ''
    """

    df_batters = pd.read_sql(batting_query, ctx)
    df_batters['ROLE'] = 'batter'
    df_bowlers = pd.read_sql(bowling_query, ctx)
    df_bowlers['ROLE'] = 'bowler'

    df_rosters = pd.concat([df_batters, df_bowlers], ignore_index=True)
    ctx.close()
    return df_rosters


@st.cache_data(ttl=7200)
def get_historical_batting_order(team, active_batters):
    ctx = get_snowflake_connection()
    batting_order = pd.read_sql(
        """
        WITH appearances AS (
            SELECT MATCH_ID, INNINGS, BATTING_TEAM, STRIKER AS PLAYER,
                   MIN(DELIVERY_SEQ) AS FIRST_DELIVERY
            FROM RAW_DELIVERIES
            WHERE BATTING_TEAM = %s AND STRIKER IS NOT NULL
            GROUP BY MATCH_ID, INNINGS, BATTING_TEAM, STRIKER
            UNION ALL
            SELECT MATCH_ID, INNINGS, BATTING_TEAM, NON_STRIKER AS PLAYER,
                   MIN(DELIVERY_SEQ) AS FIRST_DELIVERY
            FROM RAW_DELIVERIES
            WHERE BATTING_TEAM = %s AND NON_STRIKER IS NOT NULL
            GROUP BY MATCH_ID, INNINGS, BATTING_TEAM, NON_STRIKER
        ),
        first_appearance AS (
            SELECT MATCH_ID, INNINGS, PLAYER, MIN(FIRST_DELIVERY) AS FIRST_DELIVERY
            FROM appearances
            GROUP BY MATCH_ID, INNINGS, PLAYER
        ),
        ordered_innings AS (
            SELECT PLAYER,
                   DENSE_RANK() OVER (
                       PARTITION BY MATCH_ID, INNINGS ORDER BY FIRST_DELIVERY
                   ) AS BATTING_POSITION
            FROM first_appearance
        )
        SELECT PLAYER, AVG(BATTING_POSITION) AS AVG_POSITION,
               COUNT(*) AS INNINGS_PLAYED
        FROM ordered_innings
        GROUP BY PLAYER
        ORDER BY AVG_POSITION, INNINGS_PLAYED DESC, PLAYER
        """,
        ctx,
        params=(team, team),
    )
    ctx.close()
    active_batters = list(active_batters)
    historical_players = [
        player
        for player in batting_order["PLAYER"].tolist()
        if player in active_batters
    ]
    return historical_players + [
        player for player in active_batters if player not in historical_players
    ]


@st.cache_data(ttl=7200)
def get_available_venues():
    ctx = get_snowflake_connection()
    venues = pd.read_sql(
        "SELECT DISTINCT VENUE FROM RAW_DELIVERIES WHERE VENUE IS NOT NULL ORDER BY VENUE",
        ctx,
    )["VENUE"].tolist()
    ctx.close()
    return sorted({canonicalize_venue_name(venue) for venue in venues})

@st.cache_data(ttl=7200)
def get_detailed_stats(batter, bowler):
    ctx = get_snowflake_connection()
    h2h_query = """
        SELECT
            COALESCE(SUM(RUNS_OFF_BAT), 0) AS TOTAL_RUNS,
            COUNT(BALL) AS TOTAL_BALLS,
            COALESCE(SUM(CASE WHEN RUNS_OFF_BAT >= 4 THEN 1 ELSE 0 END), 0)
                AS TOTAL_BOUNDARIES,
            COALESCE(SUM(CASE
                WHEN PLAYER_DISMISSED = STRIKER
                 AND LOWER(WICKET_TYPE) IN (
                    'bowled', 'caught', 'caught and bowled', 'lbw',
                    'stumped', 'hit wicket'
                 )
                THEN 1 ELSE 0 END), 0) AS TOTAL_WICKETS
        FROM RAW_DELIVERIES
        WHERE STRIKER = %s AND BOWLER = %s
    """
    df_h2h = pd.read_sql(h2h_query, ctx, params=(batter, bowler)).iloc[0]

    bat_query = "SELECT COALESCE(SUM(RUNS_OFF_BAT), 0) as RUNS, COUNT(BALL) as BALLS, COUNT(DISTINCT MATCH_ID) as TOTAL_INNINGS, COALESCE(SUM(CASE WHEN PLAYER_DISMISSED = %s AND WICKET_TYPE IS NOT NULL THEN 1 ELSE 0 END), 0) as TOTAL_DISMISSALS FROM RAW_DELIVERIES WHERE STRIKER = %s"
    df_bat = pd.read_sql(bat_query, ctx, params=(batter, batter)).iloc[0]

    bowl_query = "SELECT COALESCE(SUM(RUNS_OFF_BAT), 0) as RUNS_CONCEDED, COUNT(BALL) as BALLS_BOWLED, COALESCE(SUM(CASE WHEN PLAYER_DISMISSED IS NOT NULL AND WICKET_TYPE IS NOT NULL THEN 1 ELSE 0 END), 0) as WICKETS_TAKEN FROM RAW_DELIVERIES WHERE BOWLER = %s"
    df_bowl = pd.read_sql(bowl_query, ctx, params=(bowler,)).iloc[0]
    ctx.close()
    return df_h2h, df_bat, df_bowl


@st.cache_data(ttl=7200)
def get_matchup_event_evidence(batter, bowler, innings):
    ctx = get_snowflake_connection()
    query = """
        SELECT
            COUNT(BALL) AS TOTAL_BALLS,
            COALESCE(SUM(CASE WHEN RUNS_OFF_BAT >= 4 THEN 1 ELSE 0 END), 0)
                AS TOTAL_BOUNDARIES,
            COALESCE(SUM(CASE
                WHEN PLAYER_DISMISSED = STRIKER
                 AND LOWER(WICKET_TYPE) IN (
                    'bowled', 'caught', 'caught and bowled', 'lbw',
                    'stumped', 'hit wicket'
                 )
                THEN 1 ELSE 0 END), 0) AS TOTAL_WICKETS
        FROM RAW_DELIVERIES
        WHERE STRIKER = %s AND BOWLER = %s AND INNINGS = %s
    """
    evidence = pd.read_sql(
        query, ctx, params=(batter, bowler, innings)
    ).iloc[0]
    ctx.close()
    return evidence


@st.cache_data(ttl=7200)
def get_matrix_evidence(batters, bowlers, innings):
    batter_names = tuple(batters)
    bowler_names = tuple(bowlers)
    if not batter_names or not bowler_names:
        return pd.DataFrame(), pd.DataFrame()

    batter_placeholders = ",".join(["%s"] * len(batter_names))
    bowler_placeholders = ",".join(["%s"] * len(bowler_names))
    ctx = get_snowflake_connection()

    evidence_query = f"""
        WITH latest_striker_form AS (
            SELECT STRIKER, STRIKER_RECENT_BOUNDARY_RATE,
                   STRIKER_RECENT_RUN_RATE
            FROM ANALYTICAL_MATCHUP_FEATURES
            WHERE STRIKER IN ({batter_placeholders})
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY STRIKER
                ORDER BY TRY_TO_DATE(START_DATE) DESC, MATCH_ID DESC,
                         INNINGS DESC, BALL DESC
            ) = 1
        ),
        latest_bowler_form AS (
            SELECT BOWLER, BOWLER_RECENT_BOUNDARY_RATE,
                   BOWLER_RECENT_DISMISSAL_RATE,
                   BOWLER_RECENT_RUN_RATE
            FROM ANALYTICAL_MATCHUP_FEATURES
            WHERE BOWLER IN ({bowler_placeholders})
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY BOWLER
                ORDER BY TRY_TO_DATE(START_DATE) DESC, MATCH_ID DESC,
                         INNINGS DESC, BALL DESC
            ) = 1
        )
        SELECT
            d.STRIKER,
            d.BOWLER,
            COUNT(d.BALL) AS TOTAL_BALLS,
            COALESCE(SUM(CASE WHEN d.RUNS_OFF_BAT = 0 THEN 1 ELSE 0 END), 0)
                AS TOTAL_DOTS,
            COALESCE(SUM(d.RUNS_OFF_BAT), 0) AS TOTAL_RUNS,
            COALESCE(SUM(CASE WHEN d.RUNS_OFF_BAT >= 4 THEN 1 ELSE 0 END), 0)
                AS TOTAL_BOUNDARIES,
            COALESCE(SUM(CASE
                WHEN d.PLAYER_DISMISSED = d.STRIKER
                 AND LOWER(d.WICKET_TYPE) IN (
                    'bowled', 'caught', 'caught and bowled', 'lbw',
                    'stumped', 'hit wicket'
                 )
                THEN 1 ELSE 0 END), 0) AS TOTAL_WICKETS,
            MAX(sf.STRIKER_RECENT_BOUNDARY_RATE)
                AS STRIKER_RECENT_BOUNDARY_RATE,
            MAX(sf.STRIKER_RECENT_RUN_RATE)
                AS STRIKER_RECENT_RUN_RATE,
            MAX(bf.BOWLER_RECENT_BOUNDARY_RATE)
                AS BOWLER_RECENT_BOUNDARY_RATE,
            MAX(bf.BOWLER_RECENT_DISMISSAL_RATE)
                AS BOWLER_RECENT_DISMISSAL_RATE,
            MAX(bf.BOWLER_RECENT_RUN_RATE)
                AS BOWLER_RECENT_RUN_RATE
        FROM RAW_DELIVERIES d
        LEFT JOIN latest_striker_form sf ON d.STRIKER = sf.STRIKER
        LEFT JOIN latest_bowler_form bf ON d.BOWLER = bf.BOWLER
        WHERE d.STRIKER IN ({batter_placeholders})
          AND d.BOWLER IN ({bowler_placeholders})
          AND d.INNINGS = %s
        GROUP BY d.STRIKER, d.BOWLER
    """
    evidence_params = (
        batter_names
        + bowler_names
        + batter_names
        + bowler_names
        + (innings,)
    )
    evidence = pd.read_sql(evidence_query, ctx, params=evidence_params)

    profile_query = f"""
        SELECT
            BOWLER,
            COALESCE(SUM(CASE
                WHEN LOWER(WICKET_TYPE) IN (
                    'bowled', 'caught', 'caught and bowled', 'lbw',
                    'stumped', 'hit wicket'
                )
                THEN 1 ELSE 0 END), 0) AS WICKETS_TAKEN
        FROM RAW_DELIVERIES
        WHERE BOWLER IN ({bowler_placeholders})
        GROUP BY BOWLER
    """
    profiles = pd.read_sql(profile_query, ctx, params=bowler_names)
    ctx.close()
    return evidence, profiles

@st.cache_data(ttl=7200)
def fetch_knn_candidate_pool(balls_left):
    ctx = get_snowflake_connection()
    query = """
        SELECT MATCH_ID, VENUE, STRIKER, NON_STRIKER, BALLS_REMAINING, RUNS_NEEDED,
               CURRENT_WICKETS_LOST, REQUIRED_RUN_RATE, CHASE_WON
        FROM ANALYTICAL_MATCHUP_FEATURES
        WHERE BALLS_REMAINING BETWEEN %s AND %s
    """
    df = pd.read_sql(query, ctx, params=(max(balls_left - 15, 0), balls_left + 15))
    ctx.close()
    return df


@st.cache_data(ttl=7200)
def fetch_high_pressure_candidate_pool():
    ctx = get_snowflake_connection()
    query = """
        SELECT MATCH_ID, VENUE, STRIKER, NON_STRIKER, BALLS_REMAINING, RUNS_NEEDED,
               CURRENT_WICKETS_LOST, REQUIRED_RUN_RATE, CHASE_WON
        FROM ANALYTICAL_MATCHUP_FEATURES
        WHERE BALLS_REMAINING BETWEEN 1 AND 30
          AND REQUIRED_RUN_RATE >= 12
    """
    df = pd.read_sql(query, ctx)
    ctx.close()
    return df


@st.cache_data(ttl=7200)
def fetch_phase_run_samples(innings, balls_remaining):
    if balls_remaining > 84:
        lower_bound, upper_bound = 85, 120
    elif balls_remaining > 30:
        lower_bound, upper_bound = 31, 84
    else:
        lower_bound, upper_bound = 0, 30

    ctx = get_snowflake_connection()
    query = """
        SELECT EVENT_TYPE, RUNS_OFF_BAT
        FROM ANALYTICAL_MATCHUP_FEATURES
        WHERE INNINGS = %s
          AND BALLS_REMAINING BETWEEN %s AND %s
          AND EVENT_TYPE IN (0, 1)
    """
    samples = pd.read_sql(
        query,
        ctx,
        params=(innings, lower_bound, upper_bound),
    )
    ctx.close()
    safe_runs = samples.loc[
        samples["EVENT_TYPE"] == 0, "RUNS_OFF_BAT"
    ].astype(int).tolist()
    boundary_runs = samples.loc[
        samples["EVENT_TYPE"] == 1, "RUNS_OFF_BAT"
    ].astype(int).tolist()
    return safe_runs, boundary_runs


@st.cache_data(ttl=7200)
def fetch_recent_event_form(batter, bowler):
    ctx = get_snowflake_connection()
    query = """
        SELECT
            COALESCE((
                SELECT STRIKER_RECENT_BOUNDARY_RATE
                FROM ANALYTICAL_MATCHUP_FEATURES
                WHERE STRIKER = %s
                ORDER BY TRY_TO_DATE(START_DATE) DESC, MATCH_ID DESC,
                         INNINGS DESC, BALL DESC
                LIMIT 1
            ), 0.0) AS STRIKER_RECENT_BOUNDARY_RATE,
            COALESCE((
                SELECT BOWLER_RECENT_BOUNDARY_RATE
                FROM ANALYTICAL_MATCHUP_FEATURES
                WHERE BOWLER = %s
                ORDER BY TRY_TO_DATE(START_DATE) DESC, MATCH_ID DESC,
                         INNINGS DESC, BALL DESC
                LIMIT 1
            ), 0.0) AS BOWLER_RECENT_BOUNDARY_RATE,
            COALESCE((
                SELECT BOWLER_RECENT_DISMISSAL_RATE
                FROM ANALYTICAL_MATCHUP_FEATURES
                WHERE BOWLER = %s
                ORDER BY TRY_TO_DATE(START_DATE) DESC, MATCH_ID DESC,
                         INNINGS DESC, BALL DESC
                LIMIT 1
            ), 0.0) AS BOWLER_RECENT_DISMISSAL_RATE,
            COALESCE((
                SELECT STRIKER_RECENT_RUN_RATE
                FROM ANALYTICAL_MATCHUP_FEATURES
                WHERE STRIKER = %s
                ORDER BY TRY_TO_DATE(START_DATE) DESC, MATCH_ID DESC,
                         INNINGS DESC, BALL DESC
                LIMIT 1
            ), 0.0) AS STRIKER_RECENT_RUN_RATE,
            COALESCE((
                SELECT BOWLER_RECENT_RUN_RATE
                FROM ANALYTICAL_MATCHUP_FEATURES
                WHERE BOWLER = %s
                ORDER BY TRY_TO_DATE(START_DATE) DESC, MATCH_ID DESC,
                         INNINGS DESC, BALL DESC
                LIMIT 1
            ), 0.0) AS BOWLER_RECENT_RUN_RATE
    """
    form = pd.read_sql(
        query, ctx, params=(batter, bowler, bowler, batter, bowler)
    ).iloc[0]
    ctx.close()
    return form


@st.cache_data(ttl=7200)
def fetch_venue_event_form(venue):
    ctx = get_snowflake_connection()
    form = pd.read_sql(
        """
        SELECT COALESCE(VENUE_RECENT_RUN_RATE, 0.0)
            AS VENUE_RECENT_RUN_RATE
        FROM ANALYTICAL_MATCHUP_FEATURES
        WHERE VENUE = %s OR SPLIT_PART(VENUE, ',', 1) = %s
        ORDER BY TRY_TO_DATE(START_DATE) DESC, MATCH_ID DESC,
                 INNINGS DESC, BALL DESC
        LIMIT 1
        """,
        ctx,
        params=(venue, canonicalize_venue_name(venue)),
    )
    ctx.close()
    if form.empty:
        return 0.0
    return float(form.iloc[0]["VENUE_RECENT_RUN_RATE"])


@st.cache_data(ttl=7200)
def fetch_venue_par_stats(venue):
    ctx = get_snowflake_connection()
    totals = pd.read_sql(
        """
        SELECT MATCH_ID, SEASON, VENUE,
               SUM(RUNS_OFF_BAT + EXTRAS) AS INNINGS_TOTAL
        FROM RAW_DELIVERIES
        WHERE INNINGS = 1
        GROUP BY MATCH_ID, SEASON, VENUE
        """,
        ctx,
    )
    ctx.close()
    totals["CANONICAL_VENUE"] = totals["VENUE"].map(
        canonicalize_venue_name
    )
    totals["NUMERIC_SEASON"] = pd.to_numeric(
        totals["SEASON"], errors="coerce"
    )
    latest_season = totals["NUMERIC_SEASON"].max()
    comparable = totals[
        (totals["CANONICAL_VENUE"] == canonicalize_venue_name(venue))
        & (totals["NUMERIC_SEASON"] >= latest_season - 4)
    ]
    if comparable.empty:
        comparable = totals[
            totals["CANONICAL_VENUE"] == canonicalize_venue_name(venue)
        ]
    if comparable.empty:
        return None
    scores = comparable["INNINGS_TOTAL"].astype(float)
    return {
        "matches": int(len(scores)),
        "par_score": float(scores.median()),
        "lower_score": float(scores.quantile(0.25)),
        "upper_score": float(scores.quantile(0.75)),
    }

def calculate_historical_win_percentage(
    balls_left,
    wickets_lost,
    current_score,
    target_score,
    current_run_rate,
    striker_name=None,
):
    df_features = fetch_knn_candidate_pool(balls_left)
    return estimate_chase_win_percentage(
        balls_left,
        wickets_lost,
        current_score,
        target_score,
        current_run_rate,
        df_features,
        striker_name,
    )



def get_ordered_event_probs(model, feature_row):
    """
    Returns [P(dot/single), P(boundary), P(wicket)] in that fixed order, looked up via
    model.classes_ rather than assumed from predict_proba's raw column position (EVENT_TYPE
    encodes 0=dot/single, 1=boundary, 2=wicket -- but predict_proba's columns follow
    model.classes_, which is not guaranteed to be [0, 1, 2] in that order).
    """
    raw_proba = model.predict_proba(feature_row)[0]
    class_index = {label: i for i, label in enumerate(model.classes_)}
    return np.array([
        raw_proba[class_index[0]] if 0 in class_index else 0.0,
        raw_proba[class_index[1]] if 1 in class_index else 0.0,
        raw_proba[class_index[2]] if 2 in class_index else 0.0,
    ])


def get_ordered_outcome_probs(model, feature_row):
    if (
        not hasattr(model, "predict_outcome_proba")
        or getattr(model, "run_model", None) is None
    ):
        return None
    return model.predict_outcome_proba(feature_row)[0]


def render_delivery_outcome_chart(probabilities):
    labels = [
        "Dot",
        "1 Run",
        "2 Runs",
        "3 Runs",
        "Four",
        "Six",
        "Striker Dismissal",
    ]
    colors = [
        "#2563EB",
        "#4F46E5",
        "#7C3AED",
        "#A855F7",
        "#0F766E",
        "#0891B2",
        "#F43F5E",
    ]
    chart_data = pd.DataFrame(
        {
            "Outcome": labels,
            "Probability": np.asarray(probabilities) * 100,
            "Color": colors,
        }
    )
    chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8, size=22)
        .encode(
            x=alt.X(
                "Probability:Q",
                title=None,
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(format=".0f", grid=True, labelPadding=8),
            ),
            y=alt.Y(
                "Outcome:N",
                title=None,
                sort=labels,
                axis=alt.Axis(labelPadding=10),
            ),
            color=alt.Color(
                "Outcome:N",
                scale=alt.Scale(domain=labels, range=colors),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Outcome:N"),
                alt.Tooltip("Probability:Q", format=".1f"),
            ],
        )
        .properties(height=245, background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(
            gridColor="#E6EEE6",
            labelColor="#475569",
            domain=False,
            tickColor="#CBD5E1",
            labelFont="Inter",
            titleFont="Inter",
        )
    )
    st.altair_chart(chart, use_container_width=True, theme=None)


def render_event_probability_chart(probabilities):
    df_events = pd.DataFrame({
        "Event": ["0-3 Runs / No Dismissal", "Boundary", "Striker Dismissal"],
        "Probability": [probabilities[0] * 100, probabilities[1] * 100, probabilities[2] * 100],
        "Color": ["#2563EB", "#0F766E", "#F43F5E"]
    })
    chart = (
        alt.Chart(df_events)
        .mark_bar(cornerRadiusTopRight=10, cornerRadiusBottomRight=10, size=26)
        .encode(
            x=alt.X(
                "Probability:Q",
                title=None,
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(format=".0f", grid=True, labelPadding=8)
            ),
            y=alt.Y("Event:N", title=None, sort=None, axis=alt.Axis(labelPadding=12)),
            color=alt.Color("Event:N", scale=alt.Scale(range=df_events["Color"].tolist()), legend=None),
            tooltip=[alt.Tooltip("Event:N"), alt.Tooltip("Probability:Q", format=".1f")]
        )
        .properties(height=150, background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(
            gridColor="#E6EEE6",
            labelColor="#475569",
            domain=False,
            tickColor="#CBD5E1",
            labelFont="Inter",
            titleFont="Inter"
        )
    )
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True, theme=None)
    st.markdown('</div>', unsafe_allow_html=True)

def render_matchup_threat_chart(df_analysis, batter_name):
    chart_df = df_analysis[[
        "Active Opposing Bowler",
        f"{batter_name} Wicket Danger %",
        f"{batter_name} Boundary Leak %"
    ]].melt("Active Opposing Bowler", var_name="Signal", value_name="Percent")
    chart_df["Signal"] = chart_df["Signal"].str.replace(f"{batter_name} ", "", regex=False).str.replace(" %", "", regex=False)

    chart = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
        .encode(
            x=alt.X(
                "Active Opposing Bowler:N",
                title=None,
                sort="-y",
                axis=alt.Axis(labelAngle=-35, labelLimit=140, labelPadding=10)
            ),
            y=alt.Y(
                "Percent:Q",
                title="Predicted probability (%)",
                axis=alt.Axis(grid=True, labelPadding=8)
            ),
            color=alt.Color(
                "Signal:N",
                scale=alt.Scale(range=["#F43F5E", "#0F766E"]),
                legend=alt.Legend(orient="top", title=None, symbolType="square")
            ),
            xOffset="Signal:N",
            tooltip=[
                alt.Tooltip("Active Opposing Bowler:N", title="Bowler"),
                alt.Tooltip("Signal:N"),
                alt.Tooltip("Percent:Q", format=".1f")
            ]
        )
        .properties(height=360, background="transparent")
        .configure_view(strokeWidth=0, fill="transparent")
        .configure_axis(
            gridColor="#E6EEE6",
            labelColor="#475569",
            titleColor="#334155",
            domain=False,
            tickColor="#CBD5E1",
            labelFont="Inter",
            titleFont="Inter"
        )
        .configure_legend(labelColor="#334155", labelFont="Inter")
    )
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True, theme=None)
    st.markdown('</div>', unsafe_allow_html=True)

# --- APP LAYOUT RUNNER ---
model, encoders, win_model = load_ml_objects()
df_rosters = get_team_rosters()
available_venues = get_available_venues()
batting_hands, bowling_types = load_player_metadata()
model_metrics = load_model_metrics()
feature_freshness = fetch_feature_freshness()
drift_status = assess_model_drift(model_metrics, feature_freshness)
show_model_validation = os.getenv(
    "SHOW_MODEL_VALIDATION", "false"
).strip().lower() in {"1", "true", "yes", "on"}

if model is None or win_model is None or df_rosters.empty:
    st.error("❌ Critical system initialization files missing database context mappings.")
else:
    all_batters_global = sorted(df_rosters[df_rosters['ROLE'] == 'batter']['PLAYER'].unique())
    all_bowlers_global = sorted(df_rosters[df_rosters['ROLE'] == 'bowler']['PLAYER'].unique())
    all_teams = sorted(df_rosters['TEAM'].unique())

    st.markdown("""
        <div class="hero-wrap">
            <div class="hero-left">
                <div class="hero-badge">🏏</div>
                <div>
                    <p class="hero-title">IPL Strategy Engine</p>
                    <p class="hero-sub">Predictive operations center for ball-by-ball matchup intelligence</p>
                </div>
            </div>
            <div class="hero-right">
                <div class="session-chip">Model Build<span>v2.3-knn</span></div>
                <span class="live-pill"><span class="live-dot"></span>LIVE MODEL</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    tab_labels = [
        "📊  Live Simulation Deck",
        "🛡️  Franchise Matchup Matrix",
    ]
    if show_model_validation:
        tab_labels.append("🧪  Model Validation")
    application_tabs = st.tabs(tab_labels)
    tab1, tab2 = application_tabs[:2]
    tab3 = application_tabs[2] if show_model_validation else None

    with tab1:
        c_setup1, c_setup2 = st.columns([1, 2], gap="large")
        with c_setup1:
            st.markdown('<p class="section-label"><span class="tick"></span>Match Parameters</p>', unsafe_allow_html=True)
            st.markdown('<div class="config-panel">', unsafe_allow_html=True)
            sim_batter = st.selectbox("Active Striker", all_batters_global, index=0)
            eligible_non_strikers = [
                name for name in all_batters_global if name != sim_batter
            ]
            sim_non_striker = st.selectbox(
                "Non-Striker", eligible_non_strikers, index=0
            )
            eligible_incoming_batters = [
                name
                for name in all_batters_global
                if name not in {sim_batter, sim_non_striker}
            ]
            sim_incoming_batter = st.selectbox(
                "Next Batter If Wicket Falls",
                eligible_incoming_batters,
                index=0,
            )
            eligible_bowlers = [name for name in all_bowlers_global if name != sim_batter]
            sim_bowler = st.selectbox("Active Bowler", eligible_bowlers, index=0)
            sim_venue = st.selectbox("Venue", available_venues, index=0)
            current_score = st.number_input("Current Runs Scored", min_value=0, value=120)
            wickets_down = st.number_input("Wickets Lost", min_value=0, max_value=9, value=3, step=1)
            ov_col1, ov_col2 = st.columns(2)
            with ov_col1:
                overs_whole = st.number_input("Overs Completed", min_value=0, max_value=19, value=15, step=1)
            with ov_col2:
                balls_in_over = st.number_input("Balls This Over", min_value=0, max_value=6, value=0, step=1,
                                                 help="0-6: balls bowled in the current over (6 = over just completed)")
            target_to_chase = st.number_input("Target to Chase", min_value=0, value=160)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_setup2:
            st.markdown('<p class="section-label"><span class="tick"></span>Live In-Game Metrics</p>', unsafe_allow_html=True)
            h2h, bat_p, bowl_p = get_detailed_stats(sim_batter, sim_bowler)

            k_prof1, k_prof2 = st.columns(2)
            with k_prof1:
                b_sr = (bat_p['RUNS'] / bat_p['BALLS']) * 100 if bat_p['BALLS'] > 0 else 0
                b_avg = bat_p['RUNS'] / int(bat_p['TOTAL_DISMISSALS']) if int(bat_p['TOTAL_DISMISSALS']) > 0 else bat_p['RUNS'] / max(int(bat_p['TOTAL_INNINGS']), 1)
                st.markdown(f"""
                    <div class="stat-card tone-floodlight">
                        <div class="stat-card-head">
                            <span class="stat-card-title">{sim_batter} · Career</span>
                            <span class="stat-card-badge">🏏</span>
                        </div>
                        <p class="stat-card-value stat-mono">{int(bat_p['RUNS'])} <span style="font-size:14px;color:var(--text-muted);">runs</span></p>
                        <p class="stat-card-foot">Avg <b>{b_avg:.2f}</b> &nbsp;·&nbsp; SR <b>{b_sr:.1f}</b> &nbsp;·&nbsp; Inn <b>{int(bat_p['TOTAL_INNINGS'])}</b></p>
                    </div>
                """, unsafe_allow_html=True)
            with k_prof2:
                bw_econ = (bowl_p['RUNS_CONCEDED'] / bowl_p['BALLS_BOWLED']) * 6 if bowl_p['BALLS_BOWLED'] > 0 else 0
                bw_wickets = int(bowl_p['WICKETS_TAKEN'])
                bw_avg = bowl_p['RUNS_CONCEDED'] / bw_wickets if bw_wickets > 0 else np.nan
                avg_str = f"{bw_avg:.2f}" if not np.isnan(bw_avg) else "N/A"
                st.markdown(f"""
                    <div class="stat-card tone-wicket">
                        <div class="stat-card-head">
                            <span class="stat-card-title">{sim_bowler} · Career</span>
                            <span class="stat-card-badge">🎯</span>
                        </div>
                        <p class="stat-card-value stat-mono">{bw_wickets} <span style="font-size:14px;color:var(--text-muted);">wickets</span></p>
                        <p class="stat-card-foot">Econ <b>{bw_econ:.2f}</b> &nbsp;·&nbsp; Avg <b>{avg_str}</b> &nbsp;·&nbsp; Balls <b>{int(bowl_p['BALLS_BOWLED'])}</b></p>
                    </div>
                """, unsafe_allow_html=True)

            # Restored Head-to-Head Statistics Panel Block
            h2h_runs = int(h2h['TOTAL_RUNS'])
            h2h_balls = int(h2h['TOTAL_BALLS'])
            h2h_wickets = int(h2h['TOTAL_WICKETS'])
            h2h_sr = (h2h_runs / h2h_balls) * 100 if h2h_balls > 0 else 0.0

            st.markdown(f"""
                <div class="stat-card tone-amber">
                    <div class="stat-card-head">
                        <span class="stat-card-title">Head-to-Head History · {sim_batter} vs {sim_bowler}</span>
                        <span class="stat-card-badge">⚔️</span>
                    </div>
                    <p class="stat-card-value stat-mono">{h2h_runs} <span style="font-size:14px;color:var(--text-muted);">runs off</span> {h2h_balls} <span style="font-size:14px;color:var(--text-muted);">balls</span></p>
                    <p class="stat-card-foot">Matchup Dismissals <b>{h2h_wickets}</b> &nbsp;·&nbsp; Matchup SR <b>{h2h_sr:.1f}</b></p>
                </div>
            """, unsafe_allow_html=True)

            total_balls_bowled = overs_whole * 6 + balls_in_over
            balls_remaining = max(120 - total_balls_bowled, 0)
            crr = (current_score / total_balls_bowled) * 6 if total_balls_bowled > 0 else 0.0
            rrr = ((target_to_chase - current_score) / balls_remaining) * 6 if target_to_chase > 0 and balls_remaining > 0 else 0.0
            runs_needed = max(target_to_chase - current_score, 0)

            st.markdown(f"""
                <div class="mini-strip">
                    <div class="mini-kpi"><span>Balls Left</span><b>{balls_remaining}</b></div>
                    <div class="mini-kpi"><span>Runs Needed</span><b>{runs_needed}</b></div>
                    <div class="mini-kpi"><span>Current RR</span><b>{crr:.2f}</b></div>
                    <div class="mini-kpi"><span>Required RR</span><b>{rrr:.2f}</b></div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('<p class="section-label"><span class="tick"></span>Win Probability</p>', unsafe_allow_html=True)
            win_probability_source = "Historically trained chase model"
            if runs_needed <= 0:
                win_pct = 100.0
                win_probability_source = "Deterministic: target reached"
            elif balls_remaining <= 0 or wickets_down >= 10:
                win_pct = 0.0
                win_probability_source = "Deterministic: innings complete"
            else:
                win_features = build_win_feature_frame(
                    balls_remaining,
                    current_score,
                    runs_needed,
                    wickets_down,
                    crr,
                    rrr,
                    sim_batter,
                    sim_non_striker,
                    sim_venue,
                )
                win_class_index = list(win_model.classes_).index(1)
                win_pct = (
                    win_model.predict_proba(win_features)[0][win_class_index] * 100
                )
                if runs_needed / max(balls_remaining, 1) > 2.5:
                    pressure_history = fetch_high_pressure_candidate_pool()
                    historical_pressure_pct = estimate_high_pressure_from_history(
                        pressure_history,
                        balls_remaining,
                        runs_needed,
                        wickets_down,
                        sim_batter,
                        sim_non_striker,
                        sim_venue,
                    )
                    if historical_pressure_pct is not None:
                        win_pct = historical_pressure_pct
                        win_probability_source = (
                            "Nearest comparable high-pressure historical matches"
                        )

            st.markdown(f"""
                <div class="wp-wrap">
                    <div class="wp-labels">
                        <span class="side chasing"><span class="dotmark"></span>Chasing<span class="val">{win_pct:.1f}%</span></span>
                        <span class="side defending">Defending<span class="val">{(100.0 - win_pct):.1f}%</span><span class="dotmark"></span></span>
                    </div>
                    <div class="wp-track">
                        <div class="wp-fill" style="width:{win_pct:.1f}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.caption(f"Win probability source: {win_probability_source}")

            try:
                recent_event_form = fetch_recent_event_form(
                    sim_batter, sim_bowler
                )
                venue_recent_run_rate = fetch_venue_event_form(sim_venue)
                event_features = build_event_feature_frame(
                    sim_batter,
                    sim_bowler,
                    balls_remaining,
                    wickets_down,
                    crr,
                    rrr,
                    innings=2,
                    current_team_score=current_score,
                    batting_hand=batting_hands.get(sim_batter, "Unknown"),
                    bowling_type=bowling_types.get(sim_bowler, "Unknown"),
                    striker_recent_boundary_rate=float(
                        recent_event_form[
                            "STRIKER_RECENT_BOUNDARY_RATE"
                        ]
                    ),
                    bowler_recent_boundary_rate=float(
                        recent_event_form[
                            "BOWLER_RECENT_BOUNDARY_RATE"
                        ]
                    ),
                    bowler_recent_dismissal_rate=float(
                        recent_event_form[
                            "BOWLER_RECENT_DISMISSAL_RATE"
                        ]
                    ),
                    striker_recent_run_rate=float(
                        recent_event_form["STRIKER_RECENT_RUN_RATE"]
                    ),
                    bowler_recent_run_rate=float(
                        recent_event_form["BOWLER_RECENT_RUN_RATE"]
                    ),
                    venue=sim_venue,
                    venue_recent_run_rate=venue_recent_run_rate,
                )
                ml_p = get_ordered_event_probs(model, event_features)
                outcome_p = get_ordered_outcome_probs(model, event_features)

                st.markdown('<p class="section-label"><span class="tick"></span>Next-Ball Event Forecast</p>', unsafe_allow_html=True)
                if h2h_balls < 20:
                    st.caption(
                        f"Low-confidence matchup ({int(h2h_balls)} historical balls): "
                        "forecast relies mostly on broader historical player and match context."
                    )
                if outcome_p is not None:
                    expected_ball_runs = float(
                        outcome_p @ model.outcome_run_values_
                    )
                    pm1, pm2, pm3, pm4 = st.columns(4)
                    pm1.metric("Expected Runs / Ball", f"{expected_ball_runs:.2f}")
                    pm2.metric("Dot Ball", f"{outcome_p[0] * 100:.1f}%")
                    pm3.metric("Boundary", f"{ml_p[1] * 100:.1f}%")
                    pm4.metric("Striker Dismissal", f"{ml_p[2] * 100:.1f}%")
                    render_delivery_outcome_chart(outcome_p)
                    st.caption(
                        "Exact-run probabilities are learned from historical "
                        "deliveries; wicket and boundary totals retain separate "
                        "probability calibration."
                    )
                    if getattr(model, "outcome_adapter", None) is not None:
                        st.caption(
                            "Recent-season drift adjustment is active because "
                            "it improved strict chronological backtesting."
                        )
                else:
                    pm1, pm2, pm3 = st.columns(3)
                    with pm1:
                        st.markdown(f'<div class="forecast-tile f-neutral"><div class="fhead"><span class="fdot"></span><span class="flabel">0-3 Runs / Safe</span></div><div class="fvalue">{ml_p[0]*100:.1f}%</div></div>', unsafe_allow_html=True)
                    with pm2:
                        st.markdown(f'<div class="forecast-tile f-boundary"><div class="fhead"><span class="fdot"></span><span class="flabel">Boundary</span></div><div class="fvalue">{ml_p[1]*100:.1f}%</div></div>', unsafe_allow_html=True)
                    with pm3:
                        st.markdown(f'<div class="forecast-tile f-wicket"><div class="fhead"><span class="fdot"></span><span class="flabel">Striker Dismissal</span></div><div class="fvalue">{ml_p[2]*100:.1f}%</div></div>', unsafe_allow_html=True)
                    render_event_probability_chart(ml_p)
                    st.info(
                        "Run src/models.py once to enable granular dot, run, "
                        "four, and six probabilities."
                    )
                safe_run_samples, boundary_run_samples = fetch_phase_run_samples(
                    2, balls_remaining
                )
                over_forecast = simulate_dynamic_over(
                    model,
                    sim_batter,
                    sim_non_striker,
                    sim_bowler,
                    balls_remaining,
                    wickets_down,
                    current_score,
                    target_to_chase,
                    crr,
                    rrr,
                    safe_run_samples,
                    boundary_run_samples,
                    innings=2,
                    batting_hands=batting_hands,
                    bowling_types=bowling_types,
                    replacement_batter=sim_incoming_batter,
                    striker_recent_boundary_rate=float(
                        recent_event_form[
                            "STRIKER_RECENT_BOUNDARY_RATE"
                        ]
                    ),
                    bowler_recent_boundary_rate=float(
                        recent_event_form[
                            "BOWLER_RECENT_BOUNDARY_RATE"
                        ]
                    ),
                    bowler_recent_dismissal_rate=float(
                        recent_event_form[
                            "BOWLER_RECENT_DISMISSAL_RATE"
                        ]
                    ),
                    striker_recent_run_rate=float(
                        recent_event_form["STRIKER_RECENT_RUN_RATE"]
                    ),
                    bowler_recent_run_rate=float(
                        recent_event_form["BOWLER_RECENT_RUN_RATE"]
                    ),
                    venue=sim_venue,
                    venue_recent_run_rate=venue_recent_run_rate,
                )
                st.markdown(
                    '<p class="section-label"><span class="tick"></span>'
                    'Next-Over Historical Simulation</p>',
                    unsafe_allow_html=True,
                )
                over_col1, over_col2, over_col3, over_col4 = st.columns(4)
                over_col1.metric(
                    "Expected Runs", f"{over_forecast['expected_runs']:.1f}"
                )
                over_col2.metric(
                    "Likely Range",
                    f"{over_forecast['low_runs']:.0f}–{over_forecast['high_runs']:.0f}",
                )
                over_col3.metric(
                    "Wicket in Over",
                    f"{over_forecast['wicket_probability'] * 100:.1f}%",
                )
                over_col4.metric(
                    "Boundary in Over",
                    f"{over_forecast['boundary_probability'] * 100:.1f}%",
                )
                st.caption(
                    "1,500 dynamic simulations updating score, wickets, strike, "
                    "balls, CRR and RRR after every delivery."
                )
                if st.button("Run First-Innings Score Projection"):
                    first_safe_runs, first_boundary_runs = (
                        fetch_phase_run_samples(1, balls_remaining)
                    )
                    innings_projection = simulate_dynamic_over(
                        model,
                        sim_batter,
                        sim_non_striker,
                        sim_bowler,
                        balls_remaining,
                        wickets_down,
                        current_score,
                        0,
                        crr,
                        0.0,
                        first_safe_runs,
                        first_boundary_runs,
                        innings=1,
                        batting_hands=batting_hands,
                        bowling_types=bowling_types,
                        simulations=400,
                        replacement_batter=sim_incoming_batter,
                        max_deliveries=balls_remaining,
                        striker_recent_boundary_rate=float(
                            recent_event_form[
                                "STRIKER_RECENT_BOUNDARY_RATE"
                            ]
                        ),
                        bowler_recent_boundary_rate=float(
                            recent_event_form[
                                "BOWLER_RECENT_BOUNDARY_RATE"
                            ]
                        ),
                        bowler_recent_dismissal_rate=float(
                            recent_event_form[
                                "BOWLER_RECENT_DISMISSAL_RATE"
                            ]
                        ),
                        striker_recent_run_rate=float(
                            recent_event_form["STRIKER_RECENT_RUN_RATE"]
                        ),
                        bowler_recent_run_rate=float(
                            recent_event_form["BOWLER_RECENT_RUN_RATE"]
                        ),
                        venue=sim_venue,
                        venue_recent_run_rate=venue_recent_run_rate,
                    )
                    st.markdown(
                        '<p class="section-label"><span class="tick"></span>'
                        'First-Innings Projection</p>',
                        unsafe_allow_html=True,
                    )
                    projection_columns = st.columns(4)
                    projection_columns[0].metric(
                        "Projected Score",
                        f"{innings_projection['expected_final_score']:.0f}",
                    )
                    projection_columns[1].metric(
                        "80% Range",
                        f"{innings_projection['final_score_low']:.0f}–"
                        f"{innings_projection['final_score_high']:.0f}",
                    )
                    projection_columns[2].metric(
                        "Reach 180",
                        f"{innings_projection['reach_180_probability'] * 100:.1f}%",
                    )
                    projection_columns[3].metric(
                        "Reach 200",
                        f"{innings_projection['reach_200_probability'] * 100:.1f}%",
                    )
                    venue_par = fetch_venue_par_stats(sim_venue)
                    if venue_par:
                        projected_score = innings_projection[
                            "expected_final_score"
                        ]
                        if projected_score < venue_par["lower_score"]:
                            assessment = "Below par"
                        elif projected_score > venue_par["upper_score"]:
                            assessment = "Above par"
                        else:
                            assessment = "Competitive"
                        st.info(
                            f"{sim_venue} recent par: {venue_par['par_score']:.0f} "
                            f"(middle 50%: {venue_par['lower_score']:.0f}–"
                            f"{venue_par['upper_score']:.0f}) · "
                            f"projection assessment: {assessment} · "
                            f"{venue_par['matches']} historical matches."
                        )
                    st.caption(
                        "Dynamic first-innings projection from 400 simulations "
                        "using first-innings phase history."
                    )
            except Exception as ml_err:
                st.error(f"⚠️ ML Prediction Error: {str(ml_err)}")

    with tab2:
        st.markdown('<p class="section-label"><span class="tick"></span>Opposing Franchise Matchup Matrix</p>', unsafe_allow_html=True)
        innings_context = st.selectbox(
            "Innings Context",
            ["Chasing", "Batting First"],
            index=0,
            help="Batting First uses historical first-innings deliveries and no required run rate.",
        )
        matrix_innings = 1 if innings_context == "Batting First" else 2
        matrix_rrr = 0.0 if matrix_innings == 1 else rrr
        t_col1, t_col2 = st.columns(2)
        with t_col1: my_team = st.selectbox("Your Analytics Context Franchise", all_teams, index=0)
        with t_col2: opposing_team = st.selectbox("Target Opponent Context", [t for t in all_teams if t != my_team], index=0)

        available_batters = sorted(
            df_rosters[df_rosters['TEAM'] == my_team]['PLAYER'].unique()
        )
        historical_batting_order = get_historical_batting_order(
            my_team, tuple(available_batters)
        )
        available_bowlers = sorted(df_rosters[(df_rosters['TEAM'] == opposing_team) & (df_rosters['ROLE'] == 'bowler')]['PLAYER'].unique())
        previous_over_bowler = st.selectbox(
            "Bowler Used in Previous Over",
            ["None"] + available_bowlers,
            index=0,
            help="The previous-over bowler cannot bowl consecutive overs.",
        )
        bowler_overs_used = {}
        with st.expander("Bowling Quotas", expanded=False):
            st.caption(
                "Set overs already completed. Bowlers at four overs are "
                "excluded from recommendations."
            )
            quota_columns = st.columns(3)
            for bowler_index, bowler_name in enumerate(available_bowlers):
                with quota_columns[bowler_index % 3]:
                    bowler_overs_used[bowler_name] = st.number_input(
                        bowler_name,
                        min_value=0,
                        max_value=4,
                        value=0,
                        step=1,
                        key=f"quota_{opposing_team}_{bowler_name}",
                    )
        recommendation_bowlers = [
            bowler_name
            for bowler_name in available_bowlers
            if bowler_overs_used.get(bowler_name, 0) < 4
            and bowler_name != previous_over_bowler
        ]
        if not recommendation_bowlers:
            st.warning(
                "No eligible bowlers remain after quota and previous-over constraints."
            )

        p_col1, p_col2 = st.columns(2)
        with p_col1: batter_1 = st.selectbox("Franchise Batter A", available_batters, index=0)
        with p_col2: batter_2 = st.selectbox("Franchise Batter B", [b for b in available_batters if b != batter_1], index=min(1, len(available_batters)-1))
        incoming_batters = [
            batter
            for batter in historical_batting_order
            if batter not in (batter_1, batter_2)
        ]
        if incoming_batters:
            st.caption(
                "Automatic historical batting order: "
                + " → ".join(incoming_batters[:6])
            )
        available_plan_overs = sum(
            max(4 - bowler_overs_used.get(bowler, 0), 0)
            for bowler in available_bowlers
        )
        maximum_plan_overs = max(
            1,
            min(
                5,
                int(np.ceil(max(balls_remaining, 1) / 6)),
                max(available_plan_overs, 1),
            ),
        )
        plan_overs = st.selectbox(
            "Overs to Optimize",
            options=list(range(1, maximum_plan_overs + 1)),
            index=min(3, maximum_plan_overs) - 1,
            help="Plans up to five future overs using legal bowling sequences.",
        )

        batter_1_eligible = bool(
            df_rosters[
                (df_rosters['TEAM'] == my_team)
                & (df_rosters['PLAYER'] == batter_1)
            ]['ANALYTICS_ELIGIBLE'].any()
        )
        batter_2_eligible = bool(
            df_rosters[
                (df_rosters['TEAM'] == my_team)
                & (df_rosters['PLAYER'] == batter_2)
            ]['ANALYTICS_ELIGIBLE'].any()
        )
        batters_have_evidence = batter_1_eligible and batter_2_eligible
        if not batters_have_evidence:
            unsupported = [
                name
                for name, eligible in [
                    (batter_1, batter_1_eligible),
                    (batter_2, batter_2_eligible),
                ]
                if not eligible
            ]
            st.warning(
                "Analytics unavailable for "
                + ", ".join(unsupported)
                + ": active player, but fewer than 60 career batting deliveries."
            )

        st.caption(f"Context: {balls_remaining} balls left · {wickets_down} wkts down · "
                   f"CRR {crr:.2f} · RRR {rrr:.2f} (live state set in the Live In-Game Metrics tab)")
        st.caption(
            "Probability history: "
            + (
                "first-innings deliveries; required run rate excluded."
                if matrix_innings == 1
                else "second-innings chase deliveries."
            )
        )
        st.markdown('<hr class="hr-fade" />', unsafe_allow_html=True)
        analysis_records = []
        if batters_have_evidence:
            matrix_evidence, matrix_profiles = get_matrix_evidence(
                tuple(
                    dict.fromkeys(
                        (batter_1, batter_2, *incoming_batters[:4])
                    )
                ),
                tuple(recommendation_bowlers),
                matrix_innings,
            )
        else:
            matrix_evidence, matrix_profiles = pd.DataFrame(), pd.DataFrame()
        evidence_lookup = {
            (row['STRIKER'], row['BOWLER']): row
            for _, row in matrix_evidence.iterrows()
        }
        profile_lookup = {
            row['BOWLER']: int(row['WICKETS_TAKEN'])
            for _, row in matrix_profiles.iterrows()
        }
        prediction_keys = []
        prediction_frames = []
        matrix_venue_run_rate = fetch_venue_event_form(sim_venue)
        if batters_have_evidence:
            for bowler in recommendation_bowlers:
                for batter in (batter_1, batter_2):
                    recent_form = evidence_lookup.get((batter, bowler), {})
                    prediction_keys.append((batter, bowler))
                    prediction_frames.append(
                        build_event_feature_frame(
                            batter,
                            bowler,
                            balls_remaining,
                            wickets_down,
                            crr,
                            matrix_rrr,
                            innings=matrix_innings,
                            current_team_score=current_score,
                            batting_hand=batting_hands.get(batter, "Unknown"),
                            bowling_type=bowling_types.get(bowler, "Unknown"),
                            striker_recent_boundary_rate=float(
                                recent_form.get(
                                    "STRIKER_RECENT_BOUNDARY_RATE", 0
                                )
                                or 0
                            ),
                            bowler_recent_boundary_rate=float(
                                recent_form.get(
                                    "BOWLER_RECENT_BOUNDARY_RATE", 0
                                )
                                or 0
                            ),
                            bowler_recent_dismissal_rate=float(
                                recent_form.get(
                                    "BOWLER_RECENT_DISMISSAL_RATE", 0
                                )
                                or 0
                            ),
                            striker_recent_run_rate=float(
                                recent_form.get(
                                    "STRIKER_RECENT_RUN_RATE", 0
                                )
                                or 0
                            ),
                            bowler_recent_run_rate=float(
                                recent_form.get(
                                    "BOWLER_RECENT_RUN_RATE", 0
                                )
                                or 0
                            ),
                            venue=sim_venue,
                            venue_recent_run_rate=matrix_venue_run_rate,
                        )
                    )
        probability_lookup = {}
        outcome_probability_lookup = {}
        if prediction_frames:
            batch_features = pd.concat(prediction_frames, ignore_index=True)
            batch_probabilities = model.predict_proba(batch_features)
            class_index = {label: i for i, label in enumerate(model.classes_)}
            probability_lookup = {
                key: np.array(
                    [
                        probabilities[class_index[0]],
                        probabilities[class_index[1]],
                        probabilities[class_index[2]],
                    ]
                )
                for key, probabilities in zip(
                    prediction_keys, batch_probabilities
                )
            }
            if (
                hasattr(model, "predict_outcome_proba")
                and getattr(model, "run_model", None) is not None
            ):
                batch_outcome_probabilities = model.predict_outcome_proba(
                    batch_features
                )
                outcome_probability_lookup = dict(
                    zip(prediction_keys, batch_outcome_probabilities)
                )
        for bowler in recommendation_bowlers:
            if not batters_have_evidence:
                break
            try:
                h2h_1 = evidence_lookup.get((batter_1, bowler), {})
                matchup_balls_1 = float(h2h_1.get('TOTAL_BALLS', 0))
                p1 = probability_lookup[(batter_1, bowler)]
                matchup_boundaries_1 = float(h2h_1.get('TOTAL_BOUNDARIES', 0))
                matchup_wickets_1 = float(h2h_1.get('TOTAL_WICKETS', 0))
                p1_boundary = blend_model_with_matchup_evidence(
                    p1[1], matchup_boundaries_1, matchup_balls_1
                )
                p1_wicket = blend_model_with_matchup_evidence(
                    p1[2], matchup_wickets_1, matchup_balls_1
                )
                outcome_1 = outcome_probability_lookup.get(
                    (batter_1, bowler)
                )
                if outcome_1 is not None:
                    p1_dot = blend_model_with_matchup_evidence(
                        outcome_1[0],
                        float(h2h_1.get('TOTAL_DOTS', 0)),
                        matchup_balls_1,
                    )
                    p1_expected_runs = blend_mean_with_matchup_evidence(
                        outcome_1 @ model.outcome_run_values_,
                        float(h2h_1.get('TOTAL_RUNS', 0)),
                        matchup_balls_1,
                    )
                else:
                    p1_dot = np.nan
                    p1_expected_runs = np.nan

                h2h_2 = evidence_lookup.get((batter_2, bowler), {})
                matchup_balls_2 = float(h2h_2.get('TOTAL_BALLS', 0))
                p2 = probability_lookup[(batter_2, bowler)]
                matchup_boundaries_2 = float(h2h_2.get('TOTAL_BOUNDARIES', 0))
                matchup_wickets_2 = float(h2h_2.get('TOTAL_WICKETS', 0))
                p2_boundary = blend_model_with_matchup_evidence(
                    p2[1], matchup_boundaries_2, matchup_balls_2
                )
                p2_wicket = blend_model_with_matchup_evidence(
                    p2[2], matchup_wickets_2, matchup_balls_2
                )
                outcome_2 = outcome_probability_lookup.get(
                    (batter_2, bowler)
                )
                if outcome_2 is not None:
                    p2_dot = blend_model_with_matchup_evidence(
                        outcome_2[0],
                        float(h2h_2.get('TOTAL_DOTS', 0)),
                        matchup_balls_2,
                    )
                    p2_expected_runs = blend_mean_with_matchup_evidence(
                        outcome_2 @ model.outcome_run_values_,
                        float(h2h_2.get('TOTAL_RUNS', 0)),
                        matchup_balls_2,
                    )
                else:
                    p2_dot = np.nan
                    p2_expected_runs = np.nan

                analysis_records.append({
                    "Active Opposing Bowler": bowler,
                    f"{batter_1} Wicket Danger %": round(p1_wicket * 100, 1),
                    f"{batter_1} Boundary Leak %": round(p1_boundary * 100, 1),
                    f"{batter_1} Dot Ball %": round(p1_dot * 100, 1),
                    f"{batter_1} Expected Runs/Ball": round(
                        p1_expected_runs, 2
                    ),
                    f"{batter_1} Evidence": (
                        f"{int(matchup_balls_1)} balls · "
                        f"{matchup_confidence_label(matchup_balls_1)}"
                    ),
                    f"{batter_2} Wicket Danger %": round(p2_wicket * 100, 1),
                    f"{batter_2} Boundary Leak %": round(p2_boundary * 100, 1),
                    f"{batter_2} Dot Ball %": round(p2_dot * 100, 1),
                    f"{batter_2} Expected Runs/Ball": round(
                        p2_expected_runs, 2
                    ),
                    f"{batter_2} Evidence": (
                        f"{int(matchup_balls_2)} balls · "
                        f"{matchup_confidence_label(matchup_balls_2)}"
                    ),
                    "Total Wickets Taken": profile_lookup.get(bowler, 0)
                })
            except Exception: continue

        if analysis_records:
            df_analysis = pd.DataFrame(analysis_records)
            danger_cols = [f"{batter_1} Wicket Danger %", f"{batter_2} Wicket Danger %"]
            leak_cols = [f"{batter_1} Boundary Leak %", f"{batter_2} Boundary Leak %"]
            dot_cols = [f"{batter_1} Dot Ball %", f"{batter_2} Dot Ball %"]
            expected_run_cols = [
                f"{batter_1} Expected Runs/Ball",
                f"{batter_2} Expected Runs/Ball",
            ]
            average_danger = df_analysis[danger_cols].mean(axis=1)
            average_leak = df_analysis[leak_cols].mean(axis=1)
            threat_percentile = average_danger.rank(pct=True)
            granular_available = not df_analysis[
                expected_run_cols
            ].isna().all().all()
            if granular_available:
                average_expected_runs = df_analysis[
                    expected_run_cols
                ].mean(axis=1)
                average_dot_rate = df_analysis[dot_cols].mean(axis=1)
                run_control_percentile = average_expected_runs.rank(
                    pct=True, ascending=False
                )
                dot_control_percentile = average_dot_rate.rank(pct=True)
                df_analysis["Recommendation Score"] = (
                    threat_percentile * 45
                    + run_control_percentile * 35
                    + dot_control_percentile * 20
                ).round(1)
            else:
                control_percentile = average_leak.rank(
                    pct=True, ascending=False
                )
                df_analysis["Recommendation Score"] = (
                    (threat_percentile + control_percentile) * 50
                ).round(1)
            median_danger = average_danger.median()
            median_leak = average_leak.median()
            evidence_columns = [
                f"{batter_1} Evidence",
                f"{batter_2} Evidence",
            ]
            explanations = []
            for row_index in df_analysis.index:
                reasons = []
                if average_danger.loc[row_index] >= median_danger:
                    reasons.append("above-median wicket threat")
                if average_leak.loc[row_index] <= median_leak:
                    reasons.append("below-median boundary risk")
                if granular_available:
                    if (
                        average_expected_runs.loc[row_index]
                        <= average_expected_runs.median()
                    ):
                        reasons.append("below-median expected runs")
                    if (
                        average_dot_rate.loc[row_index]
                        >= average_dot_rate.median()
                    ):
                        reasons.append("above-median dot pressure")
                evidence_text = " / ".join(
                    str(df_analysis.loc[row_index, column])
                    for column in evidence_columns
                )
                reasons.append(f"evidence {evidence_text}")
                explanations.append("; ".join(reasons))
            df_analysis["Why"] = explanations
            df_analysis = df_analysis.sort_values(
                by="Recommendation Score", ascending=False
            ).reset_index(drop=True)
            best_bowler = df_analysis.iloc[0]
            st.success(
                f"Recommended bowler: {best_bowler['Active Opposing Bowler']} "
                f"· matchup score {best_bowler['Recommendation Score']:.1f}/100"
            )
            st.caption(
                "Recommendation score is a relative rank across the available "
                "bowlers, balancing dismissal threat, expected runs conceded, "
                "and dot-ball pressure against both selected batters."
            )
            try:
                styled_df = (
                    df_analysis.style
                    .background_gradient(subset=danger_cols, cmap="Reds", vmin=0, vmax=60)
                    .background_gradient(subset=leak_cols, cmap="Greens", vmin=0, vmax=30)
                    .background_gradient(subset=dot_cols, cmap="Blues", vmin=0, vmax=70)
                    .format({
                        **{c: "{:.1f}%" for c in danger_cols + leak_cols + dot_cols},
                        **{c: "{:.2f}" for c in expected_run_cols},
                    })
                    .set_properties(**{"font-family": "Inter, sans-serif", "font-size": "13px"})
                )
                st.dataframe(styled_df, use_container_width=True)
            except ImportError:
                # matplotlib not available for background_gradient — fall back to plain table
                st.dataframe(df_analysis, use_container_width=True)
            st.markdown(
                '<p class="section-label"><span class="tick"></span>'
                'Multi-Over Bowling Plan</p>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Beam search evaluates legal sequences using projected runs, "
                "wickets, boundary risk, strike context, quotas, and the "
                "previous-over restriction."
            )
            if st.button(
                "Generate Bowling Plan",
                disabled=not bool(recommendation_bowlers),
                use_container_width=True,
            ):
                optimizer_context = {}
                for matchup_key, evidence_row in evidence_lookup.items():
                    context = dict(evidence_row)
                    context["VENUE_RECENT_RUN_RATE"] = matrix_venue_run_rate
                    optimizer_context[matchup_key] = context
                context_signature = tuple(
                    sorted(
                        (
                            matchup_key,
                            round(
                                float(context.get("TOTAL_BALLS", 0) or 0),
                                1,
                            ),
                            round(
                                float(
                                    context.get(
                                        "STRIKER_RECENT_RUN_RATE", 0
                                    )
                                    or 0
                                ),
                                4,
                            ),
                            round(
                                float(
                                    context.get(
                                        "BOWLER_RECENT_RUN_RATE", 0
                                    )
                                    or 0
                                ),
                                4,
                            ),
                        )
                        for matchup_key, context in optimizer_context.items()
                    )
                )
                planner_cache_key = (
                    tuple(available_bowlers),
                    batter_1,
                    batter_2,
                    tuple(incoming_batters),
                    int(balls_remaining),
                    int(wickets_down),
                    int(current_score),
                    round(float(crr), 2),
                    round(float(matrix_rrr), 2),
                    tuple(sorted(bowler_overs_used.items())),
                    previous_over_bowler,
                    int(plan_overs),
                    int(matrix_innings),
                    int(target_to_chase),
                    sim_venue,
                    context_signature,
                    os.path.getmtime(MODEL_PATH),
                )
                planner_cache = st.session_state.setdefault(
                    "bowling_plan_cache", {}
                )
                if planner_cache_key in planner_cache:
                    bowling_plan = planner_cache[planner_cache_key]
                    st.caption("Loaded the unchanged plan from session cache.")
                else:
                    bowling_plan = load_cached_plan(
                        PLAN_CACHE_PATH, planner_cache_key
                    )
                    if bowling_plan is not None:
                        st.caption("Loaded the unchanged plan from persistent cache.")
                    else:
                        with st.spinner("Evaluating legal bowling sequences..."):
                            bowling_plan = optimize_bowling_plan(
                                model,
                                available_bowlers,
                                batter_1,
                                batter_2,
                                balls_remaining,
                                wickets_down,
                                current_score,
                                crr,
                                matrix_rrr,
                                overs_used=bowler_overs_used,
                                previous_bowler=previous_over_bowler
                                if previous_over_bowler != "None"
                                else None,
                                overs_to_plan=plan_overs,
                                innings=matrix_innings,
                                target_score=target_to_chase,
                                venue=sim_venue,
                                incoming_batters=incoming_batters,
                                batting_hands=batting_hands,
                                bowling_types=bowling_types,
                                matchup_context=optimizer_context,
                                win_model=win_model,
                                beam_width=6,
                                monte_carlo_simulations=80,
                            )
                        store_cached_plan(
                            PLAN_CACHE_PATH, planner_cache_key, bowling_plan
                        )
                    planner_cache[planner_cache_key] = bowling_plan
                    while len(planner_cache) > 8:
                        planner_cache.pop(next(iter(planner_cache)))
                primary_plan = bowling_plan["primary"]
                alternative_plan = bowling_plan["alternative"]
                if primary_plan is None:
                    st.warning(
                        "No legal sequence is available for the selected quotas."
                    )
                else:
                    primary_mc = primary_plan.get("monte_carlo", {})
                    plan_a, plan_b = st.columns(2)
                    with plan_a:
                        st.success(
                            "Plan A: " + " → ".join(primary_plan["sequence"])
                        )
                        plan_metrics = st.columns(3)
                        plan_metrics[0].metric(
                            "Expected Runs",
                            f"{primary_mc.get('expected_runs', primary_plan['expected_runs']):.1f}",
                        )
                        plan_metrics[1].metric(
                            "Expected Wickets",
                            f"{primary_mc.get('expected_wickets', primary_plan['expected_wickets']):.2f}",
                        )
                        projected_win = primary_plan[
                            "projected_batting_win_probability"
                        ]
                        if projected_win is not None:
                            plan_metrics[2].metric(
                                "Batting Win %",
                                f"{projected_win * 100:.1f}%",
                            )
                        else:
                            plan_metrics[2].metric(
                                "Projected Score",
                                f"{primary_plan['score']:.0f}",
                            )
                    with plan_b:
                        if alternative_plan is not None:
                            alternative_mc = alternative_plan.get(
                                "monte_carlo", {}
                            )
                            st.info(
                                "Plan B — lower boundary risk: "
                                + " → ".join(
                                    alternative_plan["sequence"]
                                )
                            )
                            st.metric(
                                "Expected Runs",
                                f"{alternative_mc.get('expected_runs', alternative_plan['expected_runs']):.1f}",
                            )
                        else:
                            st.info(
                                "No materially different legal alternative."
                            )
                    validation = bowling_plan.get("validation")
                    if validation is not None:
                        validation_columns = st.columns(4)
                        validation_columns[0].metric(
                            "Plan A Run Range",
                            f"{primary_mc['runs_low']:.0f}–"
                            f"{primary_mc['runs_high']:.0f}",
                        )
                        validation_columns[1].metric(
                            "Plan A Wickets",
                            f"{primary_mc['expected_wickets']:.2f}",
                        )
                        validation_columns[2].metric(
                            "Plan A Beats B",
                            f"{validation['primary_better_probability'] * 100:.1f}%",
                        )
                        validation_columns[3].metric(
                            "Simulations",
                            f"{validation['simulations']}",
                        )
                        if (
                            primary_mc[
                                "projected_win_probability"
                            ]
                            is not None
                            and primary_mc.get(
                                "projected_win_probability_low"
                            )
                            is not None
                            and primary_mc.get(
                                "projected_win_probability_high"
                            )
                            is not None
                        ):
                            st.caption(
                                "Plan A projected batting win range: "
                                f"{primary_mc['projected_win_probability_low'] * 100:.1f}%–"
                                f"{primary_mc['projected_win_probability_high'] * 100:.1f}%."
                            )
                        if validation["preferred_plan"] == "B":
                            st.warning(
                                "Risk-adjusted preference: Plan B. Plan A's "
                                "advantage is unstable and Plan B has materially "
                                "lower downside run risk."
                            )
                        elif validation["stable"]:
                            st.success(
                                "Plan A is stable across the simulated match paths."
                            )
                        else:
                            st.info(
                                "Plan A and Plan B are statistically close; "
                                "treat the ranking as low confidence."
                            )
                    plan_rows = []
                    for over_number, step in enumerate(
                        primary_plan["steps"], start=1
                    ):
                        plan_rows.append(
                            {
                                "Planned Over": over_number,
                                "Bowler": step["bowler"],
                                "Expected Runs": step["expected_runs"],
                                "Expected Wickets": step[
                                    "expected_wickets"
                                ],
                                "Dot Ball %": step["dot_probability"] * 100,
                                "Boundary in Over %": (
                                    step["boundary_in_over"] * 100
                                ),
                                "Direct Evidence": (
                                    f"{step['evidence_balls']:.0f} avg balls "
                                    f"({matchup_confidence_label(step['evidence_balls'])})"
                                ),
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(plan_rows).style.format(
                            {
                                "Expected Runs": "{:.1f}",
                                "Expected Wickets": "{:.2f}",
                                "Dot Ball %": "{:.1f}%",
                                "Boundary in Over %": "{:.1f}%",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    export_columns = st.columns(2)
                    export_columns[0].download_button(
                        "Download plan CSV",
                        build_plan_csv(plan_rows),
                        file_name="bowling_plan.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    export_columns[1].download_button(
                        "Download plan PDF",
                        build_plan_pdf(
                            primary_plan,
                            validation,
                            plan_rows,
                            (
                                f"{my_team} vs {opposing_team}; "
                                f"{balls_remaining} balls left; "
                                f"{wickets_down} wickets down"
                            ),
                        ),
                        file_name="bowling_plan.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            st.markdown('<p class="section-label"><span class="tick"></span>Visual Matchup Threat Spectrum</p>', unsafe_allow_html=True)
            render_matchup_threat_chart(df_analysis, batter_1)

    if show_model_validation:
        with tab3:
            st.markdown(
                '<p class="section-label"><span class="tick"></span>'
                'Held-Out Match Backtesting</p>',
                unsafe_allow_html=True,
            )
            if not model_metrics:
                st.info("Run src/models.py to generate backtest metrics.")
            else:
                if drift_status["retrain_recommended"]:
                    st.warning(
                        "Automatic retraining alert: "
                        + " ".join(drift_status["reasons"])
                    )
                else:
                    st.success(
                        "Drift monitor: current validation remains within "
                        "configured tolerances."
                    )
                has_outcome_metrics = "outcome_log_loss" in model_metrics
                metric_columns = st.columns(7 if has_outcome_metrics else 5)
                metric_columns[0].metric(
                    "Event Log Loss",
                    f"{model_metrics['event_log_loss']:.4f}",
                )
                metric_columns[1].metric(
                    "Boundary Brier",
                    f"{model_metrics['boundary_brier']:.4f}",
                )
                metric_columns[2].metric(
                    "Dismissal Brier",
                    f"{model_metrics['dismissal_brier']:.4f}",
                )
                metric_columns[3].metric(
                    "Win Log Loss",
                    f"{model_metrics['win_log_loss']:.4f}",
                )
                metric_columns[4].metric(
                    "Win Brier",
                    f"{model_metrics['win_brier']:.4f}",
                )
                if has_outcome_metrics:
                    metric_columns[5].metric(
                        "Outcome Log Loss",
                        f"{model_metrics['outcome_log_loss']:.4f}",
                    )
                    metric_columns[6].metric(
                        "Expected Runs MAE",
                        f"{model_metrics['expected_runs_mae']:.3f}",
                    )
                st.caption(
                    f"Evaluated on {model_metrics['event_test_deliveries']:,} "
                    f"deliveries from held-out matches · latest season "
                    f"{model_metrics['latest_season']} · four-season probability half-life."
                )
                if "latest_season_event_log_loss" in model_metrics:
                    st.info(
                        f"True forward test — trained on earlier seasons and tested "
                        f"only on {model_metrics['latest_season']}: log loss "
                        f"{model_metrics['latest_season_event_log_loss']:.4f} across "
                    f"{model_metrics['latest_season_test_deliveries']:,} deliveries."
                )
                if "rolling_latest_season_log_loss" in model_metrics:
                    st.caption(
                        f"Rolling current-season calibration log loss: "
                        f"{model_metrics['rolling_latest_season_log_loss']:.4f}."
                    )
                if "selected_event_model" in model_metrics:
                    st.caption(
                        f"Selected event model: "
                        f"{model_metrics['selected_event_model']} · candidate log loss "
                        f"linear {model_metrics['linear_candidate_log_loss']:.4f}, "
                        f"boosted {model_metrics['boosted_candidate_log_loss']:.4f}."
                    )
                if "selected_run_model" in model_metrics:
                    baseline_loss = model_metrics["naive_outcome_log_loss"]
                    model_loss = model_metrics["outcome_log_loss"]
                    comparison = (
                        "better"
                        if model_loss < baseline_loss
                        else "not better"
                    )
                    st.info(
                        f"Granular model: {model_metrics['selected_run_model']} "
                        f"with held-out log loss {model_loss:.4f}; naive "
                        f"historical baseline {baseline_loss:.4f} "
                        f"({comparison} than baseline)."
                    )
                    candidate_text = ", ".join(
                        f"{name} {loss:.4f}"
                        for name, loss in model_metrics[
                            "run_candidate_log_losses"
                        ].items()
                    )
                    st.caption(
                        "Match-grouped run-model validation: "
                        + candidate_text
                    )
                if "latest_season_outcome_log_loss" in model_metrics:
                    st.caption(
                        f"Forward {model_metrics['latest_season']} granular "
                        f"log loss {model_metrics['latest_season_outcome_log_loss']:.4f}; "
                        f"expected-runs MAE "
                        f"{model_metrics['latest_season_expected_runs_mae']:.3f}."
                    )
                if "rolling_outcome_log_loss" in model_metrics:
                    adapter_status = (
                        "deployed"
                        if model_metrics["drift_adapter_deployed"]
                        else "not deployed"
                    )
                    st.caption(
                        f"Rolling drift recalibration: log loss "
                        f"{model_metrics['rolling_outcome_log_loss']:.4f}, "
                        f"expected-runs MAE "
                        f"{model_metrics['rolling_expected_runs_mae']:.3f}; "
                        f"{adapter_status}."
                    )
                if "phase_expected_runs_mae" in model_metrics:
                    phase_text = ", ".join(
                        f"{phase} {error:.3f}"
                        for phase, error in model_metrics[
                            "phase_expected_runs_mae"
                        ].items()
                    )
                    st.caption("Expected-runs MAE by phase: " + phase_text)
                if "outcome_calibration" in model_metrics:
                    outcome_names = {
                        "0": "Dot",
                        "1": "1 Run",
                        "2": "2 Runs",
                        "3": "3 Runs",
                        "4": "Four",
                        "6": "Six",
                        "7": "Dismissal",
                    }
                    calibration_rows = []
                    for outcome, values in model_metrics[
                        "outcome_calibration"
                    ].items():
                        calibration_rows.append(
                            {
                                "Outcome": outcome_names.get(
                                    outcome, outcome
                                ),
                                "Observed %": (
                                    values["observed_rate"] * 100
                                ),
                                "Predicted %": (
                                    values["predicted_rate"] * 100
                                ),
                                "Calibration Gap %": (
                                    values["calibration_gap"] * 100
                                ),
                                "Brier": values["brier"],
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(calibration_rows).style.format(
                            {
                                "Observed %": "{:.2f}%",
                                "Predicted %": "{:.2f}%",
                                "Calibration Gap %": "{:.2f}%",
                                "Brier": "{:.4f}",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
