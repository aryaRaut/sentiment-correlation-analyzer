"""
Streamlit Page: Model Performance Monitoring & Analytics (Page 4)

Comprehensive model tracking dashboard featuring:
- Core Accuracy Tabs (Yesterday, Past 7 Days, Past 30 Days)
- P&L Backtest Simulation (AI Strategy vs Buy & Hold)
- Confusion Matrix Heatmap
- 5-Day Rolling Accuracy Trend
- Top "Big Miss" Forensic Headline Analysis
- Sector-Wise Accuracy Bar Chart
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_processed_data, load_news_data, SECTOR_MAP

# Page Configuration
st.set_page_config(
    page_title="Model Performance Monitoring",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Premium Dark Theme Financial Dashboard)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0f1724 0%, #1a2332 100%);
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

h1 { font-size: 2rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.25rem !important; }

/* Section Header */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #ffffff;
    padding-bottom: 12px;
    margin-bottom: 20px;
    border-bottom: 2px solid rgba(41, 98, 255, 0.3);
    display: flex;
    align-items: center;
    gap: 12px;
}

/* Metric Card */
.metric-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border-radius: 12px;
    padding: 20px 24px;
    border-left: 4px solid #2962ff;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 12px;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}
.metric-card .card-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    font-weight: 500;
}
.metric-card .card-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 4px;
}
.metric-card .card-value-blue { color: #2962ff; }
.metric-card .card-value-green { color: #22c55e; }
.metric-card .card-value-red { color: #ef4444; }
.metric-card .card-value-gold { color: #f9a825; }

/* Chart Container */
.chart-container {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 16px;
}

/* Divider */
.custom-divider {
    margin: 28px 0;
    border: 0;
    height: 1px;
    background: linear-gradient(to right, transparent, rgba(41, 98, 255, 0.3), transparent);
}

/* Tabs Customization */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: rgba(30, 41, 59, 0.5);
    border-radius: 8px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 16px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #2962ff;
    color: #ffffff;
}

/* Dataframe Styling */
.stDataFrame {
    background: rgba(30, 41, 59, 0.5) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
.stDataFrame thead {
    background: rgba(41, 98, 255, 0.2) !important;
}
.stDataFrame tbody tr:hover {
    background: rgba(41, 98, 255, 0.1) !important;
}

/* Selectbox Styling */
.stSelectbox > div > div {
    background-color: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
.stSelectbox > div > div:hover {
    border-color: #2962ff !important;
}

/* Metric Boxes */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

/* Success/Warning/Info Boxes */
.stAlert {
    border-radius: 12px !important;
    background-color: rgba(30, 41, 59, 0.8) !important;
    border-left: 4px solid #2962ff !important;
}

/* Expander Styling (keep but minimal) */
.streamlit-expanderHeader {
    background-color: rgba(30, 41, 59, 0.5) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
}
.streamlit-expanderContent {
    background-color: rgba(30, 41, 59, 0.3) !important;
    border-radius: 0 0 8px 8px !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb {
    background: #2962ff;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #1a73e8;
}

/* Success Message */
.success-message {
    background: rgba(34, 197, 94, 0.1);
    border-left: 4px solid #22c55e;
    padding: 12px 16px;
    border-radius: 8px;
    color: #22c55e;
    font-weight: 500;
}

/* Table Container */
.table-container {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    padding: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)

# Main Title & Subheader
st.markdown("""
<div style="padding-bottom: 16px; border-bottom: 2px solid rgba(41, 98, 255, 0.3); margin-bottom: 24px;">
    <h1 style="margin: 0; padding: 0;">🎯 Model Performance Monitoring & Analytics</h1>
    <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 0.95rem;">Empirical Validation of XGBoost Sentiment Model: Real-Time Accuracy, P&L Backtest, Confusion Matrix & Sector Drift</p>
</div>
""", unsafe_allow_html=True)

# Load Data via Cached Loader
with st.spinner("Loading performance analytics & model inferences..."):
    df = load_processed_data()
    news_df = load_news_data()

if df.empty:
    st.error("Unable to load processed dataset. Please check data/processed/processed_dataset.csv.")
    st.stop()

# Available Stocks Dropdown (Top Control)
stocks_list = sorted(list(df["stock"].unique()))
selected_stock = st.selectbox("📌 Select Stock Ticker", options=["All Stocks"] + stocks_list, index=0)

# Filter Dataset based on Selection
if selected_stock != "All Stocks":
    df_filtered = df[df["stock"] == selected_stock].copy()
else:
    df_filtered = df.copy()

df_filtered = df_filtered.sort_values("date").reset_index(drop=True)
max_date = df_filtered["date"].max()

def fetch_headlines_for_misses(misses_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """Fetches relevant news headlines for big miss dates and stocks."""
    miss_results = []
    for _, row in misses_df.iterrows():
        m_date = row["date"]
        m_stock = row["stock"]
        actual_str = "UP 🟢" if row["actual_direction"] == 1 else "DOWN 🔴"
        pred_str = "UP 🟢" if row["predicted_direction"] == 1 else "DOWN 🔴"
        ret_val = row["next_day_return"]
        ret_str = f"{ret_val:+.2%}"
        
        matching_news = pd.DataFrame()
        if not news_df.empty:
            matching_news = news_df[(news_df["date"] == m_date) & (news_df["stock"] == m_stock)]
        
        headlines = []
        if not matching_news.empty:
            for _, n_row in matching_news.head(2).iterrows():
                h_text = n_row.get("Headline", n_row.get("headline", ""))
                h_src = n_row.get("Source", n_row.get("source", "News"))
                headlines.append(f"• [{h_src}] {h_text}")
        
        miss_results.append({
            "Date": str(m_date),
            "Stock": m_stock,
            "Actual Direction": actual_str,
            "Predicted Direction": pred_str,
            "Actual Return": ret_str,
            "Headlines": "\n".join(headlines) if headlines else "No news logged for this date."
        })
    return pd.DataFrame(miss_results)

# ---------------------------------------------------------
# Core Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📅 Yesterday", "📆 Past Week (7 Days)", "📅 Past Month (30 Days)"])

# ---------------------------------------------------------
# TAB 1: Yesterday Performance
# ---------------------------------------------------------
with tab1:
    st.markdown(f'<div class="section-header">📅 Yesterday\'s Performance Overview ({max_date})</div>', unsafe_allow_html=True)
    
    # Filter for yesterday's data
    yesterday_df = df_filtered[df_filtered["date"] == max_date]
    yesterday_df = yesterday_df.dropna(subset=["actual_direction", "predicted_direction", "next_day_return"])
    
    if yesterday_df.empty:
        st.warning("⚠️ Insufficient valid data for Yesterday.")
    elif selected_stock != "All Stocks":
        y_row = yesterday_df.iloc[0]
        actual_up = y_row["actual_direction"] == 1
        pred_up = y_row["predicted_direction"] == 1
        was_correct = y_row["actual_direction"] == y_row["predicted_direction"]
        ret_pct = y_row["next_day_return"]
        
        c1, c2, c3 = st.columns(3)
        with c1:
            act_text = "UP 🟢" if actual_up else "DOWN 🔴"
            act_color = "card-value-green" if actual_up else "card-value-red"
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Actual Movement</div>
                <div class="card-value {act_color}">{act_text}</div>
                <div class="card-subtext">Return: {ret_pct:+.2%}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            pred_text = "UP 🟢" if pred_up else "DOWN 🔴"
            pred_color = "card-value-green" if pred_up else "card-value-red"
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Model Prediction</div>
                <div class="card-value {pred_color}">{pred_text}</div>
                <div class="card-subtext">XGBoost Signal</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            corr_text = "YES 🟢" if was_correct else "NO 🔴"
            corr_color = "card-value-green" if was_correct else "card-value-red"
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Was it Right?</div>
                <div class="card-value {corr_color}">{corr_text}</div>
                <div class="card-subtext">Direction Match</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        total_stocks = len(yesterday_df)
        correct_preds = int((yesterday_df["actual_direction"] == yesterday_df["predicted_direction"]).sum())
        acc_pct = (correct_preds / total_stocks * 100) if total_stocks > 0 else 0
        actual_ups = int((yesterday_df["actual_direction"] == 1).sum())
        pred_ups = int((yesterday_df["predicted_direction"] == 1).sum())
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Universe Yesterday Accuracy</div>
                <div class="card-value card-value-blue">{acc_pct:.1f}%</div>
                <div class="card-subtext">Across {total_stocks} Stocks</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Correct Predictions</div>
                <div class="card-value card-value-green">{correct_preds} / {total_stocks}</div>
                <div class="card-subtext">Successful Hits</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Actual Up vs Pred Up</div>
                <div class="card-value card-value-gold">{actual_ups} Up / {pred_ups} Pred</div>
                <div class="card-subtext">Bullish Distribution</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### Yesterday's Prediction Breakdown Across Universe")
        y_summary = yesterday_df[["stock", "actual_direction", "predicted_direction", "next_day_return"]].copy()
        y_summary["Actual Movement"] = y_summary["actual_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
        y_summary["Model Prediction"] = y_summary["predicted_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
        y_summary["Correct?"] = (y_summary["actual_direction"] == y_summary["predicted_direction"]).apply(lambda x: "Yes 🟢" if x else "No 🔴")
        y_summary["Next-Day Return"] = y_summary["next_day_return"].apply(lambda x: f"{x:+.2%}")
        
        st.dataframe(
            y_summary[["stock", "Actual Movement", "Model Prediction", "Correct?", "Next-Day Return"]].rename(columns={"stock": "Stock Ticker"}),
            use_container_width=True,
            hide_index=True
        )

    # ❌ Biggest Misses Section (Visible Table, NO expander for table!)
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">❌ Biggest Misses</div>', unsafe_allow_html=True)

    misses_yest = yesterday_df[yesterday_df["actual_direction"] != yesterday_df["predicted_direction"]].copy()
    if misses_yest.empty:
        st.markdown('<div class="success-message">🎉 No prediction misses in this period!</div>', unsafe_allow_html=True)
    else:
        misses_yest["abs_return"] = misses_yest["next_day_return"].abs()
        top_misses_yest = misses_yest.sort_values("abs_return", ascending=False).head(5)

        miss_df = top_misses_yest[["date", "stock", "actual_direction", "predicted_direction", "next_day_return"]].copy()
        miss_df["Actual"] = miss_df["actual_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
        miss_df["Predicted"] = miss_df["predicted_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
        miss_df["Return"] = miss_df["next_day_return"].apply(lambda x: f"{x:+.2%}")

        st.dataframe(
            miss_df[["date", "stock", "Actual", "Predicted", "Return"]].rename(
                columns={"date": "Date", "stock": "Stock"}
            ),
            use_container_width=True,
            hide_index=True
        )

        miss_table_yest = fetch_headlines_for_misses(top_misses_yest, news_df)
        with st.expander("📰 View News Context for Misses", expanded=False):
            for _, mrow in miss_table_yest.iterrows():
                st.markdown(f"**{mrow['Date']} - {mrow['Stock']}**")
                st.caption(f"📰 {mrow['Headlines']}")

# ---------------------------------------------------------
# TAB 2: Past Week (7 Days)
# ---------------------------------------------------------
with tab2:
    st.markdown(f'<div class="section-header">📆 Past Week Performance (Last 7 Days ending {max_date})</div>', unsafe_allow_html=True)
    
    min_date_7d = max_date - pd.Timedelta(days=7)
    df_7d = df_filtered[df_filtered["date"] >= min_date_7d].copy()
    df_7d = df_7d.dropna(subset=["actual_direction", "predicted_direction", "next_day_return"])
    
    if len(df_7d) < 1:
        st.warning("⚠️ Insufficient valid data for the past week.")
    else:
        y_true_7d = df_7d["actual_direction"].astype(int)
        y_pred_7d = df_7d["predicted_direction"].astype(int)
        acc_7d = accuracy_score(y_true_7d, y_pred_7d) * 100
        
        # Weekly Accuracy Metric Card
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 20px;">
            <div class="card-title">📆 Weekly Accuracy (7 Days)</div>
            <div class="card-value card-value-gold">{acc_7d:.1f}%</div>
            <div class="card-subtext">Evaluated across past 7 trading days</div>
        </div>
        """, unsafe_allow_html=True)
        
        col_cm, col_info = st.columns([1, 1])
        with col_cm:
            st.markdown("### Confusion Matrix (Counts)")
            cm_7d = confusion_matrix(y_true_7d, y_pred_7d, labels=[0, 1])
            cm_df_7d = pd.DataFrame(
                cm_7d,
                index=["Actual Down (0)", "Actual Up (1)"],
                columns=["Predicted Down (0)", "Predicted Up (1)"]
            )
            st.dataframe(cm_df_7d.style.highlight_max(axis=None, color="rgba(34, 197, 94, 0.3)"), use_container_width=True)
            
        with col_info:
            st.markdown("### 7-Day Sample Summary")
            total_samples = len(df_7d)
            correct_samples = int((y_true_7d == y_pred_7d).sum())
            st.markdown(f"""
            <div class="chart-container" style="padding: 20px;">
                <p style="font-size: 1rem; color: #ffffff; margin-bottom: 8px;">• <strong>Total Evaluated Predictions:</strong> <span style="color: #2962ff;">{total_samples}</span></p>
                <p style="font-size: 1rem; color: #ffffff; margin-bottom: 8px;">• <strong>Correct Direction Predictions:</strong> <span style="color: #22c55e;">{correct_samples}</span></p>
                <p style="font-size: 1rem; color: #ffffff; margin-bottom: 0;">• <strong>Incorrect Direction Predictions:</strong> <span style="color: #ef4444;">{total_samples - correct_samples}</span></p>
            </div>
            """, unsafe_allow_html=True)

        # ❌ Biggest Misses Section (Visible Table, NO expander for table!)
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">❌ Biggest Misses</div>', unsafe_allow_html=True)

        misses_7d = df_7d[df_7d["actual_direction"] != df_7d["predicted_direction"]].copy()
        if misses_7d.empty:
            st.markdown('<div class="success-message">🎉 No prediction misses in this period!</div>', unsafe_allow_html=True)
        else:
            misses_7d["abs_return"] = misses_7d["next_day_return"].abs()
            top3_misses_7d = misses_7d.sort_values("abs_return", ascending=False).head(5)

            miss_df_7d = top3_misses_7d[["date", "stock", "actual_direction", "predicted_direction", "next_day_return"]].copy()
            miss_df_7d["Actual"] = miss_df_7d["actual_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
            miss_df_7d["Predicted"] = miss_df_7d["predicted_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
            miss_df_7d["Return"] = miss_df_7d["next_day_return"].apply(lambda x: f"{x:+.2%}")

            st.dataframe(
                miss_df_7d[["date", "stock", "Actual", "Predicted", "Return"]].rename(
                    columns={"date": "Date", "stock": "Stock"}
                ),
                use_container_width=True,
                hide_index=True
            )

            miss_table_7d = fetch_headlines_for_misses(top3_misses_7d, news_df)
            with st.expander("📰 View News Context for Misses", expanded=False):
                for _, mrow in miss_table_7d.iterrows():
                    st.markdown(f"**{mrow['Date']} - {mrow['Stock']}**")
                    st.caption(f"📰 {mrow['Headlines']}")

# ---------------------------------------------------------
# TAB 3: Past Month (30 Days)
# ---------------------------------------------------------
with tab3:
    st.markdown(f'<div class="section-header">📅 Past Month Deep-Dive Analytics (Last 30 Days ending {max_date})</div>', unsafe_allow_html=True)
    
    min_date_30d = max_date - pd.Timedelta(days=30)
    df_30d = df_filtered[df_filtered["date"] >= min_date_30d].copy()
    df_30d = df_30d.dropna(subset=["actual_direction", "predicted_direction", "next_day_return"])
    
    if len(df_30d) < 1:
        st.warning("⚠️ Insufficient valid data for the past month.")
    else:
        y_true_30d = df_30d["actual_direction"].astype(int)
        y_pred_30d = df_30d["predicted_direction"].astype(int)
        
        acc_30d = accuracy_score(y_true_30d, y_pred_30d) * 100
        prec_30d = precision_score(y_true_30d, y_pred_30d, zero_division=0) * 100
        rec_30d = recall_score(y_true_30d, y_pred_30d, zero_division=0) * 100
        f1_30d = f1_score(y_true_30d, y_pred_30d, zero_division=0) * 100
        
        # Monthly Accuracy & 3 Metric Cards
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 20px;">
            <div class="card-title">📅 Monthly Accuracy (30 Days)</div>
            <div class="card-value card-value-gold">{acc_30d:.1f}%</div>
            <div class="card-subtext">Overall 30-Day Predictive Precision</div>
        </div>
        """, unsafe_allow_html=True)
        
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Precision (Up Predictions)</div>
                <div class="card-value card-value-blue">{prec_30d:.1f}%</div>
                <div class="card-subtext">Positive Predictive Value</div>
            </div>
            """, unsafe_allow_html=True)
        with m_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Recall (Captured Ups)</div>
                <div class="card-value card-value-green">{rec_30d:.1f}%</div>
                <div class="card-subtext">True Positive Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with m_c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">F1-Score</div>
                <div class="card-value card-value-blue">{f1_30d:.1f}%</div>
                <div class="card-subtext">Harmonic Mean</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # P&L Simulation
        st.markdown("### 💰 P&L Backtest Simulation (AI Strategy vs. Buy & Hold)")
        st.caption("Strategy logic: Long (+return) if AI predicts UP (1), Short (-return) if AI predicts DOWN (0).")
        
        df_30d_sorted = df_30d.sort_values("date").copy()
        df_30d_sorted["ai_return"] = np.where(
            df_30d_sorted["predicted_direction"] == 1,
            df_30d_sorted["next_day_return"],
            -df_30d_sorted["next_day_return"]
        )
        df_30d_sorted["bh_return"] = df_30d_sorted["next_day_return"]
        
        daily_pnl = df_30d_sorted.groupby("date")[["ai_return", "bh_return"]].mean().reset_index()
        daily_pnl["AI Strategy P&L (%)"] = daily_pnl["ai_return"].cumsum() * 100
        daily_pnl["Buy & Hold P&L (%)"] = daily_pnl["bh_return"].cumsum() * 100
        
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=daily_pnl["date"], y=daily_pnl["AI Strategy P&L (%)"],
            mode="lines+markers", name="AI Strategy P&L",
            line=dict(color="#22c55e", width=3)
        ))
        fig_pnl.add_trace(go.Scatter(
            x=daily_pnl["date"], y=daily_pnl["Buy & Hold P&L (%)"],
            mode="lines+markers", name="Buy & Hold P&L",
            line=dict(color="#f9a825", width=2, dash="dash")
        ))
        fig_pnl.update_layout(
            title=dict(
                text="Cumulative P&L Comparison (Past 30 Days)",
                font=dict(color="#ffffff", family="Inter")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            xaxis=dict(
                title=dict(  # ✅ FIXED: titlefont → nested title dict
                    text="Date",
                    font=dict(color="#ffffff")
                ),
                showgrid=True,
                gridcolor="#2a3a4a",
                tickfont=dict(color="#94a3b8")
            ),
            yaxis=dict(
                title=dict(  # ✅ FIXED: titlefont → nested title dict
                    text="Cumulative Return (%)",
                    font=dict(color="#ffffff")
                ),
                showgrid=True,
                gridcolor="#2a3a4a",
                tickfont=dict(color="#94a3b8")
            ),
            height=400,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(
                orientation="h",
                y=1.15,
                x=0.01,
                font=dict(color="#ffffff", family="Inter"),
                bgcolor="rgba(30, 41, 59, 0.6)"
            )
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_pnl, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        col_hm, col_roll = st.columns(2)
        
        # Confusion Matrix Heatmap
        with col_hm:
            st.markdown("### 🔥 Confusion Matrix Heatmap")
            cm_30d = confusion_matrix(y_true_30d, y_pred_30d, labels=[0, 1])
            fig_cm = px.imshow(
                cm_30d,
                labels=dict(x="Predicted Direction", y="Actual Direction", color="Count"),
                x=["Predicted Down (0)", "Predicted Up (1)"],
                y=["Actual Down (0)", "Actual Up (1)"],
                text_auto=True,
                color_continuous_scale="Blues",
                title="Confusion Matrix (30-Day Count)"
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                font=dict(color="#ffffff", family="Inter"),
                height=380,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Rolling Accuracy Trend
        with col_roll:
            st.markdown("### 📈 5-Day Rolling Accuracy Trend")
            df_30d_sorted["is_correct"] = (df_30d_sorted["actual_direction"] == df_30d_sorted["predicted_direction"]).astype(float)
            daily_acc = df_30d_sorted.groupby("date")["is_correct"].mean().reset_index()
            daily_acc["5D_Rolling_Accuracy"] = daily_acc["is_correct"].rolling(window=5, min_periods=1).mean() * 100
            
            fig_roll = px.line(
                daily_acc,
                x="date",
                y="5D_Rolling_Accuracy",
                title="5-Day Rolling Accuracy (%)",
                labels={"date": "Date", "5D_Rolling_Accuracy": "Accuracy (%)"}
            )
            fig_roll.add_hline(y=50, line_dash="dash", line_color="#ef4444", annotation_text="50% Baseline (Random Guess)")
            fig_roll.update_traces(line_color="#2962ff", line_width=3)
            fig_roll.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                font=dict(color="#ffffff", family="Inter"),
                xaxis=dict(showgrid=True, gridcolor="#2a3a4a", tickfont=dict(color="#94a3b8")),
                yaxis=dict(showgrid=True, gridcolor="#2a3a4a", tickfont=dict(color="#94a3b8"), range=[0, 100]),
                height=380,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_roll, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Sector-Wise Accuracy Bar Chart
        if selected_stock == "All Stocks":
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🏭 Sector-Wise Accuracy Performance (30 Days)")
            st.caption("Categorized by industry sectors to highlight model domain efficacy.")
            
            df_30d_sector = df_30d.copy()
            if "Sector" not in df_30d_sector.columns:
                df_30d_sector["Sector"] = df_30d_sector["stock"].map(SECTOR_MAP).fillna("Other")
            
            sector_acc = df_30d_sector.groupby("Sector").apply(
                lambda g: (g["actual_direction"] == g["predicted_direction"]).mean() * 100
            ).reset_index(name="Accuracy")
            
            sector_acc = sector_acc.sort_values("Accuracy", ascending=True)
            
            def get_color(acc):
                if acc < 50.0:
                    return "#ef4444"
                elif acc <= 60.0:
                    return "#f59e0b"
                else:
                    return "#22c55e"
                    
            sector_acc["Color"] = sector_acc["Accuracy"].apply(get_color)
            
            fig_sec = px.bar(
                sector_acc,
                x="Accuracy",
                y="Sector",
                orientation="h",
                text=sector_acc["Accuracy"].apply(lambda x: f"{x:.1f}%"),
                title="Model Accuracy by Sector (%)"
            )
            fig_sec.update_traces(marker_color=sector_acc["Color"], textposition="outside")
            fig_sec.add_vline(x=50, line_dash="dash", line_color="#94a3b8", annotation_text="50% Baseline")
            fig_sec.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30, 41, 59, 0.4)",
                font=dict(color="#ffffff", family="Inter"),
                height=400,
                xaxis=dict(range=[0, 100], showgrid=True, gridcolor="#2a3a4a", tickfont=dict(color="#94a3b8")),
                yaxis=dict(tickfont=dict(color="#ffffff")),
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_sec, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ❌ Biggest Misses Section (Visible Table, NO expander for table!)
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">❌ Biggest Misses</div>', unsafe_allow_html=True)

        misses_30d = df_30d[df_30d["actual_direction"] != df_30d["predicted_direction"]].copy()
        if misses_30d.empty:
            st.markdown('<div class="success-message">🎉 No prediction misses in this period!</div>', unsafe_allow_html=True)
        else:
            misses_30d["abs_return"] = misses_30d["next_day_return"].abs()
            top3_misses_30d = misses_30d.sort_values("abs_return", ascending=False).head(5)

            miss_df_30d = top3_misses_30d[["date", "stock", "actual_direction", "predicted_direction", "next_day_return"]].copy()
            miss_df_30d["Actual"] = miss_df_30d["actual_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
            miss_df_30d["Predicted"] = miss_df_30d["predicted_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
            miss_df_30d["Return"] = miss_df_30d["next_day_return"].apply(lambda x: f"{x:+.2%}")

            st.dataframe(
                miss_df_30d[["date", "stock", "Actual", "Predicted", "Return"]].rename(
                    columns={"date": "Date", "stock": "Stock"}
                ),
                use_container_width=True,
                hide_index=True
            )

            miss_table_30d = fetch_headlines_for_misses(top3_misses_30d, news_df)
            with st.expander("📰 View News Context for Misses", expanded=False):
                for _, mrow in miss_table_30d.iterrows():
                    st.markdown(f"**{mrow['Date']} - {mrow['Stock']}**")
                    st.caption(f"📰 {mrow['Headlines']}")