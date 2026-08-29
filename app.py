"""
Streamlit Web Application for Sentiment-to-Price Correlation Analyzer.

Interactive dashboard showcasing:
- Stock price & news sentiment dual-axis time series trend
- Correlation heatmaps & per-stock statistical breakdowns
- XGBoost classification metrics vs Random & Same-As-Yesterday baselines
- Feature importances and live news feeds with FinBERT sentiment tags
"""

import os
import sys
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
from plotly.subplots import make_subplots
import streamlit as st

st.cache_data.clear()
st.cache_resource.clear()

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import (
    DATA_RAW_DIR, 
    DATA_PROCESSED_DIR, 
    OUTPUT_FIGURES_DIR, 
    OUTPUT_REPORTS_DIR, 
    NSE_STOCKS
)
from src.data_collector import DataCollector
from src.sentiment_analyzer import SentimentAnalyzer
from src.feature_engineering import FeatureEngineer
from src.correlation_analyzer import CorrelationAnalyzer
from src.prediction_model import SentimentPredictor

# Page Configuration
st.set_page_config(
    page_title="NSE Sentiment-to-Price Correlation Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Modern & Minimalist UI Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* App Background */
.stApp {
    background-color: #0f172a;
    color: #f8fafc;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

h1 { font-size: 1.8rem !important; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.35rem !important; margin-top: 1rem !important; }
h3 { font-size: 1.15rem !important; }

/* Subtitle */
.stCaption {
    color: #94a3b8 !important;
}

/* Metric Cards */
.metric-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    height: 100%;
}
.metric-card .card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    font-weight: 500;
}
.metric-card .card-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 4px;
}
.metric-card .card-subtext {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
}
.card-value-blue { color: #3b82f6 !important; }
.card-value-green { color: #22c55e !important; }
.card-value-red { color: #ef4444 !important; }
.card-value-gold { color: #eab308 !important; }

.metric-card.border-green { border-left: 4px solid #22c55e; }
.metric-card.border-red { border-left: 4px solid #ef4444; }
.metric-card.border-blue { border-left: 4px solid #3b82f6; }

/* Custom Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #1e293b;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #334155;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 16px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #3b82f6;
    color: #ffffff;
}

/* Dataframe & Table */
.stDataFrame {
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* News Cards */
.news-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: border-color 0.15s ease;
}
.news-card:hover {
    border-color: #475569;
}
.badge-pos { background: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
.badge-neg { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }
.badge-neu { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }

/* Hide Streamlit Header Padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_or_generate_pipeline_data():
    """Loads existing processed data or executes light pipeline collection."""
    processed_path = DATA_PROCESSED_DIR / "processed_dataset.csv"
    news_path = DATA_RAW_DIR / "raw_news.csv"
    
    if os.path.exists(processed_path) and os.path.exists(news_path):
        processed_df = pd.read_csv(processed_path)
        news_df = pd.read_csv(news_path)
    else:
        collector = DataCollector(days_history=60)
        prices = collector.fetch_stock_prices()
        news = collector.fetch_stock_news()
        
        analyzer = SentimentAnalyzer()
        news_df = analyzer.analyze_dataframe(news)
        
        fe = FeatureEngineer(prices, news_df)
        processed_df = fe.merge_and_build_features()
        
    return processed_df, news_df

# Load Data
with st.spinner("Loading market data & sentiment matrix..."):
    processed_df, news_df = load_or_generate_pipeline_data()

processed_df["Date"] = pd.to_datetime(processed_df["Date"]).dt.date
news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.date

# Header Section
st.title("📈 NSE Stock Sentiment & Return Predictor")
st.caption("AI-Powered Financial News Sentiment Analysis (FinBERT) vs. Next-Day Return Predictability")

# Sidebar Controls
st.sidebar.header("Dashboard Controls")
selected_stock = st.sidebar.selectbox("Select NSE Stock Ticker", NSE_STOCKS, index=0)

if st.sidebar.button("🔄 Refresh Data Pipeline", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Re-running pipeline...")

st.sidebar.divider()
st.sidebar.markdown("""
<div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.5;">
    <strong>Model Engine:</strong> XGBoost + FinBERT<br/>
    <strong>Target Horizon:</strong> T+1 Next-Day Direction<br/>
    <strong>Universe:</strong> NSE Nifty Top Stocks<br/>
    <strong>Research Features:</strong> <a href="/Advanced_Analytics" target="_self" style="color:#3b82f6; font-weight:600;">🔬 5_Advanced_Analytics</a>
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Sentiment Trend", 
    "🔥 Correlation Analysis", 
    "🤖 Predictive Model", 
    "📰 Live News & Signals"
])

# ---------------------------------------------------------
# TAB 1: Price & Sentiment Trend Overlay
# ---------------------------------------------------------
with tab1:
    st.subheader(f"Price vs. Sentiment Trend — {selected_stock}")
    stock_df = processed_df[processed_df["Symbol"] == selected_stock].sort_values("Date")
    
    if not stock_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        latest = stock_df.iloc[-1]
        eff_sentiment = latest.get('effective_sentiment', latest['avg_sentiment'])
        news_cnt = int(latest['news_count'])
        recent_news_cnt = int(stock_df.tail(5)['news_count'].sum())
        
        sent_color_class = "card-value-green" if eff_sentiment >= 0 else "card-value-red"
        
        # Check target_up prediction
        if pd.isna(latest.get('target_up')) or np.isnan(latest.get('target_up', np.nan)):
            sp = SentimentPredictor(processed_df)
            X_train, X_test, y_train, y_test, _ = sp.time_based_split()
            sp.train_xgboost(X_train, y_train)
            
            latest_features = pd.DataFrame([latest[SentimentPredictor.FEATURE_COLS]])
            prob_up = float(sp.model.predict_proba(latest_features)[0][1])
            pred_class = 1 if prob_up >= 0.5 else 0
            conf = max(prob_up, 1.0 - prob_up)
            
            target_display = "UP 🟢" if pred_class == 1 else "DOWN 🔴"
            target_color_class = "card-value-green" if pred_class == 1 else "card-value-red"
            dir_border_class = "border-green" if pred_class == 1 else "border-red"
            subtext = f"AI Tomorrow • Conf: {conf:.1%}"
        elif latest['target_up'] == 1:
            target_display = "UP 🟢"
            target_color_class = "card-value-green"
            dir_border_class = "border-green"
            subtext = "Target (Next-Day Direction)"
        else:
            target_display = "DOWN 🔴"
            target_color_class = "card-value-red"
            dir_border_class = "border-red"
            subtext = "Target (Next-Day Direction)"

        with col1:
            st.markdown(f"""
            <div class="metric-card border-blue">
                <div class="card-title">Latest Close Price</div>
                <div class="card-value card-value-blue">₹{latest['Close']:.2f}</div>
                <div class="card-subtext">NSE: {selected_stock}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Mean Sentiment</div>
                <div class="card-value {sent_color_class}">{eff_sentiment:+.3f}</div>
                <div class="card-subtext">FinBERT Scale (-1 to +1)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">News Count</div>
                <div class="card-value card-value-gold">{news_cnt}</div>
                <div class="card-subtext">5D Volume: {recent_news_cnt} articles</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card {dir_border_class}">
                <div class="card-title">Direction Signal</div>
                <div class="card-value {target_color_class}">{target_display}</div>
                <div class="card-subtext">{subtext}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Interactive Plotly Dual-Axis Chart in dark theme
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add background shading for positive / negative sentiment days
        for _, row in stock_df.iterrows():
            v_color = "rgba(34, 197, 94, 0.06)" if row["avg_sentiment"] >= 0 else "rgba(239, 68, 68, 0.06)"
            fig.add_vrect(
                x0=row["Date"], x1=row["Date"],
                fillcolor=v_color, opacity=0.5,
                layer="below", line_width=0
            )

        # Blue line for FinBERT Sentiment Score (left axis)
        fig.add_trace(
            gg.Scatter(
                x=stock_df["Date"], 
                y=stock_df["avg_sentiment"], 
                name="FinBERT Sentiment", 
                line=dict(color="#3b82f6", width=2),
                mode="lines+markers",
                marker=dict(
                    color=stock_df["avg_sentiment"].apply(lambda x: "#22c55e" if x >= 0 else "#ef4444"),
                    size=7
                ),
                hovertemplate="Date: %{x}<br>Sentiment: %{y:+.3f}"
            ),
            secondary_y=False
        )

        # Orange line for Stock Close Price (right axis)
        fig.add_trace(
            gg.Scatter(
                x=stock_df["Date"], 
                y=stock_df["Close"], 
                name="Close Price (₹)", 
                line=dict(color="#f97316", width=2.5),
                hovertemplate="Date: %{x}<br>Close: ₹%{y:.2f}"
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=dict(
                text=f"Daily Close Price & FinBERT Sentiment ({selected_stock})",
                font=dict(color="#f8fafc", size=15, family="Inter")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            xaxis=dict(
                title=dict(text="Date", font=dict(color="#94a3b8")),
                showgrid=True,
                gridcolor="#334155",
                tickfont=dict(color="#94a3b8")
            ),
            height=420,
            hovermode="x unified",
            margin=dict(l=15, r=15, t=50, b=15),
            legend=dict(
                orientation="h", 
                y=1.08, 
                x=0.01,
                font=dict(color="#f8fafc", family="Inter"),
                bgcolor="rgba(30, 41, 59, 0.6)"
            )
        )
        fig.update_yaxes(
            title_text="Sentiment Score (-1 to +1)", 
            secondary_y=False,
            showgrid=True,
            gridcolor="#334155",
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8")
        )
        fig.update_yaxes(
            title_text="Stock Close Price (₹)", 
            secondary_y=True,
            showgrid=False,
            tickfont=dict(color="#94a3b8"),
            title_font=dict(color="#94a3b8")
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No processed data available for {selected_stock}")

# ---------------------------------------------------------
# TAB 2: Correlation Analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("Statistical Correlation Matrix & Universe Breakdown")
    
    ca = CorrelationAnalyzer(processed_df)
    overall_stats, stock_corrs = ca.compute_correlations()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card border-blue">
            <div class="card-title">Pearson Correlation (r)</div>
            <div class="card-value card-value-blue">{overall_stats['r']:.4f}</div>
            <div class="card-subtext">Linear Dependency Metric</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">p-Value</div>
            <div class="card-value card-value-gold">{overall_stats['p_value']:.4f}</div>
            <div class="card-subtext">Statistical Significance Test</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Total Trading Samples</div>
            <div class="card-value card-value-green">{overall_stats['n']}</div>
            <div class="card-subtext">Aggregated Universe Records</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### Sentiment vs. Next-Day Returns")
        fig_scatter = px.scatter(
            processed_df, 
            x="avg_sentiment", 
            y="next_day_return", 
            color="Symbol",
            hover_data=["Date", "Symbol", "news_count"],
            title="Sentiment Score vs Next-Day Return Across Universe"
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            xaxis=dict(showgrid=True, gridcolor="#334155", tickfont=dict(color="#94a3b8")),
            yaxis=dict(showgrid=True, gridcolor="#334155", tickfont=dict(color="#94a3b8")),
            height=400,
            margin=dict(l=15, r=15, t=40, b=15)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_right:
        st.markdown("### Per-Stock Correlation Table")
        st.dataframe(stock_corrs.style.background_gradient(cmap="Blues", subset=["Correlation"]), use_container_width=True, height=400)

# ---------------------------------------------------------
# TAB 3: Predictive Machine Learning Model
# ---------------------------------------------------------
with tab3:
    st.subheader("XGBoost Directional Predictor vs. Baselines")
    
    sp = SentimentPredictor(processed_df)
    results_df, _ = sp.evaluate()
    
    st.markdown("### Model Evaluation Summary (Time-Based Train/Test Split)")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=["Accuracy", "F1 Score"], color="rgba(34, 197, 94, 0.25)"), use_container_width=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("### Feature Importance Plot")
        feat_img = OUTPUT_FIGURES_DIR / "feature_importance.png"
        if os.path.exists(feat_img):
            st.image(str(feat_img), use_container_width=True)
        else:
            st.info("Feature importance plot loading...")
            
    with col_img2:
        st.markdown("### Confusion Matrix")
        cm_img = OUTPUT_FIGURES_DIR / "confusion_matrix.png"
        if os.path.exists(cm_img):
            st.image(str(cm_img), use_container_width=True)
        else:
            st.info("Confusion matrix loading...")

# ---------------------------------------------------------
# TAB 4: Live News & Sentiment Feed
# ---------------------------------------------------------
with tab4:
    st.subheader(f"Latest Financial News & FinBERT Signals — {selected_stock}")
    
    stock_news = news_df[news_df["Symbol"] == selected_stock].sort_values("Date", ascending=False)
    
    if not stock_news.empty:
        for _, row in stock_news.head(15).iterrows():
            lbl = str(row.get("sentiment_label", "neutral")).lower()
            if "pos" in lbl:
                badge = "<span class='badge-pos'>POSITIVE</span>"
            elif "neg" in lbl:
                badge = "<span class='badge-neg'>NEGATIVE</span>"
            else:
                badge = "<span class='badge-neu'>NEUTRAL</span>"
                
            conf = row.get("confidence", 0.8)
            src = row.get("Source", "Market News")
            date_str = str(row.get("Date", ""))
            headline = row.get("Headline", "")
            url = row.get("URL", "#")

            st.markdown(f"""
            <div class="news-card">
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#3b82f6; font-weight:600; font-size:0.85rem;'>📅 {date_str} | {src}</span>
                    <span>{badge} <span style='color:#64748b; font-size:0.8rem; margin-left:6px;'>(Conf: {conf:.1%})</span></span>
                </div>
                <h4 style='margin:8px 0 0 0; font-size:1.05rem;'><a href='{url}' target='_blank' style='color:#f8fafc; text-decoration:none;'>{headline}</a></h4>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(f"No recent news articles recorded for {selected_stock}.")

st.divider()
st.caption("NSE Sentiment-Correlation Analyzer • Built with Streamlit, Plotly, FinBERT & XGBoost")

