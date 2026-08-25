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

# Premium Financial Dashboard Custom CSS Styling
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
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    margin-top: 4px;
}
.metric-card .card-subtext {
    font-size: 0.85rem;
    color: #94a3b8;
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

/* FinBERT News Badges */
.badge-pos { background-color: rgba(34, 197, 94, 0.2); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.4); padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.75rem; }
.badge-neg { background-color: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.4); padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.75rem; }
.badge-neu { background-color: rgba(148, 163, 184, 0.2); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.4); padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 0.75rem; }

.news-card {
    background: rgba(30, 41, 59, 0.6);
    border-radius: 12px;
    padding: 16px 20px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-bottom: 12px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.news-card:hover {
    border-color: rgba(41, 98, 255, 0.4);
    transform: translateY(-1px);
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
with st.spinner("Loading market data & sentiment inference matrix..."):
    processed_df, news_df = load_or_generate_pipeline_data()

processed_df["Date"] = pd.to_datetime(processed_df["Date"]).dt.date
news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.date

last_updated_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

# Header Section
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 16px; border-bottom: 2px solid rgba(41, 98, 255, 0.3); margin-bottom: 24px;">
    <div>
        <h1 style="margin: 0; padding: 0;">📈 NSE Stock Sentiment & Price Movement Analyzer</h1>
        <p style="color: #94a3b8; margin: 6px 0 0 0; font-size: 0.95rem;">AI-Powered Financial News Sentiment Analysis (FinBERT) vs. Next-Day Return Predictability</p>
    </div>
    <div style="text-align: right; color: #94a3b8; font-size: 0.85rem; background: rgba(30, 41, 59, 0.5); padding: 8px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05);">
        <span>Pipeline: <strong style="color: #22c55e;">● Active & Ready</strong></span><br/>
        <span>Last Updated: <strong style="color: #ffffff;">{last_updated_str}</strong></span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Dashboard Controls")
selected_stock = st.sidebar.selectbox("Select NSE Stock Ticker", NSE_STOCKS, index=0)

run_pipeline_btn = st.sidebar.button("🔄 Refresh Data Pipeline", use_container_width=True)
if run_pipeline_btn:
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Re-running pipeline...")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="color: #94a3b8; font-size: 0.8rem;">
    <strong>Model Engine:</strong> XGBoost + FinBERT<br/>
    <strong>Target Horizon:</strong> T+1 Next-Day Direction<br/>
    <strong>Universe:</strong> NSE Nifty Top Stocks
</div>
""", unsafe_allow_html=True)

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price & Sentiment Trend", 
    "🔥 Correlation Analysis", 
    "🤖 Predictive ML Model", 
    "📰 Live News & FinBERT Signals"
])

# ---------------------------------------------------------
# TAB 1: Price & Sentiment Trend Overlay
# ---------------------------------------------------------
with tab1:
    st.markdown(f'<div class="section-header">Price vs. Sentiment Trend Analysis — {selected_stock}</div>', unsafe_allow_html=True)
    stock_df = processed_df[processed_df["Symbol"] == selected_stock].sort_values("Date")
    
    if not stock_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        latest = stock_df.iloc[-1]
        eff_sentiment = latest.get('effective_sentiment', latest['avg_sentiment'])
        news_cnt = int(latest['news_count'])
        recent_news_cnt = int(stock_df.tail(5)['news_count'].sum())
        
        sent_color_class = "card-value-green" if eff_sentiment >= 0 else "card-value-red"
        
        # Check if actual target_up is known or pending
        if pd.isna(latest.get('target_up')) or np.isnan(latest.get('target_up', np.nan)):
            # Predict direction for tomorrow using trained XGBoost model
            sp = SentimentPredictor(processed_df)
            X_train, X_test, y_train, y_test, _ = sp.time_based_split()
            sp.train_xgboost(X_train, y_train)
            
            latest_features = pd.DataFrame([latest[SentimentPredictor.FEATURE_COLS]])
            prob_up = float(sp.model.predict_proba(latest_features)[0][1])
            pred_class = 1 if prob_up >= 0.5 else 0
            conf = max(prob_up, 1.0 - prob_up)
            
            target_display = "UP 🟢" if pred_class == 1 else "DOWN 🔴"
            target_color_class = "card-value-green" if pred_class == 1 else "card-value-red"
            subtext = f"AI Prediction (Tomorrow) • Conf: {conf:.1%}"
        elif latest['target_up'] == 1:
            target_display = "UP 🟢"
            target_color_class = "card-value-green"
            subtext = "Target (Next-Day Direction)"
        else:
            target_display = "DOWN 🔴"
            target_color_class = "card-value-red"
            subtext = "Target (Next-Day Direction)"

        with col1:
            st.markdown(f"""
            <div class="metric-card">
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
            <div class="metric-card">
                <div class="card-title">Direction Signal</div>
                <div class="card-value {target_color_class}">{target_display}</div>
                <div class="card-subtext">{subtext}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # Interactive Plotly Dual-Axis Chart in dark theme
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            gg.Scatter(
                x=stock_df["Date"], 
                y=stock_df["Close"], 
                name="Close Price (INR)", 
                line=dict(color="#2962ff", width=3),
                hovertemplate="Date: %{x}<br>Close: ₹%{y:.2f}"
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            gg.Bar(
                x=stock_df["Date"], 
                y=stock_df["avg_sentiment"], 
                name="Sentiment Score", 
                opacity=0.5,
                marker=dict(color=stock_df["avg_sentiment"].apply(lambda x: "#22c55e" if x >= 0 else "#ef4444")),
                hovertemplate="Date: %{x}<br>Sentiment: %{y:+.3f}"
            ),
            secondary_y=True
        )

        fig.update_layout(
            title=dict(
                text=f"Daily Close Price & FinBERT Sentiment Score ({selected_stock})",
                font=dict(color="#ffffff", size=16, family="Inter")
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            xaxis=dict(
                title=dict(  # ✅ FIXED: titlefont → title=dict(font=...)
                    text="Date",
                    font=dict(color="#ffffff")
                ),
                showgrid=True,
                gridcolor="#2a3a4a",
                tickfont=dict(color="#94a3b8")
            ),
            yaxis=dict(  # ✅ Added yaxis with proper title structure
                title=dict(
                    text="Price / Sentiment",
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
                y=1.12, 
                x=0.01,
                font=dict(color="#ffffff", family="Inter"),
                bgcolor="rgba(30, 41, 59, 0.6)"
            )
        )
        fig.update_yaxes(
            title_text="Stock Close Price (₹)", 
            secondary_y=False,
            showgrid=True,
            gridcolor="#2a3a4a",
            tickfont=dict(color="#94a3b8"),
            titlefont=dict(color="#ffffff")
        )
        fig.update_yaxes(
            title_text="Sentiment Score (-1 to +1)", 
            secondary_y=True,
            showgrid=False,
            tickfont=dict(color="#94a3b8"),
            titlefont=dict(color="#ffffff")
        )

        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning(f"No processed data available for {selected_stock}")

# ---------------------------------------------------------
# TAB 2: Correlation Analysis
# ---------------------------------------------------------
with tab2:
    st.markdown('<div class="section-header">Statistical Correlation Matrix & Per-Stock Breakdown</div>', unsafe_allow_html=True)
    
    ca = CorrelationAnalyzer(processed_df)
    overall_stats, stock_corrs = ca.compute_correlations()
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Overall Pearson Correlation (r)</div>
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

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("### Sentiment Score vs. Next-Day Returns Scatter Plot")
        fig_scatter = px.scatter(
            processed_df, 
            x="avg_sentiment", 
            y="next_day_return", 
            color="Symbol",
            hover_data=["Date", "Symbol", "news_count"],
            trendline="ols",
            title="Sentiment Score vs Next-Day Return Across Universe"
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            font=dict(color="#ffffff", family="Inter"),
            xaxis=dict(showgrid=True, gridcolor="#2a3a4a", tickfont=dict(color="#94a3b8")),
            yaxis=dict(showgrid=True, gridcolor="#2a3a4a", tickfont=dict(color="#94a3b8")),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_right:
        st.markdown("### Per-Stock Correlation Table")
        st.dataframe(stock_corrs.style.background_gradient(cmap="Blues", subset=["Correlation"]), use_container_width=True, height=400)

# ---------------------------------------------------------
# TAB 3: Predictive Machine Learning Model
# ---------------------------------------------------------
with tab3:
    st.markdown('<div class="section-header">XGBoost Directional Predictor vs. Baselines</div>', unsafe_allow_html=True)
    
    sp = SentimentPredictor(processed_df)
    results_df, _ = sp.evaluate()
    
    st.markdown("### Model Evaluation Summary (Time-Based Train/Test Split)")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=["Accuracy", "F1 Score"], color="rgba(34, 197, 94, 0.3)"), use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("### Feature Importance Plot")
        feat_img = OUTPUT_FIGURES_DIR / "feature_importance.png"
        if os.path.exists(feat_img):
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.image(str(feat_img), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Feature importance plot loading...")
            
    with col_img2:
        st.markdown("### Confusion Matrix")
        cm_img = OUTPUT_FIGURES_DIR / "confusion_matrix.png"
        if os.path.exists(cm_img):
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.image(str(cm_img), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Confusion matrix loading...")

# ---------------------------------------------------------
# TAB 4: Live News & Sentiment Feed
# ---------------------------------------------------------
with tab4:
    st.markdown(f'<div class="section-header">Latest Financial News & FinBERT Inferences — {selected_stock}</div>', unsafe_allow_html=True)
    
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
                    <span style='color:#2962ff; font-weight:600; font-size:0.85rem;'>📅 {date_str} | {src}</span>
                    <span>{badge} <span style='color:#94a3b8; font-size:0.8rem; margin-left:6px;'>(Conf: {conf:.1%})</span></span>
                </div>
                <h4 style='margin:10px 0 0 0; font-size:1.05rem;'><a href='{url}' target='_blank' style='color:#ffffff; text-decoration:none;'>{headline}</a></h4>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(f"No recent news articles recorded for {selected_stock}.")

# Clean Dashboard Footer
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 12px 0;">
    NSE Sentiment-Correlation Analyzer • Engineered with Streamlit, Plotly, FinBERT & XGBoost
</div>
""", unsafe_allow_html=True)

