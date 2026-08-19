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

# Custom Styling (Dark-themed glassmorphism elements)
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2a2e39;
        text-align: center;
    }
    .badge-pos { background-color: #26a69a; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-neg { background-color: #ef5350; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-neu { background-color: #787b86; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
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

# Main App Layout
st.markdown("<div class='main-header'>📈 NSE Stock Sentiment & Price Movement Analyzer</div>", unsafe_allow_html=True)
st.caption("AI-Powered Financial News Sentiment Analysis (FinBERT) vs. Next-Day Return Predictability")

# Sidebar Controls
st.sidebar.header("⚙️ Dashboard Controls")
selected_stock = st.sidebar.selectbox("Select NSE Stock Ticker", NSE_STOCKS, index=0)

run_pipeline_btn = st.sidebar.button("🔄 Refresh Data Pipeline")
if run_pipeline_btn:
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Re-running pipeline...")

# Load Data
with st.spinner("Loading market data & sentiment inference matrix..."):
    processed_df, news_df = load_or_generate_pipeline_data()

processed_df["Date"] = pd.to_datetime(processed_df["Date"]).dt.date
news_df["Date"] = pd.to_datetime(news_df["Date"]).dt.date

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
    st.subheader(f"Price vs. Sentiment Trend Analysis — {selected_stock}")
    stock_df = processed_df[processed_df["Symbol"] == selected_stock].sort_values("Date")
    
    if not stock_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        latest = stock_df.iloc[-1]
        eff_sentiment = latest.get('effective_sentiment', latest['avg_sentiment'])
        news_cnt = int(latest['news_count'])
        recent_news_cnt = int(stock_df.tail(5)['news_count'].sum())
        
        col1.metric("Latest Close", f"₹{latest['Close']:.2f}")
        col2.metric("Mean Sentiment", f"{eff_sentiment:+.3f}")
        col3.metric("News Count", f"{news_cnt}", f"5D Volume: {recent_news_cnt}")
        
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
            
            target_display = "PREDICTED UP 🟢" if pred_class == 1 else "PREDICTED DOWN 🔴"
            col4.metric("AI Prediction (Tomorrow)", target_display, f"Conf: {conf:.1%}")
        elif latest['target_up'] == 1:
            col4.metric("Target (Next-Day Direction)", "UP 🟢")
        else:
            col4.metric("Target (Next-Day Direction)", "DOWN 🔴")

        # Interactive Plotly Dual-Axis Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            gg.Scatter(x=stock_df["Date"], y=stock_df["Close"], name="Close Price (INR)", line=dict(color="#2962FF", width=2.5)),
            secondary_y=False
        )
        
        fig.add_trace(
            gg.Bar(x=stock_df["Date"], y=stock_df["avg_sentiment"], name="Sentiment Score", opacity=0.4,
                   marker=dict(color=stock_df["avg_sentiment"].apply(lambda x: "#26a69a" if x >= 0 else "#ef5350"))),
            secondary_y=True
        )

        fig.update_layout(
            title=f"Daily Close Price & FinBERT Sentiment Score ({selected_stock})",
            xaxis_title="Date",
            height=500,
            hovermode="x unified",
            template="plotly_dark",
            legend=dict(orientation="h", y=1.1)
        )
        fig.update_yaxes(title_text="Stock Close Price (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score (-1 to +1)", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No processed data available for {selected_stock}")

# ---------------------------------------------------------
# TAB 2: Correlation Analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("Statistical Correlation Matrix & Per-Stock Breakdown")
    
    ca = CorrelationAnalyzer(processed_df)
    overall_stats, stock_corrs = ca.compute_correlations()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Pearson Correlation (r)", f"{overall_stats['r']:.4f}")
    c2.metric("p-Value", f"{overall_stats['p_value']:.4f}")
    c3.metric("Total Trading Samples", overall_stats['n'])

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.markdown("##### Sentiment Score vs. Next-Day Returns Scatter Plot")
        fig_scatter = px.scatter(
            processed_df, 
            x="avg_sentiment", 
            y="next_day_return", 
            color="Symbol",
            hover_data=["Date", "Symbol", "news_count"],
            trendline="ols",
            title="Sentiment Score vs Next-Day Return Across Universe",
            template="plotly_dark"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col_right:
        st.markdown("##### Per-Stock Correlation Table")
        st.dataframe(stock_corrs.style.background_gradient(cmap="Blues", subset=["Correlation"]), height=400)

# ---------------------------------------------------------
# TAB 3: Predictive Machine Learning Model
# ---------------------------------------------------------
with tab3:
    st.subheader("XGBoost Directional Predictor vs. Baselines")
    
    sp = SentimentPredictor(processed_df)
    results_df, _ = sp.evaluate()
    
    st.markdown("##### Model Evaluation Summary (Time-Based Train/Test Split)")
    st.dataframe(results_df.style.highlight_max(axis=0, subset=["Accuracy", "F1 Score"], color="#1b5e20"), use_container_width=True)

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("##### Feature Importance Plot")
        feat_img = OUTPUT_FIGURES_DIR / "feature_importance.png"
        if os.path.exists(feat_img):
            st.image(str(feat_img), use_container_width=True)
        else:
            st.info("Feature importance plot loading...")
            
    with col_img2:
        st.markdown("##### Confusion Matrix")
        cm_img = OUTPUT_FIGURES_DIR / "confusion_matrix.png"
        if os.path.exists(cm_img):
            st.image(str(cm_img), use_container_width=True)
        else:
            st.info("Confusion matrix loading...")

# ---------------------------------------------------------
# TAB 4: Live News & Sentiment Feed
# ---------------------------------------------------------
with tab4:
    st.subheader(f"Latest Financial News & FinBERT Inferences — {selected_stock}")
    
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
            <div style='background-color:#1e222d; padding:12px; border-radius:6px; margin-bottom:8px;'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span style='color:#90caf9; font-weight:bold;'>{date_str} | {src}</span>
                    <span>{badge} (Conf: {conf:.2%})</span>
                </div>
                <h4 style='margin:6px 0;'><a href='{url}' target='_blank' style='color:#e0e0e0; text-decoration:none;'>{headline}</a></h4>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info(f"No recent news articles recorded for {selected_stock}.")
