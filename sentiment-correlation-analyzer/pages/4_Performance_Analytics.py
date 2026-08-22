"""
Streamlit Page: Model Performance Monitoring & Analytics (Page 4)

Comprehensive model tracking dashboard featuring:
- Core Accuracy Tabs (Yesterday, Past 7 Days, Past 30 Days)
- P&L Backtest Simulation (AI Strategy vs Buy & Hold)
- Confusion Matrix Heatmap
- 5-Day Rolling Accuracy Trend
- Top 3 "Big Miss" Forensic Headline Analysis
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

# Custom Glassmorphic Styling
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
        padding: 16px;
        border: 1px solid #2a2e39;
        text-align: center;
        margin-bottom: 10px;
    }
    .card-title {
        font-size: 0.85rem;
        color: #787b86;
        font-weight: 600;
        margin-bottom: 6px;
        text-transform: uppercase;
    }
    .card-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .card-value-green { color: #26a69a; }
    .card-value-red { color: #ef5350; }
    .card-value-blue { color: #2962ff; }
</style>
""", unsafe_allow_html=True)

# Main Title & Subheader
st.markdown("<div class='main-header'>🎯 Model Performance Monitoring & Analytics</div>", unsafe_allow_html=True)
st.caption("Empirical Validation of XGBoost Sentiment Model: Real-Time Accuracy, P&L Backtest, Confusion Matrix & Sector Drift")

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
    st.subheader(f"Yesterday's Performance Overview ({max_date})")
    
    yesterday_df = df_filtered[df_filtered["date"] == max_date]
    
    if yesterday_df.empty:
        st.warning("⚠️ Insufficient data available for Yesterday.")
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
            <div class='metric-card'>
                <div class='card-title'>Actual Movement</div>
                <div class='card-value {act_color}'>{act_text} ({ret_pct:+.2%})</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            pred_text = "UP 🟢" if pred_up else "DOWN 🔴"
            pred_color = "card-value-green" if pred_up else "card-value-red"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='card-title'>Model Prediction</div>
                <div class='card-value {pred_color}'>{pred_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            corr_text = "YES 🟢" if was_correct else "NO 🔴"
            corr_color = "card-value-green" if was_correct else "card-value-red"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='card-title'>Was it Right?</div>
                <div class='card-value {corr_color}'>{corr_text}</div>
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
            <div class='metric-card'>
                <div class='card-title'>Universe Yesterday Accuracy</div>
                <div class='card-value card-value-blue'>{acc_pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='card-title'>Correct Predictions</div>
                <div class='card-value card-value-green'>{correct_preds} / {total_stocks} Stocks</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='card-title'>Actual Up vs Pred Up</div>
                <div class='card-value card-value-blue'>{actual_ups} Up / {pred_ups} Pred Up</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("##### Yesterday's Prediction Breakdown Across Universe")
        y_summary = yesterday_df[["stock", "actual_direction", "predicted_direction", "next_day_return"]].copy()
        y_summary["Actual Movement"] = y_summary["actual_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
        y_summary["Model Prediction"] = y_summary["predicted_direction"].apply(lambda x: "UP 🟢" if x == 1 else "DOWN 🔴")
        y_summary["Correct?"] = (y_summary["actual_direction"] == y_summary["predicted_direction"]).apply(lambda x: "Yes 🟢" if x else "No 🔴")
        y_summary["Next-Day Return"] = y_summary["next_day_return"].apply(lambda x: f"{x:+.2%}")
        
        st.dataframe(
            y_summary[["stock", "Actual Movement", "Model Prediction", "Correct?", "Next-Day Return"]].rename(columns={"stock": "Stock Ticker"}),
            use_container_width=True
        )

# ---------------------------------------------------------
# TAB 2: Past Week (7 Days)
# ---------------------------------------------------------
with tab2:
    st.subheader(f"Past Week Performance (Last 7 Days ending {max_date})")
    
    min_date_7d = max_date - pd.Timedelta(days=7)
    df_7d = df_filtered[df_filtered["date"] >= min_date_7d].copy()
    
    if len(df_7d) < 1:
        st.warning("⚠️ Insufficient data for Past Week (7 Days).")
    else:
        y_true_7d = df_7d["actual_direction"].astype(int)
        y_pred_7d = df_7d["predicted_direction"].astype(int)
        
        acc_7d = accuracy_score(y_true_7d, y_pred_7d) * 100
        
        # Weekly Accuracy Metric
        st.metric("📆 Weekly Accuracy (7 Days)", f"{acc_7d:.1f}%")
        
        col_cm, col_info = st.columns([1, 1])
        with col_cm:
            st.markdown("##### Confusion Matrix (Counts)")
            cm_7d = confusion_matrix(y_true_7d, y_pred_7d, labels=[0, 1])
            cm_df_7d = pd.DataFrame(
                cm_7d,
                index=["Actual Down (0)", "Actual Up (1)"],
                columns=["Predicted Down (0)", "Predicted Up (1)"]
            )
            st.dataframe(cm_df_7d.style.highlight_max(axis=None, color="#1b5e20"), use_container_width=True)
            
        with col_info:
            st.markdown("##### 7-Day Sample Summary")
            total_samples = len(df_7d)
            correct_samples = int((y_true_7d == y_pred_7d).sum())
            st.write(f"• **Total Evaluated Predictions**: `{total_samples}`")
            st.write(f"• **Correct Direction Predictions**: `{correct_samples}`")
            st.write(f"• **Incorrect Direction Predictions**: `{total_samples - correct_samples}`")

        # Feature 5: Top 3 "Big Miss" Analysis for Past Week
        st.markdown("---")
        with st.expander("❌ Biggest Misses (Past 7 Days)", expanded=False):
            misses_7d = df_7d[df_7d["actual_direction"] != df_7d["predicted_direction"]].copy()
            if misses_7d.empty:
                st.success("🎉 No prediction misses in the past week!")
            else:
                misses_7d["abs_return"] = misses_7d["next_day_return"].abs()
                top3_misses_7d = misses_7d.sort_values("abs_return", ascending=False).head(3)
                
                miss_table_7d = fetch_headlines_for_misses(top3_misses_7d, news_df)
                for _, mrow in miss_table_7d.iterrows():
                    st.markdown(f"""
                    <div style='background-color:#1e222d; padding:12px; border-radius:6px; margin-bottom:8px; border-left: 4px solid #ef5350;'>
                        <strong>Date:</strong> {mrow['Date']} | <strong>Stock:</strong> {mrow['Stock']} | 
                        <strong>Actual:</strong> {mrow['Actual Direction']} | <strong>Predicted:</strong> {mrow['Predicted Direction']} | 
                        <strong>Return:</strong> <span style='color:#ef5350;'>{mrow['Actual Return']}</span><br/>
                        <div style='margin-top:6px; color:#b0bec5; font-size:0.9rem;'>
                            <strong>Forensic News Context:</strong><br/>{mrow['Headlines']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: Past Month (30 Days)
# ---------------------------------------------------------
with tab3:
    st.subheader(f"Past Month Deep-Dive Analytics (Last 30 Days ending {max_date})")
    
    min_date_30d = max_date - pd.Timedelta(days=30)
    df_30d = df_filtered[df_filtered["date"] >= min_date_30d].copy()
    
    if len(df_30d) < 1:
        st.warning("⚠️ Insufficient data for Past Month (30 Days).")
    else:
        y_true_30d = df_30d["actual_direction"].astype(int)
        y_pred_30d = df_30d["predicted_direction"].astype(int)
        
        acc_30d = accuracy_score(y_true_30d, y_pred_30d) * 100
        prec_30d = precision_score(y_true_30d, y_pred_30d, zero_division=0) * 100
        rec_30d = recall_score(y_true_30d, y_pred_30d, zero_division=0) * 100
        f1_30d = f1_score(y_true_30d, y_pred_30d, zero_division=0) * 100
        
        # Monthly Accuracy & 3 Metric Cards
        st.metric("📅 Monthly Accuracy (30 Days)", f"{acc_30d:.1f}%")
        
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='card-title'>Precision (Up Predictions)</div>
                <div class='card-value card-value-blue'>{prec_30d:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m_c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='card-title'>Recall (Captured Ups)</div>
                <div class='card-value card-value-green'>{rec_30d:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m_c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='card-title'>F1-Score</div>
                <div class='card-value card-value-blue'>{f1_30d:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Feature 2: P&L Simulation (Killer Extra #1)
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
            line=dict(color="#26a69a", width=3)
        ))
        fig_pnl.add_trace(go.Scatter(
            x=daily_pnl["date"], y=daily_pnl["Buy & Hold P&L (%)"],
            mode="lines+markers", name="Buy & Hold P&L",
            line=dict(color="#ff9800", width=2, dash="dash")
        ))
        fig_pnl.update_layout(
            title="Cumulative P&L Comparison (Past 30 Days)",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template="plotly_dark",
            height=450,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1, x=0.3)
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        st.markdown("---")
        
        col_hm, col_roll = st.columns(2)
        
        # Feature 3: Confusion Matrix Heatmap (Killer Extra #2)
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
            fig_cm.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_cm, use_container_width=True)

        # Feature 4: Rolling Accuracy Trend (Killer Extra #3)
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
            fig_roll.add_hline(y=50, line_dash="dash", line_color="#ef5350", annotation_text="50% Baseline (Random Guess)")
            fig_roll.update_traces(line_color="#2962ff", line_width=2.5)
            fig_roll.update_layout(template="plotly_dark", height=380, yaxis_range=[0, 100])
            st.plotly_chart(fig_roll, use_container_width=True)

        # Feature 5: Top 3 "Big Miss" Analysis for Past Month
        st.markdown("---")
        with st.expander("❌ Biggest Misses (Past 30 Days)", expanded=False):
            misses_30d = df_30d[df_30d["actual_direction"] != df_30d["predicted_direction"]].copy()
            if misses_30d.empty:
                st.success("🎉 No prediction misses in the past month!")
            else:
                misses_30d["abs_return"] = misses_30d["next_day_return"].abs()
                top3_misses_30d = misses_30d.sort_values("abs_return", ascending=False).head(3)
                
                miss_table_30d = fetch_headlines_for_misses(top3_misses_30d, news_df)
                for _, mrow in miss_table_30d.iterrows():
                    st.markdown(f"""
                    <div style='background-color:#1e222d; padding:12px; border-radius:6px; margin-bottom:8px; border-left: 4px solid #ef5350;'>
                        <strong>Date:</strong> {mrow['Date']} | <strong>Stock:</strong> {mrow['Stock']} | 
                        <strong>Actual:</strong> {mrow['Actual Direction']} | <strong>Predicted:</strong> {mrow['Predicted Direction']} | 
                        <strong>Return:</strong> <span style='color:#ef5350;'>{mrow['Actual Return']}</span><br/>
                        <div style='margin-top:6px; color:#b0bec5; font-size:0.9rem;'>
                            <strong>Forensic News Context:</strong><br/>{mrow['Headlines']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Feature 6: Sector-Wise Accuracy Bar Chart (Killer Extra #5)
        if selected_stock == "All Stocks":
            st.markdown("---")
            st.markdown("### 🏭 Sector-Wise Accuracy Performance (30 Days)")
            st.caption("Categorized by industry sectors to highlight model domain efficacy.")
            
            df_30d_sector = df_30d.copy()
            if "Sector" not in df_30d_sector.columns:
                df_30d_sector["Sector"] = df_30d_sector["stock"].map(SECTOR_MAP).fillna("Other")
            
            sector_acc = df_30d_sector.groupby("Sector").apply(
                lambda g: (g["actual_direction"] == g["predicted_direction"]).mean() * 100
            ).reset_index(name="Accuracy")
            
            sector_acc = sector_acc.sort_values("Accuracy", ascending=True)
            
            # Color map based on accuracy thresholds: <50% red, 50-60% yellow, >60% green
            def get_color(acc):
                if acc < 50.0:
                    return "#ef5350"
                elif acc <= 60.0:
                    return "#ffca28"
                else:
                    return "#26a69a"
                    
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
            fig_sec.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50% Baseline")
            fig_sec.update_layout(
                template="plotly_dark",
                height=400,
                xaxis_range=[0, 100],
                xaxis_title="Accuracy (%)",
                yaxis_title="Sector"
            )
            st.plotly_chart(fig_sec, use_container_width=True)
