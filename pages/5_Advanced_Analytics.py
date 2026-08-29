"""
Streamlit Page: Research-Grade Advanced Analytics (Page 5)

Dedicated dashboard for advanced financial AI features:
1. Causality & Explainability (Granger Causality, SHAP Force & Summary Plots, Lead-Lag Correlation)
2. Advanced Machine Learning (Voting Ensemble, Multi-Horizon Forecasts, PyTorch LSTM)
3. Risk & Volatility Analysis (Sharpe/Sortino/Calmar Ratios, Sentiment Volatility Regressor)
4. Practical Trading & Simulation (P&L with Transaction Costs, Kelly Position Sizing, Walk-Forward Validation)
5. Multi-Asset & Macro Analysis (Market Sentiment, Asset Correlation Network, Sector Performance)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import streamlit as st

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_processed_data, load_news_data, SECTOR_MAP
from src.causality_analyzer import test_granger_causality, find_optimal_lag
from src.ensemble_model import evaluate_ensemble_models, train_multi_horizon_models, train_lstm_model
from src.risk_metrics import compare_strategy_vs_buy_hold, predict_volatility
from src.trading_simulator import simulate_trading_with_costs, simulate_position_sized_trading, walk_forward_validation
from src.market_analyzer import calculate_market_sentiment, build_correlation_network, analyze_sector_performance
from src.prediction_model import SentimentPredictor

# Page Configuration
st.set_page_config(
    page_title="Advanced Analytics & Explainability",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, Modern & Minimalist Dark Theme Styling
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

.explanation-box {
    background-color: rgba(30, 41, 59, 0.7);
    border-left: 4px solid #3b82f6;
    padding: 10px 14px;
    border-radius: 4px;
    margin-top: 8px;
    margin-bottom: 16px;
    color: #cbd5e1;
    font-size: 0.85rem;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# Main Title
st.title("🔬 Advanced Analytics & Research Engine")
st.caption("Granger Causality, SHAP Explainability, Ensemble Voting, PyTorch LSTM, Risk Ratios & Market Networks")

# Load Dataset
with st.spinner("Initializing Advanced Analytics Engine..."):
    df = load_processed_data()

if df.empty:
    st.error("Unable to load processed dataset. Please verify data/processed/processed_dataset.csv.")
    st.stop()

# Sidebar Controls
stocks_list = sorted(list(df["stock"].unique()))
selected_stock = st.sidebar.selectbox("📌 Select NSE Stock Ticker", options=["All Stocks"] + stocks_list, index=0)

if selected_stock != "All Stocks":
    df_selected = df[df["stock"] == selected_stock].copy()
else:
    df_selected = df.copy()

# Cached expensive SHAP calculation
@st.cache_data(ttl=3600)
def compute_shap_explanations(_df):
    if not HAS_SHAP:
        return None, None, None, None, None
    sp = SentimentPredictor(_df)
    X_train, X_test, y_train, y_test, test_df = sp.time_based_split()
    model = sp.train_xgboost(X_train, y_train)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return model, explainer, shap_values, X_test, test_df

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Causality & Explainability",
    "🧪 Advanced ML & Neural Net",
    "📊 Risk & Volatility",
    "💰 Trading & Simulation",
    "🌍 Multi-Asset & Macro"
])

# ---------------------------------------------------------
# TAB 1: Causality & Explainability
# ---------------------------------------------------------
with tab1:
    st.subheader("1.1 📊 Granger Causality Testing")
    st.caption("Statistical hypothesis test determining if past news sentiment contains predictive signal for stock price returns.")

    single_stock_filter = selected_stock if selected_stock != "All Stocks" else None
    gc_results = test_granger_causality(df, stock=single_stock_filter, max_lag=5)

    if single_stock_filter:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card border-blue">
                <div class="card-title">Best Lag Tested</div>
                <div class="card-value card-value-blue">{gc_results['best_lag']} Days</div>
                <div class="card-subtext">Optimal Predictive Window</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Granger p-Value</div>
                <div class="card-value card-value-gold">{gc_results['p_value']:.4f}</div>
                <div class="card-subtext">Statistical Significance</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            causes_color = "card-value-green" if gc_results["causes"] == "Yes" else "card-value-red"
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Granger Causes Price?</div>
                <div class="card-value {causes_color}">{gc_results['causes']}</div>
                <div class="card-subtext">Threshold: p &lt; 0.05</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        summary_gc_df = gc_results.get("summary_table", pd.DataFrame())
        col_tb, col_bar = st.columns([1, 1])
        with col_tb:
            st.markdown("### Granger Causality Test Summary")
            st.dataframe(summary_gc_df.style.highlight_between(subset=["P-Value"], left=0.0, right=0.05, color="rgba(34, 197, 94, 0.25)"), use_container_width=True, height=350)
            
        with col_bar:
            st.markdown("### Granger p-Values Across Stocks")
            fig_p = px.bar(
                summary_gc_df,
                x="Stock",
                y="P-Value",
                color="Causes? (p < 0.05)",
                color_discrete_map={"Yes": "#22c55e", "No": "#ef4444"},
                title="p-Values by Stock (Lower = Stronger Causation)"
            )
            fig_p.add_hline(y=0.05, line_dash="dash", line_color="#eab308", annotation_text="Significance (p=0.05)")
            fig_p.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e293b",
                font=dict(color="#f8fafc", family="Inter"),
                height=350,
                margin=dict(l=15, r=15, t=40, b=15)
            )
            st.plotly_chart(fig_p, use_container_width=True)

    st.markdown('<div class="explanation-box">💡 <strong>Statistical Interpretation:</strong> If p &lt; 0.05, news sentiment statistically Granger-causes stock price movements at the given lag.</div>', unsafe_allow_html=True)
    st.divider()

    # 1.2 SHAP Model Explainability
    st.subheader("1.2 🧠 Why Did the Model Make This Prediction? (SHAP Values)")
    st.caption("SHAP (SHapley Additive exPlanations) breaks down exact feature contributions for any individual prediction.")

    if not HAS_SHAP:
        st.warning("⚠️ The `shap` package is not installed or enabled in the current environment. To enable SHAP force plots, run: `.\\venv\\Scripts\\python.exe -m pip install shap` and restart Streamlit using `.\\venv\\Scripts\\streamlit.exe run app.py`.")
    else:
        try:
            xgb_model, explainer, shap_vals, X_test_sample, test_df_sample = compute_shap_explanations(df_selected)

            c_sel1, c_sel2 = st.columns(2)
            with c_sel1:
                test_dates = test_df_sample["date"].astype(str).unique() if "date" in test_df_sample.columns else test_df_sample["Date"].astype(str).unique()
                selected_date_str = st.selectbox("Select Date for SHAP Breakdown", options=test_dates, index=len(test_dates)-1)
                
            sample_idx = test_df_sample[(test_df_sample["date"].astype(str) == selected_date_str) | (test_df_sample["Date"].astype(str) == selected_date_str)].index
            if len(sample_idx) > 0:
                row_pos = test_df_sample.index.get_loc(sample_idx[0])
                sample_features = X_test_sample.iloc[row_pos:row_pos+1]
                sample_shap = shap_vals[row_pos]

                col_shap1, col_shap2 = st.columns([1, 1])
                with col_shap1:
                    st.markdown(f"### Local Feature Force Breakdown ({selected_date_str})")
                    feat_impact = pd.DataFrame({
                        "Feature": X_test_sample.columns,
                        "Value": sample_features.values[0],
                        "SHAP Impact": sample_shap
                    }).sort_values("SHAP Impact", key=abs, ascending=True)

                    fig_shap_bar = px.bar(
                        feat_impact,
                        x="SHAP Impact",
                        y="Feature",
                        orientation="h",
                        color="SHAP Impact",
                        color_continuous_scale="RdYlGn",
                        title=f"Feature Push on Prediction for {selected_date_str}"
                    )
                    fig_shap_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#1e293b",
                        font=dict(color="#f8fafc", family="Inter"),
                        height=380,
                        margin=dict(l=15, r=15, t=40, b=15)
                    )
                    st.plotly_chart(fig_shap_bar, use_container_width=True)

                with col_shap2:
                    st.markdown("### Global SHAP Summary Feature Importance")
                    mean_shap = np.abs(shap_vals).mean(axis=0)
                    global_shap_df = pd.DataFrame({
                        "Feature": X_test_sample.columns,
                        "Mean |SHAP Value|": mean_shap
                    }).sort_values("Mean |SHAP Value|", ascending=False)

                    fig_glob_shap = px.bar(
                        global_shap_df,
                        x="Mean |SHAP Value|",
                        y="Feature",
                        orientation="h",
                        title="Global SHAP Feature Importance Score"
                    )
                    fig_glob_shap.update_traces(marker_color="#3b82f6")
                    fig_glob_shap.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="#1e293b",
                        font=dict(color="#f8fafc", family="Inter"),
                        height=380,
                        margin=dict(l=15, r=15, t=40, b=15),
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_glob_shap, use_container_width=True)
        except Exception as e:
            st.warning(f"Unable to compute SHAP force plot: {e}")

    st.markdown('<div class="explanation-box">💡 <strong>SHAP Guidance:</strong> Positive (green) bars push predictions towards bullish UP, while negative (red) bars push towards bearish DOWN.</div>', unsafe_allow_html=True)
    st.divider()

    # 1.3 Lead-Lag Analysis
    st.subheader("1.3 🔄 Lead-Lag Optimal Window Analysis")
    st.caption("Finds how many days prior news sentiment carries peak predictive power for stock returns.")

    lead_lag_res = find_optimal_lag(df, stock=single_stock_filter, max_lag=10)
    lag_corrs_dict = lead_lag_res.get("lag_correlations", {})
    lag_df = pd.DataFrame({
        "Lag (Days)": list(lag_corrs_dict.keys()),
        "Pearson Correlation (r)": list(lag_corrs_dict.values())
    })

    col_ll1, col_ll2 = st.columns([1, 1])
    with col_ll1:
        st.markdown(f"### Correlation vs Sentiment Lag (Peak: Lag {lead_lag_res['optimal_lag']})")
        fig_lag = px.line(
            lag_df,
            x="Lag (Days)",
            y="Pearson Correlation (r)",
            markers=True,
            title="Correlation by Sentiment Lag Shift"
        )
        fig_lag.update_traces(line_color="#3b82f6", line_width=2.5)
        fig_lag.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig_lag.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            height=350,
            margin=dict(l=15, r=15, t=40, b=15)
        )
        st.plotly_chart(fig_lag, use_container_width=True)

    with col_ll2:
        st.markdown("### Optimal Lag per Stock")
        summary_lag_df = lead_lag_res.get("summary_df", pd.DataFrame())
        if not summary_lag_df.empty:
            st.dataframe(summary_lag_df.style.highlight_max(subset=["Peak Correlation"], color="rgba(34, 197, 94, 0.25)"), use_container_width=True, height=350)
        else:
            st.info(f"Optimal Lag for {single_stock_filter}: Lag {lead_lag_res['optimal_lag']} (Correlation = {lead_lag_res['max_correlation']:.4f})")

    st.markdown('<div class="explanation-box">💡 <strong>Lead-Lag Interpretation:</strong> Shows how many days ahead news sentiment leads price reaction in the market.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: Advanced Machine Learning & Neural Networks
# ---------------------------------------------------------
with tab2:
    st.subheader("2.1 🧪 Soft Voting Ensemble Model Performance")
    st.caption("Combines predictions from XGBoost Classifier, Random Forest, and Logistic Regression using soft voting.")

    with st.spinner("Training Voting Ensemble..."):
        ens_df, _ = evaluate_ensemble_models(df_selected)

    st.dataframe(ens_df.style.highlight_max(subset=["Accuracy", "F1 Score"], color="rgba(34, 197, 94, 0.25)"), use_container_width=True)
    st.markdown('<div class="explanation-box">💡 <strong>Ensemble Advantage:</strong> Soft voting averages probabilities across non-correlated algorithms to reduce variance and boost out-of-sample stability.</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader("2.2 📈 Multi-Horizon Prediction (T+1, T+3, T+5, T+10)")
    st.caption("Trains distinct model instances to predict price direction across 1-day, 3-day, 5-day, and 10-day holding horizons.")

    with st.spinner("Training Multi-Horizon XGBoost Models..."):
        horizon_data = train_multi_horizon_models(df_selected, horizons=[1, 3, 5, 10])
        horizon_df = horizon_data["results_df"]

    col_h1, col_h2 = st.columns([1, 1])
    with col_h1:
        st.markdown("### Accuracy by Forecast Horizon")
        fig_hor = px.bar(
            horizon_df,
            x="Horizon",
            y="Accuracy",
            text=horizon_df["Accuracy"].apply(lambda x: f"{x:.1%}"),
            title="Directional Accuracy Across Time Horizons"
        )
        fig_hor.update_traces(marker_color="#3b82f6", textposition="outside")
        fig_hor.add_hline(y=0.50, line_dash="dash", line_color="#ef4444", annotation_text="50% Baseline")
        fig_hor.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            yaxis=dict(range=[0, 1.0]),
            height=350,
            margin=dict(l=15, r=15, t=40, b=15)
        )
        st.plotly_chart(fig_hor, use_container_width=True)

    with col_h2:
        st.markdown("### Horizon Metrics Table")
        st.dataframe(horizon_df[["Horizon", "Accuracy", "Precision", "Recall", "F1 Score", "Test Samples"]], use_container_width=True, height=350)

    st.markdown('<div class="explanation-box">💡 <strong>Multi-Horizon Insight:</strong> Comparing horizons reveals the temporal decay rate of news sentiment predictive signals.</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader("2.3 🔮 PyTorch LSTM Neural Network Performance")
    st.caption("Deep Learning sequence model capturing multi-day temporal dependencies across sentiment and technical features.")

    with st.spinner("Training PyTorch LSTM Neural Network..."):
        lstm_res = train_lstm_model(df_selected, epochs=25, window_size=5)

    if "error" in lstm_res:
        st.warning(lstm_res["error"])
    else:
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            st.markdown(f"""
            <div class="metric-card border-blue">
                <div class="card-title">LSTM Test Accuracy</div>
                <div class="card-value card-value-blue">{lstm_res['accuracy']:.1%}</div>
                <div class="card-subtext">5-Day Window Sequence</div>
            </div>
            """, unsafe_allow_html=True)
        with lc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">LSTM F1-Score</div>
                <div class="card-value card-value-green">{lstm_res['f1_score']:.1%}</div>
                <div class="card-subtext">Harmonic Mean</div>
            </div>
            """, unsafe_allow_html=True)
        with lc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Training Epochs</div>
                <div class="card-value card-value-gold">{lstm_res['epochs']}</div>
                <div class="card-subtext">PyTorch Adam Optimizer</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("### PyTorch LSTM Loss Curve (Training vs. Validation)")
        hist_df = lstm_res["history_df"]
        fig_loss = px.line(
            hist_df,
            x="Epoch",
            y=["Training Loss", "Validation Loss"],
            title="LSTM Loss Convergence Over Epochs"
        )
        fig_loss.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            height=350,
            margin=dict(l=15, r=15, t=40, b=15)
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    st.markdown('<div class="explanation-box">💡 <strong>LSTM Deep Learning:</strong> Recurrent memory gates preserve historical context across 5-day sequential sentiment blocks.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 3: Risk & Volatility Analysis
# ---------------------------------------------------------
with tab3:
    st.subheader("3.1 📊 Risk-Adjusted Performance Metrics")
    st.caption("Quantitative evaluation comparing Sharpe Ratio, Sortino Ratio, Maximum Drawdown, and Calmar Ratio.")

    risk_comp_df = compare_strategy_vs_buy_hold(df_selected, risk_free_rate=0.05)

    r_c1, r_c2, r_c3, r_c4 = st.columns(4)
    ai_sharpe = risk_comp_df[risk_comp_df["Metric"] == "Sharpe Ratio"]["AI Strategy"].values[0]
    ai_sortino = risk_comp_df[risk_comp_df["Metric"] == "Sortino Ratio"]["AI Strategy"].values[0]
    ai_mdd = risk_comp_df[risk_comp_df["Metric"] == "Max Drawdown"]["AI Strategy"].values[0]
    ai_calmar = risk_comp_df[risk_comp_df["Metric"] == "Calmar Ratio"]["AI Strategy"].values[0]

    with r_c1:
        st.markdown(f"""
        <div class="metric-card border-blue">
            <div class="card-title">Sharpe Ratio</div>
            <div class="card-value card-value-blue">{ai_sharpe}</div>
            <div class="card-subtext">Annualized Risk-Adjusted Return</div>
        </div>
        """, unsafe_allow_html=True)
    with r_c2:
        st.markdown(f"""
        <div class="metric-card border-green">
            <div class="card-title">Sortino Ratio</div>
            <div class="card-value card-value-green">{ai_sortino}</div>
            <div class="card-subtext">Downside Risk Penalty Only</div>
        </div>
        """, unsafe_allow_html=True)
    with r_c3:
        st.markdown(f"""
        <div class="metric-card border-red">
            <div class="card-title">Max Drawdown</div>
            <div class="card-value card-value-red">{ai_mdd}</div>
            <div class="card-subtext">Peak-to-Trough Downside</div>
        </div>
        """, unsafe_allow_html=True)
    with r_c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Calmar Ratio</div>
            <div class="card-value card-value-gold">{ai_calmar}</div>
            <div class="card-subtext">Return vs Max Drawdown</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Risk-Adjusted Metrics Summary (AI Strategy vs. Buy & Hold)")
    st.dataframe(risk_comp_df[["Metric", "AI Strategy", "Buy & Hold"]], use_container_width=True)

    st.markdown('<div class="explanation-box">💡 <strong>Risk Ratios:</strong> A Sharpe Ratio &gt; 1.0 and Sortino Ratio &gt; 1.5 indicate strong risk-adjusted returns after accounting for volatility.</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader("3.2 🎯 Sentiment Volatility Prediction")
    st.caption("Trains a Random Forest Regressor targeting absolute return magnitude (|return|) driven by sentiment volume and variance.")

    vol_res = predict_volatility(df_selected)
    vol_preds = vol_res["predictions_df"]
    vol_fi = vol_res["feature_importances"]

    v_col1, v_col2 = st.columns([1, 1])
    with v_col1:
        st.markdown("### Actual vs. Predicted Price Volatility")
        fig_vol = px.scatter(
            vol_preds,
            x="actual_volatility",
            y="predicted_volatility",
            color="Symbol" if "Symbol" in vol_preds.columns else "stock",
            title=f"Volatility Regression (MAE: {vol_res['mae']:.4f}, R²: {vol_res['r2_score']:.4f})"
        )
        fig_vol.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            height=350,
            margin=dict(l=15, r=15, t=40, b=15)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with v_col2:
        st.markdown("### Feature Importance for Volatility Prediction")
        fig_vfi = px.bar(
            vol_fi,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Volatility Driver Feature Importance"
        )
        fig_vfi.update_traces(marker_color="#22c55e")
        fig_vfi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            height=350,
            margin=dict(l=15, r=15, t=40, b=15),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_vfi, use_container_width=True)

    st.markdown('<div class="explanation-box">💡 <strong>Volatility Insight:</strong> Sentiment volatility prediction assists risk managers in scaling down position sizes during expected high-volatility regime spikes.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 4: Practical Trading & Simulation
# ---------------------------------------------------------
with tab4:
    st.subheader("4.1 💰 Advanced P&L Simulation with Transaction Costs")
    st.caption("Realistic backtesting simulation incorporating per-trade friction fees (0.1% turnover cost per trade entry/exit).")

    fee_pct = st.slider("Transaction Fee per Trade Turnover (%)", min_value=0.0, max_value=0.5, value=0.1, step=0.05) / 100.0
    cost_res = simulate_trading_with_costs(df_selected, transaction_cost=fee_pct)

    summary_tc = cost_res["summary"]
    pnl_tc_df = cost_res["pnl_df"]

    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    with t_c1:
        st.markdown(f"""
        <div class="metric-card border-blue">
            <div class="card-title">Net Strategy P&L</div>
            <div class="card-value card-value-green">{summary_tc.get('Net Strategy Return (%)', 0.0):+.2f}%</div>
            <div class="card-subtext">After Transaction Costs</div>
        </div>
        """, unsafe_allow_html=True)
    with t_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Win Rate (%)</div>
            <div class="card-value card-value-gold">{summary_tc.get('Win Rate (%)', 0.0):.1f}%</div>
            <div class="card-subtext">Across {summary_tc.get('Total Trades', 0)} Trades</div>
        </div>
        """, unsafe_allow_html=True)
    with t_c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Buy & Hold Return</div>
            <div class="card-value card-value-blue">{summary_tc.get('Buy & Hold Return (%)', 0.0):+.2f}%</div>
            <div class="card-subtext">Baseline Performance</div>
        </div>
        """, unsafe_allow_html=True)
    with t_c4:
        st.markdown(f"""
        <div class="metric-card border-red">
            <div class="card-title">Total Fees Paid</div>
            <div class="card-value card-value-red">{summary_tc.get('Total Fee Impact (%)', 0.0):.2f}%</div>
            <div class="card-subtext">Friction Impact</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Cumulative P&L Comparison (Gross vs. Net after Costs vs. Buy & Hold)")
    fig_tc = go.Figure()
    fig_tc.add_trace(go.Scatter(
        x=pnl_tc_df["date" if "date" in pnl_tc_df.columns else "Date"],
        y=pnl_tc_df["Net P&L (with Costs) (%)"],
        mode="lines", name="Net Strategy P&L (with Costs)",
        line=dict(color="#22c55e", width=2.5)
    ))
    fig_tc.add_trace(go.Scatter(
        x=pnl_tc_df["date" if "date" in pnl_tc_df.columns else "Date"],
        y=pnl_tc_df["Gross P&L (%)"],
        mode="lines", name="Gross Strategy P&L (No Costs)",
        line=dict(color="#3b82f6", width=2, dash="dot")
    ))
    fig_tc.add_trace(go.Scatter(
        x=pnl_tc_df["date" if "date" in pnl_tc_df.columns else "Date"],
        y=pnl_tc_df["Buy & Hold P&L (%)"],
        mode="lines", name="Buy & Hold P&L",
        line=dict(color="#eab308", width=2, dash="dash")
    ))
    fig_tc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font=dict(color="#f8fafc", family="Inter"),
        height=380,
        margin=dict(l=15, r=15, t=40, b=15),
        legend=dict(orientation="h", y=1.12, x=0.01, bgcolor="rgba(30, 41, 59, 0.6)")
    )
    st.plotly_chart(fig_tc, use_container_width=True)

    st.markdown('<div class="explanation-box">💡 <strong>Transaction Cost Reality:</strong> Frequent rebalancing incurs trading fees that erode gross returns.</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader("4.2 🏦 Position Sizing Strategy (Kelly Criterion)")
    st.caption("Scales trade position size based on prediction confidence (0.50 → 10% base capital, 1.00 → 100% max position).")

    kelly_res = simulate_position_sized_trading(df_selected, transaction_cost=fee_pct)
    kelly_df = kelly_res["timeline_df"]

    k_col1, k_col2 = st.columns([1, 1])
    with k_col1:
        st.markdown("### Fixed Sizing vs. Kelly Confidence Sizing Cumulative P&L")
        fig_k = go.Figure()
        fig_k.add_trace(go.Scatter(
            x=kelly_df["date" if "date" in kelly_df.columns else "Date"],
            y=kelly_df["Kelly Position-Sized P&L (%)"],
            mode="lines", name="Kelly Confidence Sizing P&L",
            line=dict(color="#22c55e", width=2.5)
        ))
        fig_k.add_trace(go.Scatter(
            x=kelly_df["date" if "date" in kelly_df.columns else "Date"],
            y=kelly_df["Fixed Sizing P&L (%)"],
            mode="lines", name="Fixed 100% Sizing P&L",
            line=dict(color="#3b82f6", width=2, dash="dash")
        ))
        fig_k.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            height=350,
            margin=dict(l=15, r=15, t=40, b=15),
            legend=dict(orientation="h", y=1.12, x=0.01, bgcolor="rgba(30, 41, 59, 0.6)")
        )
        st.plotly_chart(fig_k, use_container_width=True)

    with k_col2:
        st.markdown("### Average Dynamic Position Capital Allocation")
        fig_size = px.histogram(
            kelly_df,
            x="Average Position Size (%)",
            nbins=15,
            title=f"Position Allocation Distribution (Avg: {kelly_res['avg_position_size']}%)"
        )
        fig_size.update_traces(marker_color="#eab308")
        fig_size.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            height=350,
            margin=dict(l=15, r=15, t=40, b=15)
        )
        st.plotly_chart(fig_size, use_container_width=True)

    st.markdown('<div class="explanation-box">💡 <strong>Position Sizing Logic:</strong> Higher prediction confidence allocates larger capital size, protecting capital during weak signals.</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader("4.3 📈 Walk-Forward Rolling Window Validation")
    st.caption("Simulates true out-of-sample trading by retraining on rolling 60-day historical windows and testing on subsequent 20-day windows.")

    wf_res = walk_forward_validation(df_selected, train_window=60, test_window=20)
    wf_win_df = wf_res["windows_df"]
    wf_roll_df = wf_res["rolling_df"]

    w_c1, w_c2 = st.columns(2)
    with w_c1:
        st.markdown(f"""
        <div class="metric-card border-blue">
            <div class="card-title">Overall Walk-Forward Accuracy</div>
            <div class="card-value card-value-gold">{wf_res['overall_accuracy']:.1f}%</div>
            <div class="card-subtext">True Out-of-Sample Validation</div>
        </div>
        """, unsafe_allow_html=True)
    with w_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Walk-Forward F1-Score</div>
            <div class="card-value card-value-blue">{wf_res['overall_f1']:.1f}%</div>
            <div class="card-subtext">Harmonic Mean Across Windows</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    w_col1, w_col2 = st.columns([1, 1])
    with w_col1:
        st.markdown("### Rolling Window Test Accuracy Trend")
        if not wf_roll_df.empty:
            fig_wf = px.line(
                wf_roll_df,
                x="date" if "date" in wf_roll_df.columns else "Date",
                y="Rolling Accuracy (%)",
                title="5-Day Rolling Walk-Forward Accuracy"
            )
            fig_wf.add_hline(y=50, line_dash="dash", line_color="#ef4444", annotation_text="50% Baseline")
            fig_wf.update_traces(line_color="#3b82f6", line_width=2.5)
            fig_wf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1e293b",
                font=dict(color="#f8fafc", family="Inter"),
                yaxis=dict(range=[0, 100]),
                height=350,
                margin=dict(l=15, r=15, t=40, b=15)
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    with w_col2:
        st.markdown("### Walk-Forward Rolling Windows Breakdown")
        if not wf_win_df.empty:
            st.dataframe(wf_win_df, use_container_width=True, height=350)

    st.markdown('<div class="explanation-box">💡 <strong>Walk-Forward Validation:</strong> Prevents look-ahead bias by strictly testing models on data that strictly succeeds the training window.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 5: Multi-Asset & Macro Analysis
# ---------------------------------------------------------
with tab5:
    st.subheader("5.1 🌍 Market-Wide Sentiment & Aggregate Returns")
    st.caption("Aggregates daily sentiment across all NSE stocks to capture systemic market-wide sentiment trends.")

    mkt_sent_res = calculate_market_sentiment(df)
    mkt_df = mkt_sent_res["market_df"]

    m_c1, m_c2, m_c3 = st.columns(3)
    with m_c1:
        st.markdown(f"""
        <div class="metric-card border-blue">
            <div class="card-title">Market Sentiment Correlation (r)</div>
            <div class="card-value card-value-blue">{mkt_sent_res['r']:.4f}</div>
            <div class="card-subtext">Systemic Market Linear Fit</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">p-Value</div>
            <div class="card-value card-value-gold">{mkt_sent_res['p_value']:.4f}</div>
            <div class="card-subtext">Statistical Significance</div>
        </div>
        """, unsafe_allow_html=True)
    with m_c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">Significant Market Relationship?</div>
            <div class="card-value card-value-green">{mkt_sent_res['significant']}</div>
            <div class="card-subtext">Macro Signal</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("### Universe Market Sentiment vs. Index Cumulative Return")
    fig_mkt = make_subplots(specs=[[{"secondary_y": True}]])
    fig_mkt.add_trace(
        go.Scatter(
            x=mkt_df["Date" if "Date" in mkt_df.columns else "date"],
            y=mkt_df["market_sentiment"],
            name="Universe Market Sentiment",
            line=dict(color="#3b82f6", width=2)
        ),
        secondary_y=False
    )
    fig_mkt.add_trace(
        go.Scatter(
            x=mkt_df["Date" if "Date" in mkt_df.columns else "date"],
            y=mkt_df["cum_market_return"],
            name="Cumulative Market Return (%)",
            line=dict(color="#22c55e", width=2.5)
        ),
        secondary_y=True
    )
    fig_mkt.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1e293b",
        font=dict(color="#f8fafc", family="Inter"),
        height=380,
        margin=dict(l=15, r=15, t=40, b=15),
        legend=dict(orientation="h", y=1.12, x=0.01, bgcolor="rgba(30, 41, 59, 0.6)")
    )
    st.plotly_chart(fig_mkt, use_container_width=True)

    st.markdown('<div class="explanation-box">💡 <strong>Market Sentiment:</strong> Market-level sentiment reveals whether aggregate sentiment drives broad market index movements.</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader("5.2 📊 Cross-Asset Correlation Network Analysis")
    st.caption("Computes stock-to-stock return correlation matrix to uncover highly correlated asset clusters.")

    net_res = build_correlation_network(df)
    corr_mat = net_res["corr_matrix"]
    clusters_df = net_res["clusters"]

    n_col1, n_col2 = st.columns([3, 2])
    with n_col1:
        st.markdown("### Stock Return Correlation Matrix Heatmap")
        fig_net_hm = px.imshow(
            corr_mat,
            labels=dict(x="Stock Ticker", y="Stock Ticker", color="Correlation"),
            color_continuous_scale="RdBu_r",
            zmin=-1.0, zmax=1.0,
            title="Cross-Stock Return Correlation Heatmap"
        )
        fig_net_hm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            height=420,
            margin=dict(l=15, r=15, t=40, b=15)
        )
        st.plotly_chart(fig_net_hm, use_container_width=True)

    with n_col2:
        st.markdown("### Highly Correlated Stock Clusters")
        if not clusters_df.empty:
            st.dataframe(clusters_df.style.background_gradient(cmap="Blues", subset=["Correlation"]), use_container_width=True, height=420)
        else:
            st.info("No stock pairs exceeding threshold correlation found.")

    st.markdown('<div class="explanation-box">💡 <strong>Cluster Analysis:</strong> Stocks within the same sector or financial dependency group move together, creating cross-asset co-movement patterns.</div>', unsafe_allow_html=True)
    st.divider()

    st.subheader("5.3 📈 Sector Performance Comparison")
    st.caption("Aggregates returns, volatility, average sentiment, and model accuracy grouped by market sector.")

    sec_perf_df = analyze_sector_performance(df, sector_map=SECTOR_MAP)

    s_col1, s_col2 = st.columns([1, 1])
    with s_col1:
        st.markdown("### Sector Model Accuracy (%)")
        fig_sec_acc = px.bar(
            sec_perf_df,
            x="Model Accuracy (%)",
            y="Sector",
            orientation="h",
            text=sec_perf_df["Model Accuracy (%)"].apply(lambda x: f"{x:.1f}%"),
            title="Directional Prediction Accuracy by Sector"
        )
        fig_sec_acc.update_traces(marker_color="#22c55e", textposition="outside")
        fig_sec_acc.add_vline(x=50, line_dash="dash", line_color="#ef4444", annotation_text="50% Baseline")
        fig_sec_acc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#1e293b",
            font=dict(color="#f8fafc", family="Inter"),
            xaxis=dict(range=[0, 100]),
            height=350,
            margin=dict(l=15, r=15, t=40, b=15),
            yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig_sec_acc, use_container_width=True)

    with s_col2:
        st.markdown("### Sector Metrics Breakdown Table")
        st.dataframe(sec_perf_df, use_container_width=True, height=350)

    st.markdown('<div class="explanation-box">💡 <strong>Sector Comparison:</strong> Highlights outperforming sectors and reveals where news sentiment has the highest predictive hit rate.</div>', unsafe_allow_html=True)

st.divider()
st.caption("NSE Sentiment-Correlation Analyzer • Research-Grade Advanced Analytics Engine")
