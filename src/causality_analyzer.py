"""
Causality & Lead-Lag Analyzer Module for Sentiment-Correlation Analyzer.

Provides functions to test Granger Causality (sentiment -> return predictability)
and calculate optimal lead-lag relationships using statsmodels and cross-correlation.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
from statsmodels.tsa.stattools import grangercausalitytests

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import setup_logger

logger = setup_logger("causality_analyzer")

def test_granger_causality(df: pd.DataFrame, stock: str = None, max_lag: int = 5) -> dict:
    """
    Test if sentiment Granger-causes price returns using statsmodels.
    
    Args:
        df (pd.DataFrame): Processed stock & sentiment dataset.
        stock (str, optional): Ticker symbol. If None, computes per-stock summary table.
        max_lag (int): Maximum lag to test (default: 5).
        
    Returns:
        dict: Results containing best lag, min p-value, causation conclusion, and summary table.
    """
    if df.empty:
        return {"error": "DataFrame is empty", "summary_table": pd.DataFrame(), "best_lag": 1, "p_value": 1.0, "causes": "No"}

    # Standardize column names
    sentiment_col = "avg_sentiment" if "avg_sentiment" in df.columns else "sentiment_score"
    return_col = "next_day_return" if "next_day_return" in df.columns else "current_day_return"
    stock_col = "Symbol" if "Symbol" in df.columns else "stock"

    valid_df = df.dropna(subset=[sentiment_col, return_col]).copy()
    
    if stock is not None:
        valid_df = valid_df[valid_df[stock_col] == stock]

    if stock is not None or len(valid_df[stock_col].unique()) == 1:
        # Single stock test
        ticker = stock if stock else (valid_df[stock_col].iloc[0] if not valid_df.empty else "Stock")
        if len(valid_df) < max_lag + 10:
            return {
                "stock": ticker,
                "best_lag": 1,
                "p_value": 1.0,
                "causes": "No (Insufficient Data)",
                "details": {}
            }

        # Data format for grangercausalitytests: [y, x] where x Granger-causes y
        test_data = valid_df[[return_col, sentiment_col]].replace([np.inf, -np.inf], np.nan).dropna()
        
        # Check for zero variance
        if test_data[return_col].std() == 0 or test_data[sentiment_col].std() == 0:
            return {
                "stock": ticker,
                "best_lag": 1,
                "p_value": 1.0,
                "causes": "No (Zero Variance)",
                "details": {}
            }

        min_p = 1.0
        best_lag = 1
        lag_results = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                gc_res = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                for lag in range(1, max_lag + 1):
                    if lag in gc_res:
                        # Extract p-value from ssr_ftest
                        p_val = gc_res[lag][0]['ssr_ftest'][1]
                        lag_results[lag] = round(float(p_val), 4)
                        if p_val < min_p:
                            min_p = float(p_val)
                            best_lag = lag
            except Exception as e:
                logger.warning(f"Granger test failed for {ticker}: {e}")

        causes = "Yes" if min_p < 0.05 else "No"
        return {
            "stock": ticker,
            "best_lag": best_lag,
            "p_value": round(min_p, 4),
            "causes": causes,
            "details": lag_results
        }
    else:
        # Universe breakdown
        results = []
        for symbol, group in valid_df.groupby(stock_col):
            res = test_granger_causality(group, stock=symbol, max_lag=max_lag)
            results.append({
                "Stock": symbol,
                "Best Lag": res["best_lag"],
                "P-Value": res["p_value"],
                "Causes? (p < 0.05)": res["causes"]
            })
            
        res_df = pd.DataFrame(results)
        if not res_df.empty:
            res_df = res_df.sort_values("P-Value").reset_index(drop=True)
            
        return {
            "summary_table": res_df,
            "significant_count": len(res_df[res_df["Causes? (p < 0.05)"] == "Yes"]) if not res_df.empty else 0
        }

def find_optimal_lag(df: pd.DataFrame, stock: str = None, max_lag: int = 10) -> dict:
    """
    Find which lag gives the strongest correlation between sentiment and returns.
    
    Args:
        df (pd.DataFrame): Stock & sentiment dataset.
        stock (str, optional): Ticker symbol. If None, computes per-stock summary.
        max_lag (int): Maximum lag to shift sentiment (0 to max_lag).
        
    Returns:
        dict: Optimal lag, correlation values across lags, and per-stock summary.
    """
    if df.empty:
        return {"optimal_lag": 0, "max_correlation": 0.0, "lag_correlations": {}, "summary_df": pd.DataFrame()}

    sentiment_col = "avg_sentiment" if "avg_sentiment" in df.columns else "sentiment_score"
    return_col = "next_day_return" if "next_day_return" in df.columns else "current_day_return"
    stock_col = "Symbol" if "Symbol" in df.columns else "stock"

    if stock is not None:
        sub_df = df[df[stock_col] == stock].sort_values("Date").copy()
    else:
        sub_df = df.sort_values("Date").copy()

    lag_corrs = {}
    for lag in range(0, max_lag + 1):
        if stock is not None or len(df[stock_col].unique()) == 1:
            shifted_sent = sub_df[sentiment_col].shift(lag)
            valid = pd.DataFrame({"sent": shifted_sent, "ret": sub_df[return_col]}).dropna()
        else:
            # Group by stock to shift sentiment properly
            shifted_list = []
            for s_name, s_group in sub_df.groupby(stock_col):
                g_copy = s_group.copy()
                g_copy["sent_shift"] = g_copy[sentiment_col].shift(lag)
                shifted_list.append(g_copy)
            combined = pd.concat(shifted_list)
            valid = combined[["sent_shift", return_col]].rename(columns={"sent_shift": "sent", return_col: "ret"}).dropna()

        if len(valid) > 3 and valid["sent"].std() > 0 and valid["ret"].std() > 0:
            r, _ = stats.pearsonr(valid["sent"], valid["ret"])
            lag_corrs[lag] = round(float(r), 4)
        else:
            lag_corrs[lag] = 0.0

    # Find lag with maximum absolute correlation
    optimal_lag = max(lag_corrs.keys(), key=lambda l: abs(lag_corrs[l])) if lag_corrs else 0
    max_corr = lag_corrs.get(optimal_lag, 0.0)

    # Per-stock optimal lag summary if universe
    summary_list = []
    if stock is None and len(df[stock_col].unique()) > 1:
        for symbol, group in df.groupby(stock_col):
            single_res = find_optimal_lag(group, stock=symbol, max_lag=max_lag)
            summary_list.append({
                "Stock": symbol,
                "Optimal Lag (Days)": single_res["optimal_lag"],
                "Peak Correlation": single_res["max_correlation"]
            })
            
    summary_df = pd.DataFrame(summary_list)
    if not summary_df.empty and "Peak Correlation" in summary_df.columns:
        summary_df = summary_df.sort_values("Peak Correlation", ascending=False).reset_index(drop=True)

    return {
        "stock": stock if stock else "Universe",
        "optimal_lag": optimal_lag,
        "max_correlation": max_corr,
        "lag_correlations": lag_corrs,
        "summary_df": summary_df
    }

if __name__ == "__main__":
    from src.data_loader import load_processed_data
    df = load_processed_data()
    print("Granger Test Result:", test_granger_causality(df))
    opt_res = find_optimal_lag(df)
    print("Optimal Lag Result (Optimal):", opt_res["optimal_lag"], opt_res["max_correlation"])
    print(opt_res["summary_df"].head())
