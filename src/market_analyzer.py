"""
Multi-Asset & Macro Market Analyzer Module for Sentiment-Correlation Analyzer.

Includes:
1. Market-Wide Daily Sentiment & Index Return Aggregation.
2. Cross-Asset Correlation Matrix & Cluster Analysis.
3. Sector-Wise Performance & Sentiment Comparison.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import SECTOR_MAP
from src.utils import setup_logger

logger = setup_logger("market_analyzer")

def calculate_market_sentiment(df: pd.DataFrame) -> dict:
    """
    Calculate daily market-wide aggregate sentiment and average market returns.
    
    Args:
        df (pd.DataFrame): Stock & sentiment dataset.
        
    Returns:
        dict: Market-wide time series dataframe and statistical correlation metrics.
    """
    if df.empty:
        return {"market_df": pd.DataFrame(), "r": 0.0, "p_value": 1.0}

    date_col = "Date" if "Date" in df.columns else "date"
    sent_col = "avg_sentiment" if "avg_sentiment" in df.columns else "sentiment_score"
    ret_col = "next_day_return" if "next_day_return" in df.columns else "current_day_return"

    df_clean = df.dropna(subset=[sent_col, ret_col]).copy()
    
    daily_mkt = df_clean.groupby(date_col).agg(
        market_sentiment=(sent_col, "mean"),
        market_return=(ret_col, "mean"),
        article_volume=("news_count" if "news_count" in df.columns else sent_col, "count")
    ).reset_index()

    daily_mkt = daily_mkt.sort_values(date_col).reset_index(drop=True)
    daily_mkt["cum_market_return"] = daily_mkt["market_return"].cumsum() * 100.0

    if len(daily_mkt) > 3 and daily_mkt["market_sentiment"].std() > 0 and daily_mkt["market_return"].std() > 0:
        r, p = stats.pearsonr(daily_mkt["market_sentiment"], daily_mkt["market_return"])
    else:
        r, p = 0.0, 1.0

    return {
        "market_df": daily_mkt,
        "r": round(float(r), 4),
        "p_value": round(float(p), 4),
        "significant": "Yes (p < 0.05)" if p < 0.05 else "No"
    }

def build_correlation_network(df: pd.DataFrame, min_periods: int = 5) -> dict:
    """
    Build cross-asset return correlation matrix and identify highly correlated stock clusters.
    
    Args:
        df (pd.DataFrame): Dataset containing Symbol/stock, Date, and returns.
        min_periods (int): Minimum required overlapping dates for correlation.
        
    Returns:
        dict: Correlation matrix DataFrame and list of highly correlated stock pairs/clusters.
    """
    if df.empty:
        return {"corr_matrix": pd.DataFrame(), "clusters": []}

    date_col = "Date" if "Date" in df.columns else "date"
    stock_col = "Symbol" if "Symbol" in df.columns else "stock"
    ret_col = "next_day_return" if "next_day_return" in df.columns else "current_day_return"

    df_clean = df.dropna(subset=[stock_col, ret_col, date_col]).copy()
    
    # Pivot returns table: index = Date, columns = Stock Symbol, values = Returns
    pivoted = df_clean.pivot_table(index=date_col, columns=stock_col, values=ret_col)
    
    corr_matrix = pivoted.corr(min_periods=min_periods).fillna(0.0)

    # Extract top correlated clusters/pairs (r > 0.40 excluding diagonal)
    pairs = []
    symbols = corr_matrix.columns
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            s1, s2 = symbols[i], symbols[j]
            r_val = corr_matrix.loc[s1, s2]
            if abs(r_val) >= 0.35:
                pairs.append({
                    "Stock A": s1,
                    "Stock B": s2,
                    "Correlation": round(float(r_val), 4),
                    "Relationship": "Strongly Positive" if r_val > 0 else "Inverse Correlation"
                })

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("Correlation", ascending=False).reset_index(drop=True)

    return {
        "corr_matrix": corr_matrix,
        "clusters": pairs_df
    }

def analyze_sector_performance(df: pd.DataFrame, sector_map: dict = None) -> pd.DataFrame:
    """
    Calculate performance metrics, volatility, sentiment, and accuracy by market sector.
    
    Args:
        df (pd.DataFrame): Stock & sentiment dataset.
        sector_map (dict, optional): Ticker to sector dictionary mapping.
        
    Returns:
        pd.DataFrame: Sector performance summary dataframe.
    """
    if df.empty:
        return pd.DataFrame()

    if sector_map is None:
        sector_map = SECTOR_MAP

    stock_col = "Symbol" if "Symbol" in df.columns else "stock"
    sent_col = "avg_sentiment" if "avg_sentiment" in df.columns else "sentiment_score"
    ret_col = "next_day_return" if "next_day_return" in df.columns else "current_day_return"
    pred_col = "predicted_direction" if "predicted_direction" in df.columns else "target_up"
    act_col = "actual_direction" if "actual_direction" in df.columns else "target_up"

    df_copy = df.copy()
    if "Sector" not in df_copy.columns:
        df_copy["Sector"] = df_copy[stock_col].map(sector_map).fillna("Other")

    df_clean = df_copy.dropna(subset=[ret_col]).copy()

    sector_stats = []
    for sector_name, group in df_clean.groupby("Sector"):
        avg_ret = group[ret_col].mean() * 100.0
        volatility = group[ret_col].std() * np.sqrt(252) * 100.0 if len(group) > 1 else 0.0
        avg_sent = group[sent_col].mean() if sent_col in group.columns else 0.0
        
        # Calculate model accuracy for sector if prediction columns exist
        if pred_col in group.columns and act_col in group.columns:
            valid_preds = group.dropna(subset=[pred_col, act_col])
            acc = (valid_preds[pred_col] == valid_preds[act_col]).mean() * 100.0 if len(valid_preds) > 0 else 50.0
        else:
            acc = 50.0

        stock_count = len(group[stock_col].unique())

        sector_stats.append({
            "Sector": sector_name,
            "Stock Count": stock_count,
            "Average Return (%)": round(float(avg_ret), 2),
            "Annual Volatility (%)": round(float(volatility), 2),
            "Mean Sentiment": round(float(avg_sent), 3),
            "Model Accuracy (%)": round(float(acc), 1)
        })

    sec_df = pd.DataFrame(sector_stats)
    if not sec_df.empty:
        sec_df = sec_df.sort_values("Average Return (%)", ascending=False).reset_index(drop=True)

    return sec_df

if __name__ == "__main__":
    from src.data_loader import load_processed_data
    df = load_processed_data()
    mkt_res = calculate_market_sentiment(df)
    print("Market Sentiment r:", mkt_res["r"], "p-value:", mkt_res["p_value"])
    net_res = build_correlation_network(df)
    print("Correlation Pairs:")
    print(net_res["clusters"].head())
    sec_res = analyze_sector_performance(df)
    print("\nSector Performance Breakdown:")
    print(sec_res)
