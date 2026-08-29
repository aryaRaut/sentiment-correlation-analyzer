"""
Trading Simulator & Backtesting Module for Sentiment-Correlation Analyzer.

Includes:
1. Realistic P&L Backtest Simulation with Transaction Costs.
2. Kelly Criterion inspired Confidence Position Sizing Strategy.
3. Rolling Window Walk-Forward Validation.
"""

import sys
import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.prediction_model import SentimentPredictor
from src.utils import setup_logger

logger = setup_logger("trading_simulator")

def simulate_trading_with_costs(df: pd.DataFrame, transaction_cost: float = 0.001) -> dict:
    """
    Simulate trading P&L with realistic transaction costs applied on position changes.
    
    Args:
        df (pd.DataFrame): Data containing Date, next_day_return, predicted_direction (or probability).
        transaction_cost (float): Fee per turnover (default: 0.1% or 0.001).
        
    Returns:
        dict: Detailed backtest summary metrics and cumulative P&L time-series dataframe.
    """
    df_sorted = df.dropna(subset=["next_day_return", "predicted_direction"]).sort_values("Date").copy()
    if df_sorted.empty:
        return {"error": "Empty dataframe", "summary": {}, "pnl_df": pd.DataFrame()}

    date_col = "Date" if "Date" in df_sorted.columns else "date"
    stock_col = "Symbol" if "Symbol" in df_sorted.columns else "stock"

    # Position: 1 if Long (UP), -1 if Short (DOWN) or 0 if Cash
    positions = np.where(df_sorted["predicted_direction"] == 1, 1.0, -1.0)
    returns = df_sorted["next_day_return"].values

    # Gross return per trade step
    gross_returns = positions * returns
    
    # Calculate position changes (turnover)
    # Position change at step t = abs(position[t] - position[t-1])
    pos_changes = np.abs(np.diff(positions, prepend=positions[0]))
    cost_deductions = pos_changes * transaction_cost

    net_returns = gross_returns - cost_deductions

    df_sorted["gross_return"] = gross_returns
    df_sorted["cost"] = cost_deductions
    df_sorted["net_return"] = net_returns
    df_sorted["buy_hold_return"] = returns

    # Daily aggregated timeline (averaging across stocks for universe P&L)
    daily = df_sorted.groupby(date_col)[["gross_return", "net_return", "buy_hold_return", "cost"]].mean().reset_index()
    daily["Gross P&L (%)"] = daily["gross_return"].cumsum() * 100.0
    daily["Net P&L (with Costs) (%)"] = daily["net_return"].cumsum() * 100.0
    daily["Buy & Hold P&L (%)"] = daily["buy_hold_return"].cumsum() * 100.0

    # Trade stats
    total_trades = np.count_nonzero(pos_changes > 0)
    winning_trades = np.count_nonzero(net_returns > 0)
    win_rate = (winning_trades / len(net_returns)) * 100.0 if len(net_returns) > 0 else 0.0

    total_net_pnl = float(daily["Net P&L (with Costs) (%)"].iloc[-1]) if not daily.empty else 0.0
    total_gross_pnl = float(daily["Gross P&L (%)"].iloc[-1]) if not daily.empty else 0.0
    total_bh_pnl = float(daily["Buy & Hold P&L (%)"].iloc[-1]) if not daily.empty else 0.0
    total_fees_paid = float(daily["cost"].sum() * 100.0)

    summary = {
        "Total Trades": total_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Net Strategy Return (%)": round(total_net_pnl, 2),
        "Gross Strategy Return (%)": round(total_gross_pnl, 2),
        "Buy & Hold Return (%)": round(total_bh_pnl, 2),
        "Total Fee Impact (%)": round(total_fees_paid, 2)
    }

    return {
        "summary": summary,
        "pnl_df": daily
    }

def position_sizing(confidence: float, max_position: float = 1.0) -> float:
    """
    Kelly Criterion inspired position sizing based on model prediction confidence.
    
    Args:
        confidence (float): Model confidence / probability (0.5 to 1.0).
        max_position (float): Maximum position cap (default: 1.0 or 100%).
        
    Returns:
        float: Position size multiplier between 0.0 and max_position.
    """
    if pd.isna(confidence) or confidence <= 0.50:
        return 0.10  # Base 10% minimum position
    
    # Scale linearly or non-linearly from 0.5 -> 0.10 up to 1.0 -> max_position
    scaled = 0.10 + ((confidence - 0.50) / 0.50) * (max_position - 0.10)
    return float(np.clip(scaled, 0.10, max_position))

def simulate_position_sized_trading(df: pd.DataFrame, transaction_cost: float = 0.001) -> dict:
    """
    Simulates trading P&L comparing Fixed Position Sizing (100%) vs Confidence-based Kelly Position Sizing.
    
    Args:
        df (pd.DataFrame): Dataset containing features, next_day_return, and target_up.
        
    Returns:
        dict: Summary comparison and cumulative returns dataframe.
    """
    df_sorted = df.dropna(subset=["next_day_return"]).sort_values("Date").copy()
    date_col = "Date" if "Date" in df_sorted.columns else "date"

    # Train model to get predicted probabilities
    feature_cols = SentimentPredictor.FEATURE_COLS
    for col in feature_cols:
        if col not in df_sorted.columns:
            df_sorted[col] = 0.0
        df_sorted[col] = df_sorted[col].fillna(0.0)

    sp = SentimentPredictor(df_sorted)
    X_train, X_test, y_train, y_test, test_df = sp.time_based_split()
    model = sp.train_xgboost(X_train, y_train)

    probs = model.predict_proba(df_sorted[feature_cols])[:, 1]
    df_sorted["prob_up"] = probs
    df_sorted["confidence"] = np.maximum(probs, 1.0 - probs)
    df_sorted["pred_direction"] = (probs >= 0.5).astype(int)

    # Position multiplier per row
    df_sorted["kelly_size"] = df_sorted["confidence"].apply(position_sizing)
    df_sorted["direction_mult"] = np.where(df_sorted["pred_direction"] == 1, 1.0, -1.0)

    df_sorted["fixed_return"] = df_sorted["direction_mult"] * df_sorted["next_day_return"] - transaction_cost
    df_sorted["kelly_return"] = df_sorted["direction_mult"] * df_sorted["kelly_size"] * df_sorted["next_day_return"] - (df_sorted["kelly_size"] * transaction_cost)

    daily = df_sorted.groupby(date_col)[["fixed_return", "kelly_return", "next_day_return", "kelly_size"]].mean().reset_index()
    daily["Fixed Sizing P&L (%)"] = daily["fixed_return"].cumsum() * 100.0
    daily["Kelly Position-Sized P&L (%)"] = daily["kelly_return"].cumsum() * 100.0
    daily["Buy & Hold P&L (%)"] = daily["next_day_return"].cumsum() * 100.0
    daily["Average Position Size (%)"] = daily["kelly_size"] * 100.0

    return {
        "fixed_pnl": round(float(daily["Fixed Sizing P&L (%)"].iloc[-1]), 2) if not daily.empty else 0.0,
        "kelly_pnl": round(float(daily["Kelly Position-Sized P&L (%)"].iloc[-1]), 2) if not daily.empty else 0.0,
        "avg_position_size": round(float(daily["Average Position Size (%)"].mean()), 2) if not daily.empty else 0.0,
        "timeline_df": daily
    }

def walk_forward_validation(df: pd.DataFrame, train_window: int = 60, test_window: int = 20) -> dict:
    """
    Perform rolling window walk-forward validation simulating real-world out-of-sample trading.
    
    Args:
        df (pd.DataFrame): Dataset sorted by Date.
        train_window (int): Number of days for expanding/rolling training window (default: 60).
        test_window (int): Number of days for out-of-sample test window (default: 20).
        
    Returns:
        dict: Overall metrics, window results dataframe, and rolling accuracy timeline.
    """
    logger.info("Executing Walk-Forward Rolling Window Validation...")
    date_col = "Date" if "Date" in df.columns else "date"
    target_col = "actual_direction" if "actual_direction" in df.columns else "target_up"

    df_clean = df.dropna(subset=["next_day_return", target_col]).sort_values(date_col).reset_index(drop=True)
    dates = df_clean[date_col].unique()

    feature_cols = SentimentPredictor.FEATURE_COLS
    for col in feature_cols:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
        df_clean[col] = df_clean[col].fillna(0.0)

    if len(dates) < train_window + test_window:
        # Fallback if historical dates are fewer than requested windows
        train_window = max(10, int(len(dates) * 0.6))
        test_window = max(5, int(len(dates) * 0.2))

    window_results = []
    all_test_preds = []

    start_idx = 0
    step = 0

    while start_idx + train_window < len(dates):
        train_dates = dates[start_idx : start_idx + train_window]
        test_dates = dates[start_idx + train_window : start_idx + train_window + test_window]

        if len(test_dates) == 0:
            break

        train_data = df_clean[df_clean[date_col].isin(train_dates)]
        test_data = df_clean[df_clean[date_col].isin(test_dates)].copy()

        if len(train_data) < 20 or len(test_data) == 0:
            start_idx += test_window
            continue

        X_tr, y_tr = train_data[feature_cols], train_data[target_col]
        X_te, y_te = test_data[feature_cols], test_data[target_col]

        model = XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        test_data["wf_prediction"] = preds

        acc = accuracy_score(y_te, preds)
        step += 1

        window_results.append({
            "Window": f"Window #{step}",
            "Train Start": str(train_dates[0]),
            "Train End": str(train_dates[-1]),
            "Test Start": str(test_dates[0]),
            "Test End": str(test_dates[-1]),
            "Accuracy": round(acc * 100.0, 1),
            "Samples": len(test_data)
        })

        all_test_preds.append(test_data)
        start_idx += test_window

    if not all_test_preds:
        return {
            "overall_accuracy": 50.0,
            "overall_f1": 0.5,
            "windows_df": pd.DataFrame(),
            "rolling_df": pd.DataFrame()
        }

    combined_test = pd.concat(all_test_preds).reset_index(drop=True)
    overall_acc = accuracy_score(combined_test[target_col], combined_test["wf_prediction"]) * 100.0
    overall_f1 = f1_score(combined_test[target_col], combined_test["wf_prediction"], zero_division=0) * 100.0

    # Rolling accuracy timeline
    combined_test["is_correct"] = (combined_test[target_col] == combined_test["wf_prediction"]).astype(float)
    rolling_df = combined_test.groupby(date_col)["is_correct"].mean().reset_index()
    rolling_df["Rolling Accuracy (%)"] = rolling_df["is_correct"].rolling(window=5, min_periods=1).mean() * 100.0

    return {
        "overall_accuracy": round(overall_acc, 2),
        "overall_f1": round(overall_f1, 2),
        "windows_df": pd.DataFrame(window_results),
        "rolling_df": rolling_df
    }

if __name__ == "__main__":
    from src.data_loader import load_processed_data
    df = load_processed_data()
    sim_res = simulate_trading_with_costs(df)
    print("Simulation Summary:", sim_res["summary"])
    pos_res = simulate_position_sized_trading(df)
    print("Fixed P&L:", pos_res["fixed_pnl"], "Kelly P&L:", pos_res["kelly_pnl"])
    wf_res = walk_forward_validation(df)
    print("Walk Forward Accuracy:", wf_res["overall_accuracy"], "%")
