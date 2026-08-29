"""
Risk & Volatility Analysis Module for Sentiment-Correlation Analyzer.

Includes:
1. Risk-adjusted performance metrics: Sharpe, Sortino, Max Drawdown, Calmar Ratios.
2. Sentiment-driven price volatility prediction using RandomForestRegressor.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils import setup_logger

logger = setup_logger("risk_metrics")

def calculate_risk_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> dict:
    """
    Calculate professional risk-adjusted investment metrics.
    
    Args:
        returns (pd.Series): Series of daily percentage returns (e.g. 0.01 for 1%).
        risk_free_rate (float): Annual risk-free rate (default: 5% or 0.05).
        
    Returns:
        dict: Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Annualized Return, Volatility.
    """
    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return {
            "Annualized Return": 0.0,
            "Annualized Volatility": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Max Drawdown": 0.0,
            "Calmar Ratio": 0.0
        }

    # Daily to Annual scaling
    daily_rf = risk_free_rate / 252.0
    mean_daily = clean_returns.mean()
    std_daily = clean_returns.std()
    
    ann_return = mean_daily * 252.0
    ann_vol = std_daily * np.sqrt(252) if std_daily > 0 else 1e-6

    # Sharpe Ratio
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

    # Sortino Ratio (Downside deviation)
    downside_returns = clean_returns[clean_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 and downside_returns.std() > 0 else 1e-6
    sortino = (ann_return - risk_free_rate) / downside_std

    # Maximum Drawdown calculation
    cum_returns = (1 + clean_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    # Calmar Ratio
    calmar = ann_return / abs(max_drawdown) if abs(max_drawdown) > 1e-6 else 0.0

    return {
        "Annualized Return": round(float(ann_return), 4),
        "Annualized Volatility": round(float(ann_vol), 4),
        "Sharpe Ratio": round(float(sharpe), 4),
        "Sortino Ratio": round(float(sortino), 4),
        "Max Drawdown": round(float(max_drawdown), 4),
        "Calmar Ratio": round(float(calmar), 4)
    }

def compare_strategy_vs_buy_hold(df: pd.DataFrame, risk_free_rate: float = 0.05) -> pd.DataFrame:
    """
    Computes risk metrics for AI Directional Strategy vs Buy & Hold across the dataset.
    
    Args:
        df (pd.DataFrame): Dataset with next_day_return and predicted_direction.
        
    Returns:
        pd.DataFrame: Comparison table of metrics.
    """
    df_clean = df.dropna(subset=["next_day_return", "predicted_direction"]).copy()
    
    # AI Strategy returns: Long if predicted 1, Short if predicted 0
    ai_returns = np.where(
        df_clean["predicted_direction"] == 1,
        df_clean["next_day_return"],
        -df_clean["next_day_return"]
    )
    bh_returns = df_clean["next_day_return"]

    ai_metrics = calculate_risk_metrics(pd.Series(ai_returns), risk_free_rate=risk_free_rate)
    bh_metrics = calculate_risk_metrics(bh_returns, risk_free_rate=risk_free_rate)

    comp_rows = []
    for metric_name in ai_metrics.keys():
        ai_val = ai_metrics[metric_name]
        bh_val = bh_metrics[metric_name]
        
        # Formatting string representation
        if "Return" in metric_name or "Volatility" in metric_name or "Drawdown" in metric_name:
            ai_str = f"{ai_val:+.2%}"
            bh_str = f"{bh_val:+.2%}"
        else:
            ai_str = f"{ai_val:.2f}"
            bh_str = f"{bh_val:.2f}"
            
        comp_rows.append({
            "Metric": metric_name,
            "AI Strategy": ai_str,
            "Buy & Hold": bh_str,
            "raw_ai": ai_val,
            "raw_bh": bh_val
        })

    return pd.DataFrame(comp_rows)

def predict_volatility(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest Regressor to predict price volatility (absolute return).
    
    Args:
        df (pd.DataFrame): Processed stock & sentiment dataset.
        
    Returns:
        dict: Trained model, evaluation metrics (MAE, RMSE, R2), predictions df, feature importances.
    """
    logger.info("Training Sentiment Volatility Predictor...")
    df_clean = df.copy().dropna(subset=["next_day_return"]).reset_index(drop=True)
    
    # Target: absolute next day return (volatility)
    df_clean["volatility_target"] = df_clean["next_day_return"].abs()

    feature_cols = [
        "avg_sentiment",
        "news_count",
        "sentiment_std",
        "sentiment_rolling_std_5",
        "sentiment_ma5",
        "news_volume_lag1",
        "avg_confidence"
    ]
    for col in feature_cols:
        if col not in df_clean.columns:
            df_clean[col] = 0.0
        df_clean[col] = df_clean[col].fillna(0.0)

    # Time-based split (80/20)
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    test_df = df_clean.iloc[split_idx:].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["volatility_target"]
    X_test = test_df[feature_cols]
    y_test = test_df["volatility_target"]

    reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    test_df["predicted_volatility"] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": reg.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    date_col = "Date" if "Date" in test_df.columns else "date"
    stock_col = "Symbol" if "Symbol" in test_df.columns else "stock"

    pred_summary = test_df[[date_col, stock_col, "volatility_target", "predicted_volatility"]].rename(
        columns={"volatility_target": "actual_volatility"}
    )

    return {
        "model": reg,
        "mae": round(float(mae), 5),
        "rmse": round(float(rmse), 5),
        "r2_score": round(float(r2), 4),
        "predictions_df": pred_summary,
        "feature_importances": feat_imp
    }

if __name__ == "__main__":
    from src.data_loader import load_processed_data
    df = load_processed_data()
    comp_df = compare_strategy_vs_buy_hold(df)
    print("Risk Metrics Comparison:")
    print(comp_df[["Metric", "AI Strategy", "Buy & Hold"]])
    vol_res = predict_volatility(df)
    print("\nVolatility Prediction MAE:", vol_res["mae"], "R2:", vol_res["r2_score"])
    print(vol_res["feature_importances"].head())
