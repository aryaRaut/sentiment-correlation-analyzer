"""
Data Loader Module for Sentiment-to-Price Correlation Analyzer.

Provides cached data loading utilities for Streamlit application components and
handles automatic model inference when predicted_direction is missing.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from src.utils import DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_REPORTS_DIR, PROJECT_ROOT
from src.prediction_model import SentimentPredictor

# Sector mapping for 20 NSE stocks
SECTOR_MAP = {
    'TCS': 'IT', 
    'INFY': 'IT', 
    'WIPRO': 'IT', 
    'HCLTECH': 'IT',
    'HDFCBANK': 'Banking', 
    'ICICIBANK': 'Banking', 
    'SBIN': 'Banking', 
    'AXISBANK': 'Banking', 
    'KOTAKBANK': 'Banking', 
    'BAJFINANCE': 'Banking',
    'SUNPHARMA': 'Pharma',
    'MARUTI': 'Auto',
    'RELIANCE': 'Energy', 
    'NTPC': 'Energy', 
    'WAAREEENER': 'Energy',
    'HINDUNILVR': 'FMCG', 
    'ITC': 'FMCG', 
    'ASIANPAINT': 'FMCG', 
    'TITAN': 'FMCG',
    'LT': 'Industrials/Telecom', 
    'BHARTIARTL': 'Industrials/Telecom'
}

MODEL_SAVE_PATH = OUTPUT_REPORTS_DIR / "xgboost_model.joblib"

def ensure_model_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if 'predicted_direction' exists in df. If missing, loads saved XGBoost model
    or trains a new model, runs inference on historical data, and populates 'predicted_direction'.
    """
    df = df.copy()
    
    # Check column existence
    if "predicted_direction" in df.columns and not df["predicted_direction"].isnull().all():
        return df

    # Standardize target & stock columns for modeling
    if "actual_direction" not in df.columns and "target_up" in df.columns:
        df["actual_direction"] = df["target_up"]
    if "stock" not in df.columns and "Symbol" in df.columns:
        df["stock"] = df["Symbol"]

    feature_cols = SentimentPredictor.FEATURE_COLS
    
    # Ensure all required features are present
    missing_feats = [col for col in feature_cols if col not in df.columns]
    for mf in missing_feats:
        df[mf] = 0.0

    X = df[feature_cols].fillna(0.0)

    model = None
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model = joblib.load(MODEL_SAVE_PATH)
        except Exception:
            model = None

    if model is None:
        # Train model using SentimentPredictor
        sp = SentimentPredictor(df)
        X_train, X_test, y_train, y_test, _ = sp.time_based_split()
        model = sp.train_xgboost(X_train, y_train)
        os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
        joblib.dump(model, MODEL_SAVE_PATH)

    # Predict direction
    df["predicted_direction"] = model.predict(X)
    return df

@st.cache_data(ttl=3600)
def load_processed_data() -> pd.DataFrame:
    """
    Loads and standardizes processed stock & sentiment dataset.
    Automatically generates predictions if missing.
    """
    processed_path = DATA_PROCESSED_DIR / "processed_dataset.csv"
    if not os.path.exists(processed_path):
        from src.data_collector import DataCollector
        from src.sentiment_analyzer import SentimentAnalyzer
        from src.feature_engineering import FeatureEngineer
        
        collector = DataCollector(days_history=60)
        prices = collector.fetch_stock_prices()
        news = collector.fetch_stock_news()
        
        analyzer = SentimentAnalyzer()
        news_df = analyzer.analyze_dataframe(news)
        
        fe = FeatureEngineer(prices, news_df)
        df = fe.merge_and_build_features()
    else:
        df = pd.read_csv(processed_path)

    # Standardize column names
    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"]).dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date

    if "Symbol" in df.columns:
        df["stock"] = df["Symbol"]
    elif "stock" in df.columns:
        df["Symbol"] = df["stock"]

    if "target_up" in df.columns:
        df["actual_direction"] = df["target_up"]
    elif "actual_direction" in df.columns:
        df["target_up"] = df["actual_direction"]

    if "avg_sentiment" in df.columns:
        df["sentiment_score"] = df["avg_sentiment"]
    elif "sentiment_score" in df.columns:
        df["avg_sentiment"] = df["sentiment_score"]

    if "Sector" not in df.columns:
        df["Sector"] = df["stock"].map(SECTOR_MAP).fillna("Other")

    # Handle missing predictions requirement
    df = ensure_model_predictions(df)

    return df

@st.cache_data(ttl=3600)
def load_news_data() -> pd.DataFrame:
    """Loads and standardizes raw financial news dataset."""
    news_path = DATA_RAW_DIR / "raw_news.csv"
    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path)
        if "Date" in news_df.columns:
            news_df["date"] = pd.to_datetime(news_df["Date"]).dt.date
        elif "date" in news_df.columns:
            news_df["date"] = pd.to_datetime(news_df["date"]).dt.date

        if "Symbol" in news_df.columns:
            news_df["stock"] = news_df["Symbol"]
        elif "stock" in news_df.columns:
            news_df["Symbol"] = news_df["stock"]
        return news_df
    return pd.DataFrame()

def get_sector_mapping() -> dict:
    """Returns stock to sector mapping dictionary."""
    return SECTOR_MAP
