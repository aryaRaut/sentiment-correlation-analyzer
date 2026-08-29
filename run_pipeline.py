"""
Master Pipeline Execution Script for Sentiment-to-Price Correlation Analyzer.

Orchestrates data collection, FinBERT sentiment scoring, daily aggregation,
feature engineering, correlation analysis, XGBoost model evaluation, visualization generation,
and report exporting.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure current working directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logger, OUTPUT_FIGURES_DIR, OUTPUT_REPORTS_DIR, DATA_PROCESSED_DIR
from src.data_collector import DataCollector
from src.sentiment_analyzer import SentimentAnalyzer
from src.feature_engineering import FeatureEngineer
from src.correlation_analyzer import CorrelationAnalyzer
from src.prediction_model import SentimentPredictor

logger = setup_logger("run_pipeline")


def clean_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the processed DataFrame by removing rows with NaN values
    in critical columns and ensuring proper data types.
    """
    df = df.copy()
    
    # --- FIX 1: Drop rows with NaN next_day_return ---
    # This is the root cause of all issues. If next_day_return is NaN,
    # actual_direction will be NaN, and everything breaks.
    if "next_day_return" in df.columns:
        before = len(df)
        df = df.dropna(subset=["next_day_return"])
        after = len(df)
        if before != after:
            logger.info(f"-> Removed {before - after} rows with NaN next_day_return")
    
    # --- FIX 2: Ensure actual_direction is properly calculated ---
    if "next_day_return" in df.columns:
        df["actual_direction"] = df["next_day_return"].apply(lambda x: 1 if x > 0 else 0)
        logger.info(f"-> Recalculated actual_direction from next_day_return")
    
    # --- FIX 3: Ensure proper data types ---
    if "actual_direction" in df.columns:
        df["actual_direction"] = df["actual_direction"].astype(int)
    
    # --- FIX 4: Drop any remaining NaN rows in critical columns ---
    critical_cols = ["actual_direction", "next_day_return"]
    if "sentiment_score" in df.columns:
        critical_cols.append("sentiment_score")
    
    before = len(df)
    df = df.dropna(subset=critical_cols)
    after = len(df)
    if before != after:
        logger.info(f"-> Removed {before - after} rows with NaN in critical columns")
    
    return df


def main():
    """Executes the full Sentiment-to-Price Correlation Analyzer pipeline."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("STARTING SENTIMENT-TO-PRICE CORRELATION ANALYZER PIPELINE")
    logger.info("=" * 70)

    # Step 1 & 2: Fetch Stock Prices and News Data
    logger.info("\n[STEP 1/6] Data Collection (NSE Prices + Stock News)...")
    collector = DataCollector(days_history=180)
    price_df = collector.fetch_stock_prices()
    news_df = collector.fetch_stock_news()
    logger.info(f"-> Fetched {len(price_df)} daily price records across {price_df['Symbol'].nunique()} stocks.")
    logger.info(f"-> Fetched {len(news_df)} headline articles.")

    # Step 3: FinBERT Sentiment Inference
    logger.info("\n[STEP 2/6] Running FinBERT Sentiment Inference...")
    analyzer = SentimentAnalyzer(model_name="ProsusAI/finbert")
    news_sentiment_df = analyzer.analyze_dataframe(news_df)
    logger.info(f"-> Sentiment scoring complete. Sample predictions:")
    print(news_sentiment_df[['Date', 'Symbol', 'Headline', 'sentiment_label', 'sentiment_score']].head(3).to_string())

    # Step 4 & 5: Daily Aggregation and Feature Engineering
    logger.info("\n[STEP 3/6] Aggregating Daily Sentiment & Building Feature Matrix...")
    fe = FeatureEngineer(price_df, news_sentiment_df)
    processed_df = fe.merge_and_build_features()
    logger.info(f"-> Feature matrix built with shape {processed_df.shape}.")

    # --- NEW: Clean the data BEFORE correlation and modeling ---
    logger.info("\n[STEP 3.5/6] Cleaning Processed Data (Removing NaN Returns)...")
    processed_df = clean_processed_data(processed_df)
    logger.info(f"-> Cleaned feature matrix shape: {processed_df.shape}")

    # Step 6 & 7: Correlation Analysis & Visualization
    logger.info("\n[STEP 4/6] Computing Pearson Correlations & Generating Visualizations...")
    ca = CorrelationAnalyzer(processed_df)
    report_md_path = ca.generate_report()
    overall_stats, stock_corrs = ca.compute_correlations()
    logger.info(f"-> Overall Pearson Correlation r = {overall_stats['r']:.4f} (p = {overall_stats['p_value']:.4f})")
    logger.info(f"-> Correlation Report generated at: {report_md_path}")

    # Step 8, 9 & 10: Model Training, Evaluation, and Baselines Comparison
    logger.info("\n[STEP 5/6] Training XGBoost Classifier & Evaluating Against Baselines...")
    sp = SentimentPredictor(processed_df)
    results_df, model = sp.evaluate()
    logger.info("-> Model Comparison Results:")
    print(results_df.to_string(index=False))

    # --- NEW: Generate predictions and save them to the processed DataFrame ---
    logger.info("\n[STEP 5.5/6] Generating Predictions for All Historical Data...")
    try:
        # Get the feature columns used by the model
        feature_cols = sp.FEATURE_COLS
        
        # Ensure all required features are present
        missing_feats = [col for col in feature_cols if col not in processed_df.columns]
        for mf in missing_feats:
            processed_df[mf] = 0.0
        
        # Prepare features
        X = processed_df[feature_cols].fillna(0.0)
        
        # Generate predictions
        processed_df["predicted_direction"] = model.predict(X)
        logger.info(f"-> Generated predictions for {len(processed_df)} rows")
        
        # Ensure predictions are integers
        processed_df["predicted_direction"] = processed_df["predicted_direction"].astype(int)
        
    except Exception as e:
        logger.warning(f"-> Could not generate predictions: {e}")
        logger.warning("-> Predictions will be generated in the dashboard instead.")

    # --- NEW: Final cleanup before saving ---
    logger.info("\n[STEP 5.6/6] Final Data Cleanup...")
    processed_df = processed_df.dropna(subset=["actual_direction", "predicted_direction", "next_day_return"])
    
    # Ensure correct data types
    for col in ["actual_direction", "predicted_direction"]:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].astype(int)

    # --- NEW: Save the cleaned, prediction-enhanced dataset ---
    processed_path = DATA_PROCESSED_DIR / "processed_dataset.csv"
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    processed_df.to_csv(processed_path, index=False)
    logger.info(f"-> Saved cleaned dataset with predictions to: {processed_path}")
    logger.info(f"-> Final dataset shape: {processed_df.shape}")
    logger.info(f"-> Columns: {list(processed_df.columns)}")

    # Step 11: Summary & Launch Instructions
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    logger.info("=" * 70)
    logger.info(f"Output Figures saved to: {OUTPUT_FIGURES_DIR}")
    logger.info(f"Output Reports saved to: {OUTPUT_REPORTS_DIR}")
    logger.info(f"Processed Data saved to: {processed_path}")
    logger.info("\nTo launch the interactive Streamlit Dashboard, run:")
    logger.info("  py -m streamlit run app.py")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()