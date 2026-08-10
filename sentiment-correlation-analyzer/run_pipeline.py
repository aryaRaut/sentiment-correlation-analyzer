"""
Master Pipeline Execution Script for Sentiment-to-Price Correlation Analyzer.

Orchestrates data collection, FinBERT sentiment scoring, daily aggregation,
feature engineering, correlation analysis, XGBoost model evaluation, visualization generation,
and report exporting.
"""

import os
import sys
import time

# Ensure current working directory is on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logger, OUTPUT_FIGURES_DIR, OUTPUT_REPORTS_DIR
from src.data_collector import DataCollector
from src.sentiment_analyzer import SentimentAnalyzer
from src.feature_engineering import FeatureEngineer
from src.correlation_analyzer import CorrelationAnalyzer
from src.prediction_model import SentimentPredictor

logger = setup_logger("run_pipeline")

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
    results_df, _ = sp.evaluate()
    logger.info("-> Model Comparison Results:")
    print(results_df.to_string(index=False))

    # Step 11: Summary & Launch Instructions
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    logger.info("=" * 70)
    logger.info(f"Output Figures saved to: {OUTPUT_FIGURES_DIR}")
    logger.info(f"Output Reports saved to: {OUTPUT_REPORTS_DIR}")
    logger.info("\nTo launch the interactive Streamlit Dashboard, run:")
    logger.info("  py -m streamlit run app.py")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
