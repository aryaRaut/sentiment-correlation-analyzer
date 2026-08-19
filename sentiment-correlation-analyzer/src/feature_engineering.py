"""
Feature Engineering Module for Sentiment-to-Price Correlation Analyzer.

Aggregates headline sentiments daily per stock, aligns with daily market prices,
and constructs temporal lag features, rolling statistics, and binary classification targets.
"""

import pandas as pd
import numpy as np

from src.utils import setup_logger, DATA_PROCESSED_DIR

logger = setup_logger("feature_engineering")

class FeatureEngineer:
    """Constructs analytical features and target variables from sentiment and price data."""

    def __init__(self, price_df: pd.DataFrame, news_sentiment_df: pd.DataFrame):
        """
        Initialize FeatureEngineer.
        
        Args:
            price_df (pd.DataFrame): Raw price DataFrame with ['Date', 'Symbol', 'Close', ...].
            news_sentiment_df (pd.DataFrame): News DataFrame with sentiment metrics.
        """
        self.price_df = price_df.copy()
        self.news_df = news_sentiment_df.copy()
        
        # Ensure Date columns are standard date objects or strings
        self.price_df["Date"] = pd.to_datetime(self.price_df["Date"]).dt.date
        self.news_df["Date"] = pd.to_datetime(self.news_df["Date"]).dt.date

    def aggregate_daily_sentiment(self) -> pd.DataFrame:
        """
        Aggregates news headline sentiment daily per stock.
        
        Calculates:
        - mean sentiment (avg_sentiment)
        - total news count (news_count)
        - sentiment standard deviation (sentiment_std)
        - average confidence (avg_confidence)
        
        Returns:
            pd.DataFrame: Daily aggregated sentiment per stock.
        """
        logger.info("Aggregating news sentiment daily per stock...")
        
        daily_agg = self.news_df.groupby(["Date", "Symbol"]).agg(
            avg_sentiment=("sentiment_score", "mean"),
            news_count=("Headline", "count"),
            sentiment_std=("sentiment_score", lambda x: x.std() if len(x) > 1 else 0.0),
            avg_confidence=("confidence", "mean")
        ).reset_index()

        daily_agg["sentiment_std"] = daily_agg["sentiment_std"].fillna(0.0)
        return daily_agg

    def merge_and_build_features(self) -> pd.DataFrame:
        """
        Merges daily stock price history with aggregated sentiment and constructs lag/rolling features.
        
        Engineered Features:
        - avg_sentiment
        - sentiment_lag1, sentiment_lag2, sentiment_lag3
        - sentiment_ma3, sentiment_ma5
        - sentiment_rolling_std_5
        - news_count, news_volume_lag1
        - avg_confidence
        - next_day_return & target_up (classification target)
        
        Returns:
            pd.DataFrame: Cleaned feature matrix.
        """
        logger.info("Building feature matrix and calculating target variables...")
        daily_sentiment = self.aggregate_daily_sentiment()

        # Sort price dataframe by Symbol and Date
        price_sorted = self.price_df.sort_values(["Symbol", "Date"]).reset_index(drop=True)

        # Left join price data with daily sentiment
        merged = pd.merge(price_sorted, daily_sentiment, on=["Date", "Symbol"], how="left")

        # Fill days with no news with neutral defaults
        merged["avg_sentiment"] = merged["avg_sentiment"].fillna(0.0)
        merged["news_count"] = merged["news_count"].fillna(0).astype(int)
        merged["sentiment_std"] = merged["sentiment_std"].fillna(0.0)
        merged["avg_confidence"] = merged["avg_confidence"].fillna(0.0)

        # Process lag and rolling features per stock
        processed_stocks = []
        for symbol, group in merged.groupby("Symbol"):
            group = group.sort_values("Date").copy()

            # Exponentially decaying sentiment memory (effective_sentiment)
            raw_sent = group["avg_sentiment"].values
            news_cnt = group["news_count"].values
            eff_sent = np.zeros(len(group))
            
            curr_s = 0.0
            for k in range(len(group)):
                if news_cnt[k] > 0:
                    curr_s = raw_sent[k]
                else:
                    curr_s = curr_s * 0.85
                eff_sent[k] = round(curr_s, 4)
                
            group["effective_sentiment"] = eff_sent
            group["avg_sentiment"] = np.where(group["avg_sentiment"] == 0.0, group["effective_sentiment"], group["avg_sentiment"])

            # Lag features based on decaying sentiment
            group["sentiment_lag1"] = group["effective_sentiment"].shift(1).fillna(0.0)
            group["sentiment_lag2"] = group["effective_sentiment"].shift(2).fillna(0.0)
            group["sentiment_lag3"] = group["effective_sentiment"].shift(3).fillna(0.0)

            # Rolling averages
            group["sentiment_ma3"] = group["effective_sentiment"].rolling(window=3, min_periods=1).mean()
            group["sentiment_ma5"] = group["effective_sentiment"].rolling(window=5, min_periods=1).mean()

            # Rolling volatility (std of sentiment over 5 days)
            group["sentiment_rolling_std_5"] = group["effective_sentiment"].rolling(window=5, min_periods=1).std().fillna(0.0)

            # News volume lag and 5-day rolling news volume
            group["news_volume_lag1"] = group["news_count"].shift(1).fillna(0).astype(int)

            # Price returns and Target Variable calculation
            group["current_day_return"] = group["Close"].pct_change().fillna(0.0)
            group["next_day_close"] = group["Close"].shift(-1)
            group["next_day_return"] = (group["next_day_close"] / group["Close"]) - 1.0
            
            # Binary classification target: 1 if next_day_return > 0 else 0 (NaN for latest day)
            group["target_up"] = np.where(group["next_day_return"].isna(), np.nan, np.where(group["next_day_return"] > 0, 1.0, 0.0))

            processed_stocks.append(group)

        final_df = pd.concat(processed_stocks, ignore_index=True)
        
        # Only drop initial rows where lag features are NaN (keep latest day where next_day_return is NaN)
        final_df = final_df.dropna(subset=["sentiment_lag1", "sentiment_lag2", "sentiment_lag3"]).reset_index(drop=True)
        
        processed_file = DATA_PROCESSED_DIR / "processed_dataset.csv"
        final_df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed dataset ({len(final_df)} rows, {len(final_df.columns)} columns) to {processed_file}")
        
        return final_df

if __name__ == "__main__":
    from src.data_collector import DataCollector
    from src.sentiment_analyzer import SentimentAnalyzer
    
    collector = DataCollector(days_history=30)
    prices = collector.fetch_stock_prices()
    news = collector.fetch_stock_news()
    
    analyzer = SentimentAnalyzer()
    news_sent = analyzer.analyze_dataframe(news)
    
    fe = FeatureEngineer(prices, news_sent)
    processed = fe.merge_and_build_features()
    print(processed[["Date", "Symbol", "Close", "avg_sentiment", "next_day_return", "target_up"]].head())
