"""
Prediction Model Module for Sentiment-to-Price Correlation Analyzer.

Trains an XGBoost binary classifier to predict next-day stock price direction using
engineered sentiment features. Evaluates performance against Random and Same-As-Yesterday
baselines using time-based train/test splits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)

from src.utils import setup_logger, save_figure, OUTPUT_REPORTS_DIR

logger = setup_logger("prediction_model")

class SentimentPredictor:
    """Trains and evaluates XGBoost classification model and baselines."""

    FEATURE_COLS = [
        "avg_sentiment",
        "sentiment_lag1",
        "sentiment_lag2",
        "sentiment_lag3",
        "sentiment_ma3",
        "sentiment_ma5",
        "sentiment_rolling_std_5",
        "news_count",
        "news_volume_lag1",
        "avg_confidence"
    ]
    
    TARGET_COL = "target_up"

    def __init__(self, df: pd.DataFrame, train_ratio: float = 0.8):
        """
        Initialize SentimentPredictor.
        
        Args:
            df (pd.DataFrame): Processed feature dataset sorted by Date.
            train_ratio (float): Proportion of data to use for training (default: 0.8).
        """
        self.df = df.copy().sort_values("Date").reset_index(drop=True)
        self.train_ratio = train_ratio
        self.model = None

    def time_based_split(self) -> tuple:
        """
        Performs a temporal train/test split (no shuffling to prevent look-ahead bias).
        Drops unlabeled rows (where target_up is NaN) from train/test evaluation.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, df_test)
        """
        labeled_df = self.df.dropna(subset=[self.TARGET_COL]).reset_index(drop=True)
        split_idx = int(len(labeled_df) * self.train_ratio)
        
        train_df = labeled_df.iloc[:split_idx]
        test_df = labeled_df.iloc[split_idx:]

        X_train = train_df[self.FEATURE_COLS]
        y_train = train_df[self.TARGET_COL]

        X_test = test_df[self.FEATURE_COLS]
        y_test = test_df[self.TARGET_COL]

        logger.info(f"Time-based split created: Train samples = {len(X_train)}, Test samples = {len(X_test)}")
        return X_train, X_test, y_train, y_test, test_df

    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
        """
        Trains the XGBoost Classifier.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.
            
        Returns:
            xgb.XGBClassifier: Fitted XGBoost model.
        """
        logger.info("Training XGBoost Classifier...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
        self.model.fit(X_train, y_train)
        return self.model

    def generate_baselines(self, test_df: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Generates predictions for two baseline models:
        1. Random Baseline (50% random guessing)
        2. Same-As-Yesterday Baseline (predicts tomorrow's return direction = today's return direction)
        
        Args:
            test_df (pd.DataFrame): Test subset DataFrame containing 'current_day_return'.
            y_test (pd.Series): Ground truth target labels.
            
        Returns:
            dict: Baseline predictions for 'random' and 'same_as_yesterday'.
        """
        np.random.seed(42)
        # Random baseline
        y_pred_random = np.random.choice([0, 1], size=len(y_test))

        # Same as yesterday baseline (if today's return > 0, predict tomorrow > 0)
        if "current_day_return" in test_df.columns:
            y_pred_same_yesterday = np.where(test_df["current_day_return"] > 0, 1, 0)
        else:
            y_pred_same_yesterday = np.zeros(len(y_test), dtype=int)

        return {
            "random": y_pred_random,
            "same_as_yesterday": y_pred_same_yesterday
        }

    def evaluate(self) -> tuple:
        """
        Trains XGBoost, evaluates performance against baselines, and generates metrics table & plots.
        
        Returns:
            tuple: (results_df, xgb_preds)
        """
        X_train, X_test, y_train, y_test, test_df = self.time_based_split()
        
        # Train XGBoost
        self.train_xgboost(X_train, y_train)
        xgb_preds = self.model.predict(X_test)

        # Baseline predictions
        baselines = self.generate_baselines(test_df, y_test)
        random_preds = baselines["random"]
        same_yesterday_preds = baselines["same_as_yesterday"]

        # Calculate metrics for all models
        models = {
            "XGBoost Classifier": xgb_preds,
            "Random Baseline (50%)": random_preds,
            "Same-as-Yesterday Baseline": same_yesterday_preds
        }

        results = []
        for name, preds in models.items():
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            
            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4)
            })

        results_df = pd.DataFrame(results)
        
        # Save prediction results table to CSV
        csv_path = OUTPUT_REPORTS_DIR / "prediction_results.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved prediction results table to {csv_path}")

        # Plot Feature Importance and Confusion Matrix
        self.plot_feature_importance(X_train.columns)
        self.plot_confusion_matrix(y_test, xgb_preds)

        return results_df, xgb_preds

    def plot_feature_importance(self, feature_names: list) -> str:
        """Plots and saves XGBoost Feature Importance bar chart."""
        importance = self.model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
        imp_df = imp_df.sort_values("Importance", ascending=False).reset_index(drop=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=imp_df, hue="Feature", palette="viridis", legend=False)
        plt.title("XGBoost Feature Importance (Sentiment & Volume Signals)", fontsize=14, fontweight="bold")
        plt.xlabel("Relative Importance Score", fontsize=11)
        plt.ylabel("Engineered Feature", fontsize=11)
        plt.grid(axis="x", linestyle=":", alpha=0.6)
        
        path = save_figure(plt.gcf(), "feature_importance.png")
        logger.info(f"Saved feature importance plot to {path}")
        return path

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> str:
        """Plots and saves XGBoost Confusion Matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Predicted Down (0)", "Predicted Up (1)"],
                    yticklabels=["Actual Down (0)", "Actual Up (1)"])
        plt.title("XGBoost Classifier Confusion Matrix", fontsize=14, fontweight="bold")
        plt.ylabel("True Class", fontsize=11)
        plt.xlabel("Predicted Class", fontsize=11)
        
        path = save_figure(plt.gcf(), "confusion_matrix.png")
        logger.info(f"Saved confusion matrix plot to {path}")
        return path

if __name__ == "__main__":
    from src.data_collector import DataCollector
    from src.sentiment_analyzer import SentimentAnalyzer
    from src.feature_engineering import FeatureEngineer

    collector = DataCollector(days_history=30)
    prices = collector.fetch_stock_prices()
    news = collector.fetch_stock_news()
    
    analyzer = SentimentAnalyzer()
    news_sent = analyzer.analyze_dataframe(news)
    
    fe = FeatureEngineer(prices, news_sent)
    processed = fe.merge_and_build_features()

    sp = SentimentPredictor(processed)
    res_df, _ = sp.evaluate()
    print(res_df)
