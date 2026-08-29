"""
Sentiment Analyzer Module for Sentiment-to-Price Correlation Analyzer.

Uses Hugging Face ProsusAI/finbert (or fine-tuned FinBERT) model for financial headline
sentiment analysis, producing positive, neutral, negative probabilities, confidence scores,
and continuous sentiment metrics.
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.utils import setup_logger

logger = setup_logger("sentiment_analyzer")

class SentimentAnalyzer:
    """Performs financial sentiment analysis on news headlines using FinBERT."""

    def __init__(self, model_name: str = "ProsusAI/finbert", batch_size: int = 32):
        """
        Initialize the FinBERT Sentiment Analyzer.
        
        Args:
            model_name (str): HuggingFace model path (default: 'ProsusAI/finbert').
            batch_size (int): Batch size for inference.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
        self._load_model()

    def _load_model(self):
        """Loads the FinBERT model and tokenizer from HuggingFace, with fallback handling."""
        logger.info(f"Loading FinBERT sentiment model '{self.model_name}' (device: {'GPU' if self.device == 0 else 'CPU'})...")
        try:
            # First attempt: load requested model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                top_k=None,
                device=self.device,
                truncation=True,
                max_length=512
            )
            logger.info("FinBERT model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load '{self.model_name}' ({e}). Attempting fallback model 'kdave/FineTuned_Finbert'...")
            try:
                fallback_name = "kdave/FineTuned_Finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(fallback_name)
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    top_k=None,
                    device=self.device,
                    truncation=True,
                    max_length=512
                )
                logger.info("Fallback FinBERT model loaded successfully.")
            except Exception as e2:
                logger.error(f"Failed to load FinBERT models ({e2}). Using rule-based financial sentiment analyzer fallback.")
                self.pipeline = None

    def analyze_texts(self, headlines: list) -> list:
        """
        Analyzes a list of news headlines and returns sentiment metrics.
        
        Args:
            headlines (list): List of headline strings.
            
        Returns:
            list: List of dicts containing ['sentiment_label', 'confidence', 'pos_prob', 'neu_prob', 'neg_prob', 'sentiment_score'].
        """
        results = []
        if not headlines:
            return results

        # Clean headlines
        cleaned_headlines = [str(h).strip() if pd.notna(h) and str(h).strip() else "Neutral market update" for h in headlines]

        if self.pipeline is not None:
            # FinBERT HuggingFace inference in batches
            for i in range(0, len(cleaned_headlines), self.batch_size):
                batch = cleaned_headlines[i:i + self.batch_size]
                try:
                    outputs = self.pipeline(batch)
                    for text_outputs in outputs:
                        if isinstance(text_outputs, dict):
                            text_outputs = [text_outputs]
                        # Extract label scores
                        score_dict = {item["label"].lower(): float(item["score"]) for item in text_outputs if isinstance(item, dict) and "label" in item}
                        pos_prob = score_dict.get("positive", 0.0)
                        neu_prob = score_dict.get("neutral", 0.0)
                        neg_prob = score_dict.get("negative", 0.0)
                        
                        # Primary label & top confidence
                        max_label = max(score_dict, key=score_dict.get)
                        confidence = score_dict[max_label]
                        
                        # Continuous sentiment score formula:
                        # positive -> +confidence, negative -> -confidence, neutral -> 0
                        # Continuous composite: pos_prob - neg_prob
                        sentiment_score = pos_prob - neg_prob
                        
                        results.append({
                            "sentiment_label": max_label,
                            "confidence": round(confidence, 4),
                            "pos_prob": round(pos_prob, 4),
                            "neu_prob": round(neu_prob, 4),
                            "neg_prob": round(neg_prob, 4),
                            "sentiment_score": round(sentiment_score, 4)
                        })
                except Exception as e:
                    logger.error(f"Inference error in batch {i}: {e}. Fallback to rule-based for this batch.")
                    for text in batch:
                        results.append(self._rule_based_sentiment(text))
        else:
            # Rule-based financial sentiment analyzer fallback
            for text in cleaned_headlines:
                results.append(self._rule_based_sentiment(text))

        return results

    def analyze_dataframe(self, df: pd.DataFrame, headline_col: str = "Headline") -> pd.DataFrame:
        """
        Analyzes headlines in a DataFrame and appends sentiment columns.
        
        Args:
            df (pd.DataFrame): DataFrame containing news headlines.
            headline_col (str): Column name containing headline text.
            
        Returns:
            pd.DataFrame: DataFrame augmented with sentiment analysis columns.
        """
        logger.info(f"Running sentiment analysis on {len(df)} headlines using FinBERT...")
        headlines = df[headline_col].tolist()
        
        sentiment_records = []
        # Use tqdm for progress tracking
        chunk_size = 100
        for i in tqdm(range(0, len(headlines), chunk_size), desc="Analyzing Sentiment"):
            chunk = headlines[i:i + chunk_size]
            chunk_results = self.analyze_texts(chunk)
            sentiment_records.extend(chunk_results)
            
        sentiment_df = pd.DataFrame(sentiment_records)
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
        return result_df

    def _rule_based_sentiment(self, text: str) -> dict:
        """Fallback rule-based sentiment classifier for offline or hardware-restricted environments."""
        text_lower = text.lower()
        pos_words = ["profit", "record", "growth", "upgrade", "gain", "rise", "positive", "surge", "expansion", "partnership", "success", "buy", "rally"]
        neg_words = ["drop", "loss", "fall", "downgrade", "decline", "investigation", "pressure", "probe", "warn", "debt", "slash", "cut", "risk", "slump"]
        
        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)
        
        if pos_count > neg_count:
            label = "positive"
            conf = min(0.6 + 0.1 * pos_count, 0.95)
            pos_p, neg_p, neu_p = conf, 0.05, 1.0 - conf - 0.05
        elif neg_count > pos_count:
            label = "negative"
            conf = min(0.6 + 0.1 * neg_count, 0.95)
            neg_p, pos_p, neu_p = conf, 0.05, 1.0 - conf - 0.05
        else:
            label = "neutral"
            conf = 0.8
            pos_p, neg_p, neu_p = 0.1, 0.1, 0.8
            
        score = pos_p - neg_p
        return {
            "sentiment_label": label,
            "confidence": round(conf, 4),
            "pos_prob": round(pos_p, 4),
            "neu_prob": round(neu_p, 4),
            "neg_prob": round(neg_p, 4),
            "sentiment_score": round(score, 4)
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample_headlines = [
        "RELIANCE reports quarterly profit surge of 25%, beating estimates",
        "TCS faces margin pressures as tech spending slows globally",
        "INFY board schedules ordinary general meeting for next week"
    ]
    res = analyzer.analyze_texts(sample_headlines)
    for h, r in zip(sample_headlines, res):
        print(f"Headline: {h}\nResult: {r}\n")
