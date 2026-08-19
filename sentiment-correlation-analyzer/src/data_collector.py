"""
Data Collector Module for Sentiment-to-Price Correlation Analyzer.

Fetches stock prices via yfinance (.NS suffix) and news headlines via growfin,
Google News RSS feeds, or yfinance news, with automatic fallbacks and error handling.
"""

import os
import time
import datetime
import random
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
from tqdm import tqdm

from src.utils import (
    setup_logger, 
    DATA_RAW_DIR, 
    NSE_STOCKS, 
    get_symbol_with_suffix
)

logger = setup_logger("data_collector")

class DataCollector:
    """Handles fetching and caching of stock prices and news data."""

    def __init__(self, stocks=None, days_history=180):
        """
        Initialize the DataCollector.
        
        Args:
            stocks (list): List of NSE stock tickers without suffix (e.g., ['RELIANCE', 'TCS']).
            days_history (int): Historical period in days (default 180 ~ 6 months).
        """
        self.stocks = stocks or NSE_STOCKS
        self.days_history = days_history
        self.end_date = datetime.date.today()
        self.start_date = self.end_date - datetime.timedelta(days=self.days_history)

    def fetch_stock_prices(self) -> pd.DataFrame:
        """
        Fetches daily historical OHLCV data for all stocks using yfinance.
        
        Returns:
            pd.DataFrame: Long-format DataFrame containing ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume'].
        """
        logger.info(f"Fetching stock price data for {len(self.stocks)} NSE stocks from {self.start_date} to {self.end_date}...")
        all_prices = []

        fetch_start = self.start_date.strftime("%Y-%m-%d")
        fetch_end = (self.end_date + datetime.timedelta(days=2)).strftime("%Y-%m-%d")
        
        for symbol in tqdm(self.stocks, desc="Fetching Stock Prices"):
            yf_symbol = get_symbol_with_suffix(symbol)
            try:
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(start=fetch_start, end=fetch_end)
                if df.empty:
                    logger.warning(f"No price data retrieved for {yf_symbol}. Attempting download fallback...")
                    df = yf.download(yf_symbol, start=fetch_start, end=fetch_end, progress=False)

                if not df.empty:
                    df = df.reset_index()
                    # Handle MultiIndex columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [col[0] for col in df.columns]
                    
                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.date
                    df["Symbol"] = symbol
                    df = df[["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]]
                    all_prices.append(df)
                else:
                    logger.warning(f"Failed to fetch price data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching prices for {symbol}: {e}")

        if all_prices:
            price_df = pd.concat(all_prices, ignore_index=True)
        else:
            logger.warning("No live prices retrieved. Generating realistic synthetic price data fallback...")
            price_df = self._generate_synthetic_prices()

        raw_price_file = DATA_RAW_DIR / "raw_prices.csv"
        price_df.to_csv(raw_price_file, index=False)
        logger.info(f"Saved raw stock prices ({len(price_df)} rows) to {raw_price_file}")
        return price_df

    def fetch_stock_news(self) -> pd.DataFrame:
        """
        Fetches news headlines for each stock.
        Tries growfin library first; falls back to Google News RSS and yfinance news API.
        
        Returns:
            pd.DataFrame: DataFrame containing ['Date', 'Symbol', 'Headline', 'Source', 'URL'].
        """
        logger.info(f"Fetching news data for {len(self.stocks)} stocks...")
        all_news = []

        # Tier 1: Try growfin library if installed
        growfin_available = False
        try:
            import growfin
            growfin_available = True
            logger.info("growfin package detected. Attempting news collection via growfin...")
        except ImportError:
            logger.info("growfin library not found. Using RSS & yfinance news fallback.")

        for symbol in tqdm(self.stocks, desc="Fetching Stock News"):
            news_items = []
            
            if growfin_available:
                try:
                    gf_data = growfin.get_news(symbol) if hasattr(growfin, "get_news") else []
                    for item in gf_data:
                        news_items.append({
                            "Date": pd.to_datetime(item.get("date", datetime.date.today())).date(),
                            "Symbol": symbol,
                            "Headline": item.get("headline") or item.get("title"),
                            "Source": item.get("source", "Growfin"),
                            "URL": item.get("url", "")
                        })
                except Exception as e:
                    logger.debug(f"Growfin fetch error for {symbol}: {e}")

            # Tier 2: Google News RSS fallback if needed
            if not news_items:
                rss_items = self._fetch_rss_news(symbol)
                news_items.extend(rss_items)

            # Tier 3: yfinance news API fallback
            if not news_items:
                yf_items = self._fetch_yf_news(symbol)
                news_items.extend(yf_items)

            if news_items:
                all_news.extend(news_items)

        if all_news:
            news_df = pd.DataFrame(all_news)
            # Remove duplicate headlines per symbol
            news_df = news_df.drop_duplicates(subset=["Symbol", "Headline"]).reset_index(drop=True)
        else:
            logger.warning("No live news fetched (rate limit or offline). Generating synthetic news dataset fallback...")
            news_df = self._generate_synthetic_news()

        raw_news_file = DATA_RAW_DIR / "raw_news.csv"
        news_df.to_csv(raw_news_file, index=False)
        logger.info(f"Saved raw news headlines ({len(news_df)} rows) to {raw_news_file}")
        return news_df

    def _fetch_rss_news(self, symbol: str) -> list:
        """Fetches news headlines from Google News RSS feed for an NSE stock."""
        items = []
        queries = [f"{symbol} NSE India stock news", f"{symbol} share price", f"{symbol} quarterly results"]
        seen_titles = set()
        
        for q in queries:
            try:
                url = f"https://news.google.com/rss/search?q={q.replace(' ', '%20')}&hl=en-IN&gl=IN&ceid=IN:en"
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:20]:
                    title = entry.get('title', '').strip()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        pub_date = datetime.date.today()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            pub_date = datetime.date(*entry.published_parsed[:3])
                        
                        items.append({
                            "Date": pub_date,
                            "Symbol": symbol,
                            "Headline": title,
                            "Source": entry.get('source', {}).get('title', 'Google News'),
                            "URL": entry.get('link', '')
                        })
            except Exception as e:
                logger.debug(f"RSS fetch error for {symbol} query '{q}': {e}")
                
        return items

    def _fetch_yf_news(self, symbol: str) -> list:
        """Fetches news headlines using yfinance Ticker news attribute."""
        items = []
        try:
            yf_symbol = get_symbol_with_suffix(symbol)
            ticker = yf.Ticker(yf_symbol)
            news = ticker.news
            for item in news or []:
                title = item.get("title") or item.get("headline")
                if title:
                    pub_time = item.get("providerPublishTime")
                    pub_date = datetime.date.fromtimestamp(pub_time) if pub_time else datetime.date.today()
                    items.append({
                        "Date": pub_date,
                        "Symbol": symbol,
                        "Headline": title,
                        "Source": item.get("publisher", "Yahoo Finance"),
                        "URL": item.get("link", "")
                    })
        except Exception as e:
            logger.debug(f"yfinance news fetch error for {symbol}: {e}")
        return items

    def _generate_synthetic_prices(self) -> pd.DataFrame:
        """Generates realistic synthetic daily price series for fallback testing."""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq="B").date
        data = []
        
        base_prices = {
            "RELIANCE": 2800, "TCS": 3800, "INFY": 1500, "HDFCBANK": 1600, "ICICIBANK": 1050,
            "ITC": 450, "HINDUNILVR": 2500, "SBIN": 750, "BHARTIARTL": 1100, "KOTAKBANK": 1800,
            "BAJFINANCE": 6800, "AXISBANK": 1050, "LT": 3400, "WIPRO": 480, "HCLTECH": 1400,
            "ASIANPAINT": 2900, "MARUTI": 10500, "SUNPHARMA": 1500, "TITAN": 3600, "NTPC": 320,
            "WAAREEENER": 3000
        }
        
        np.random.seed(42)
        for symbol in self.stocks:
            price = base_prices.get(symbol, 1000)
            for d in dates:
                ret = np.random.normal(0.0005, 0.015)
                price = max(10, price * (1 + ret))
                high = price * (1 + abs(np.random.normal(0, 0.005)))
                low = price * (1 - abs(np.random.normal(0, 0.005)))
                open_p = price * (1 + np.random.normal(0, 0.003))
                volume = int(np.random.lognormal(14, 0.8))
                
                data.append({
                    "Date": d,
                    "Symbol": symbol,
                    "Open": round(open_p, 2),
                    "High": round(high, 2),
                    "Low": round(low, 2),
                    "Close": round(price, 2),
                    "Volume": volume
                })
                
        return pd.DataFrame(data)

    def _generate_synthetic_news(self) -> pd.DataFrame:
        """Generates realistic financial news headlines for fallback testing."""
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq="B").date
        templates_pos = [
            "{stock} reports record quarterly profit, exceeding analyst estimates",
            "{stock} secures $500M international contract expansion",
            "Analysts upgrade {stock} target price following robust revenue growth",
            "{stock} launches innovative AI-driven service line in India",
            "{stock} announces strategic partnership to boost market share"
        ]
        templates_neg = [
            "{stock} faces margin pressure amid rising operational expenses",
            "Regulatory scrutiny increases for {stock} over compliance issue",
            "Quarterly revenue drops for {stock} as demand slows down",
            "{stock} downgraded by major brokerage following weak guidance",
            "Supply chain disruptions impact production targets for {stock}"
        ]
        templates_neu = [
            "{stock} schedules annual general meeting for upcoming month",
            "Board of directors at {stock} considers new dividend payout plan",
            "{stock} files quarterly compliance report with NSE",
            "Market highlights: {stock} trades rangebound in quiet session",
            "Executive leadership changes announced at {stock}"
        ]
        
        data = []
        random.seed(42)
        for symbol in self.stocks:
            for d in dates:
                # Generate 1 to 4 news articles per stock per day
                num_articles = random.randint(1, 4)
                for _ in range(num_articles):
                    sentiment_choice = random.choices(["pos", "neg", "neu"], weights=[0.4, 0.3, 0.3])[0]
                    if sentiment_choice == "pos":
                        headline = random.choice(templates_pos).format(stock=symbol)
                    elif sentiment_choice == "neg":
                        headline = random.choice(templates_neg).format(stock=symbol)
                    else:
                        headline = random.choice(templates_neu).format(stock=symbol)
                        
                    data.append({
                        "Date": d,
                        "Symbol": symbol,
                        "Headline": headline,
                        "Source": random.choice(["Financial Express", "Economic Times", "Livemint", "Business Standard"]),
                        "URL": f"https://finance.example.com/news/{symbol.lower()}"
                    })
                    
        return pd.DataFrame(data)

if __name__ == "__main__":
    collector = DataCollector(days_history=60)
    prices = collector.fetch_stock_prices()
    news = collector.fetch_stock_news()
    print(f"Prices fetched: {len(prices)}, News fetched: {len(news)}")
