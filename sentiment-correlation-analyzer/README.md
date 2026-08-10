# Sentiment-to-Price Correlation Analyzer (NSE Stocks)

An end-to-end Machine Learning pipeline and Streamlit dashboard that quantifies the correlation between financial news sentiment and next-day stock price movements across 20 major National Stock Exchange of India (NSE) equities.

This mini-project serves as the sentiment processing engine for a larger Reinforcement Learning (RL) algorithmic trading system.

---

## 📌 Features

- **Multi-Source Data Ingestion**: Historical daily stock OHLCV prices via `yfinance` (`.NS` suffix) and headline news via `growfin`, Google News RSS, and Yahoo Finance APIs.
- **FinBERT Sentiment Inference**: Uses Hugging Face's `ProsusAI/finbert` (or `kdave/FineTuned_Finbert`) sequence classification model to output positive, neutral, and negative confidence scores and continuous sentiment values $S = P(\text{positive}) - P(\text{negative})$.
- **Temporal Feature Engineering**: Aggregates headline sentiment daily per stock and builds 1-3 day sentiment lags, 3/5-day rolling averages, 5-day sentiment volatility, news volume indicators, and binary price return targets (`target_up`).
- **Statistical Correlation Analysis**: Computes Pearson correlation coefficients ($r$) and two-tailed $p$-values across individual stocks and the combined market universe.
- **Predictive Machine Learning**: Trains an **XGBoost Classifier** using time-based train/test splits (80/20, `shuffle=False`) to avoid look-ahead bias and evaluates performance against two baselines:
  1. **Random Baseline** (50% uniform guessing)
  2. **Same-as-Yesterday Baseline** (predicts tomorrow's return direction = today's return direction)
- **Interactive Streamlit Dashboard**: Web UI to visualize dual-axis price/sentiment overlays, correlation heatmaps, baseline metric comparison cards, feature importances, and live news feeds with FinBERT badges.

---

## 📁 Project Structure

```
sentiment-correlation-analyzer/
├── src/
│   ├── __init__.py            # Package initialization
│   ├── data_collector.py      # Stock price & news fetchers with fallbacks
│   ├── sentiment_analyzer.py  # FinBERT inference engine & batch scoring
│   ├── feature_engineering.py # Daily aggregation, lag features & target labels
│   ├── correlation_analyzer.py # Pearson r, p-values & visualization plots
│   ├── prediction_model.py    # XGBoost classifier & baseline comparisons
│   └── utils.py               # Path configurations, logging & PDF generator
├── notebooks/
│   └── exploration.ipynb      # EDA, prototyping & interactive visualization
├── data/
│   ├── raw/                   # Raw prices & news JSON/CSV
│   └── processed/             # Cleaned & merged feature dataset
├── output/
│   ├── figures/               # Saved charts (correlation, scatter, confusion matrix, etc.)
│   └── reports/               # Markdown/PDF reports & prediction CSV
├── app.py                     # Streamlit interactive web application
├── run_pipeline.py            # Master pipeline execution script
├── requirements.txt           # Python package dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Python 3.9+ installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run End-to-End Pipeline
Execute the master execution script to fetch data, score sentiment, build feature matrices, compute correlations, train XGBoost, evaluate baselines, and save figures & reports:

```bash
python run_pipeline.py
```
*(Or on Windows: `py run_pipeline.py`)*

### 4. Launch Streamlit Dashboard
Launch the interactive web application to explore individual stocks, live news, and model predictions:

```bash
streamlit run app.py
```
*(Or on Windows: `py -m streamlit run app.py`)*

---

## 📊 Expected Output Artifacts

Upon running `python run_pipeline.py`, the following deliverables are saved automatically:

| Deliverable | Location | Description |
|-------------|----------|-------------|
| **Correlation Report** | `output/reports/correlation_report.md` | Statistical breakdown of $r$ and $p$-values per stock |
| **Prediction Results** | `output/reports/prediction_results.csv` | Accuracy, Precision, Recall, F1 score table vs baselines |
| **Correlation Bar Chart** | `output/figures/correlation_by_stock.png` | Per-stock correlation bar chart |
| **Scatter Plot** | `output/figures/sentiment_vs_return_scatter.png` | Daily sentiment vs. next-day return regression scatter |
| **Time Series Overlay** | `output/figures/time_series_overlay.png` | Close price vs sentiment trend overlay |
| **Feature Importance Plot** | `output/figures/feature_importance.png` | Relative importance of sentiment lags & volatility |
| **Confusion Matrix** | `output/figures/confusion_matrix.png` | XGBoost prediction confusion matrix |

---

## 🧠 Stock Universe (NSE Equities)
RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, ITC, HINDUNILVR, SBIN, BHARTIARTL, KOTAKBANK, TATAMOTORS, AXISBANK, LT, WIPRO, HCLTECH, ASIANPAINT, MARUTI, SUNPHARMA, TITAN, NTPC, WAAREEENER.
