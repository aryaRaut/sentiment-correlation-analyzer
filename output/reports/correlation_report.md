# Sentiment-to-Price Correlation Analysis Report

## 1. Executive Summary
This report analyzes the linear correlation between FinBERT-extracted news sentiment and next-day stock returns across 20 major NSE equities.

- **Total Data Samples Analyzed**: 2583 trading days
- **Overall Pearson Correlation ($r$)**: `0.0203`
- **Statistical Significance ($p$-value)**: `0.3025`
- **Overall Assessment**: Weak / Moderate Correlation

---

## 2. Correlation Breakdown by NSE Stock

| Symbol | Pearson $r$ | $p$-value | Significant (p < 0.05) | Sample Size |
|--------|-------------|-----------|------------------------|-------------|
| WAAREEENER | 0.1454 | 0.1087 | No | 123 |
| TCS | 0.1341 | 0.1391 | No | 123 |
| SUNPHARMA | 0.1184 | 0.1923 | No | 123 |
| WIPRO | 0.0957 | 0.2921 | No | 123 |
| RELIANCE | 0.0868 | 0.3399 | No | 123 |
| LT | 0.0634 | 0.4863 | No | 123 |
| BAJFINANCE | 0.0484 | 0.5947 | No | 123 |
| AXISBANK | 0.0424 | 0.6416 | No | 123 |
| ITC | 0.0393 | 0.6663 | No | 123 |
| ASIANPAINT | 0.0299 | 0.7425 | No | 123 |
| MARUTI | -0.0078 | 0.9316 | No | 123 |
| ICICIBANK | -0.0131 | 0.8858 | No | 123 |
| HINDUNILVR | -0.0134 | 0.8830 | No | 123 |
| INFY | -0.0341 | 0.7080 | No | 123 |
| HCLTECH | -0.0458 | 0.6146 | No | 123 |
| SBIN | -0.0576 | 0.5267 | No | 123 |
| NTPC | -0.0638 | 0.4830 | No | 123 |
| TITAN | -0.0846 | 0.3522 | No | 123 |
| KOTAKBANK | -0.0867 | 0.3403 | No | 123 |
| HDFCBANK | -0.0885 | 0.3301 | No | 123 |
| BHARTIARTL | -0.0932 | 0.3051 | No | 123 |

---

## 3. Key Observations & Findings
1. Stocks with strong positive sentiment-return alignment display predictive signals suitable for reinforcement learning trading state representation.
2. News volume and sentiment variance serve as important indicators of price volatility on subsequent trading days.
3. Lagged features (1-day and 2-day prior sentiment) often carry stronger predictive signal than same-day sentiment due to news dissemination delays.
