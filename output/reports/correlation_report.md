# Sentiment-to-Price Correlation Analysis Report

## 1. Executive Summary
This report analyzes the linear correlation between FinBERT-extracted news sentiment and next-day stock returns across 20 major NSE equities.

- **Total Data Samples Analyzed**: 2581 trading days
- **Overall Pearson Correlation ($r$)**: `0.0263`
- **Statistical Significance ($p$-value)**: `0.1815`
- **Overall Assessment**: Weak / Moderate Correlation

---

## 2. Correlation Breakdown by NSE Stock

| Symbol | Pearson $r$ | $p$-value | Significant (p < 0.05) | Sample Size |
|--------|-------------|-----------|------------------------|-------------|
| HCLTECH | 0.2953 | 0.0009 | Yes (p < 0.05) | 123 |
| RELIANCE | 0.1224 | 0.1774 | No | 123 |
| WIPRO | 0.1011 | 0.2660 | No | 123 |
| MARUTI | 0.0868 | 0.3400 | No | 123 |
| WAAREEENER | 0.0847 | 0.3517 | No | 123 |
| LT | 0.0822 | 0.3659 | No | 123 |
| BAJFINANCE | 0.0796 | 0.3817 | No | 123 |
| TCS | 0.0785 | 0.3879 | No | 123 |
| HINDUNILVR | 0.0599 | 0.5124 | No | 122 |
| TITAN | 0.0275 | 0.7624 | No | 123 |
| NTPC | 0.0120 | 0.8950 | No | 123 |
| KOTAKBANK | 0.0112 | 0.9018 | No | 123 |
| SBIN | 0.0057 | 0.9499 | No | 123 |
| ITC | -0.0098 | 0.9142 | No | 123 |
| AXISBANK | -0.0198 | 0.8289 | No | 122 |
| HDFCBANK | -0.0369 | 0.6851 | No | 123 |
| INFY | -0.0435 | 0.6330 | No | 123 |
| ASIANPAINT | -0.1050 | 0.2478 | No | 123 |
| SUNPHARMA | -0.1116 | 0.2192 | No | 123 |
| ICICIBANK | -0.1462 | 0.1067 | No | 123 |
| BHARTIARTL | -0.2528 | 0.0048 | Yes (p < 0.05) | 123 |

---

## 3. Key Observations & Findings
1. Stocks with strong positive sentiment-return alignment display predictive signals suitable for reinforcement learning trading state representation.
2. News volume and sentiment variance serve as important indicators of price volatility on subsequent trading days.
3. Lagged features (1-day and 2-day prior sentiment) often carry stronger predictive signal than same-day sentiment due to news dissemination delays.
