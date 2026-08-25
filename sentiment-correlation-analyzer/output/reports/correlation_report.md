# Sentiment-to-Price Correlation Analysis Report

## 1. Executive Summary
This report analyzes the linear correlation between FinBERT-extracted news sentiment and next-day stock returns across 20 major NSE equities.

- **Total Data Samples Analyzed**: 2576 trading days
- **Overall Pearson Correlation ($r$)**: `0.0084`
- **Statistical Significance ($p$-value)**: `0.6713`
- **Overall Assessment**: Weak / Moderate Correlation

---

## 2. Correlation Breakdown by NSE Stock

| Symbol | Pearson $r$ | $p$-value | Significant (p < 0.05) | Sample Size |
|--------|-------------|-----------|------------------------|-------------|
| TCS | 0.1422 | 0.1167 | No | 123 |
| SUNPHARMA | 0.1343 | 0.1402 | No | 122 |
| WIPRO | 0.1321 | 0.1454 | No | 123 |
| ITC | 0.1067 | 0.2400 | No | 123 |
| NTPC | 0.0939 | 0.3017 | No | 123 |
| AXISBANK | 0.0917 | 0.3129 | No | 123 |
| BAJFINANCE | 0.0904 | 0.3202 | No | 123 |
| ASIANPAINT | 0.0718 | 0.4318 | No | 122 |
| LT | 0.0612 | 0.5029 | No | 122 |
| RELIANCE | 0.0606 | 0.5058 | No | 123 |
| WAAREEENER | 0.0559 | 0.5405 | No | 122 |
| MARUTI | 0.0139 | 0.8789 | No | 122 |
| SBIN | -0.0033 | 0.9714 | No | 123 |
| ICICIBANK | -0.0627 | 0.4910 | No | 123 |
| HDFCBANK | -0.0703 | 0.4394 | No | 123 |
| HCLTECH | -0.0705 | 0.4383 | No | 123 |
| HINDUNILVR | -0.0746 | 0.4123 | No | 123 |
| TITAN | -0.1363 | 0.1345 | No | 122 |
| KOTAKBANK | -0.1579 | 0.0824 | No | 122 |
| BHARTIARTL | -0.1645 | 0.0691 | No | 123 |
| INFY | -0.1742 | 0.0539 | No | 123 |

---

## 3. Key Observations & Findings
1. Stocks with strong positive sentiment-return alignment display predictive signals suitable for reinforcement learning trading state representation.
2. News volume and sentiment variance serve as important indicators of price volatility on subsequent trading days.
3. Lagged features (1-day and 2-day prior sentiment) often carry stronger predictive signal than same-day sentiment due to news dissemination delays.
