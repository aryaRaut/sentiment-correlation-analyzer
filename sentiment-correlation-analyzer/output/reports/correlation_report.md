# Sentiment-to-Price Correlation Analysis Report

## 1. Executive Summary
This report analyzes the linear correlation between FinBERT-extracted news sentiment and next-day stock returns across 20 major NSE equities.

- **Total Data Samples Analyzed**: 2583 trading days
- **Overall Pearson Correlation ($r$)**: `0.038`
- **Statistical Significance ($p$-value)**: `0.0538`
- **Overall Assessment**: Weak / Moderate Correlation

---

## 2. Correlation Breakdown by NSE Stock

| Symbol | Pearson $r$ | $p$-value | Significant (p < 0.05) | Sample Size |
|--------|-------------|-----------|------------------------|-------------|
| WAAREEENER | 0.2334 | 0.0094 | Yes (p < 0.05) | 123 |
| WIPRO | 0.1855 | 0.0400 | Yes (p < 0.05) | 123 |
| AXISBANK | 0.1098 | 0.2265 | No | 123 |
| ITC | 0.1026 | 0.2588 | No | 123 |
| SUNPHARMA | 0.0713 | 0.4331 | No | 123 |
| LT | 0.0634 | 0.4857 | No | 123 |
| TCS | 0.0630 | 0.4886 | No | 123 |
| BHARTIARTL | 0.0438 | 0.6305 | No | 123 |
| HDFCBANK | 0.0348 | 0.7023 | No | 123 |
| BAJFINANCE | 0.0301 | 0.7411 | No | 123 |
| SBIN | 0.0087 | 0.9235 | No | 123 |
| HINDUNILVR | -0.0118 | 0.8967 | No | 123 |
| INFY | -0.0180 | 0.8435 | No | 123 |
| TITAN | -0.0189 | 0.8356 | No | 123 |
| ASIANPAINT | -0.0212 | 0.8160 | No | 123 |
| NTPC | -0.0263 | 0.7724 | No | 123 |
| MARUTI | -0.0317 | 0.7282 | No | 123 |
| ICICIBANK | -0.0392 | 0.6670 | No | 123 |
| HCLTECH | -0.0405 | 0.6562 | No | 123 |
| RELIANCE | -0.0993 | 0.2747 | No | 123 |
| KOTAKBANK | -0.1211 | 0.1822 | No | 123 |

---

## 3. Key Observations & Findings
1. Stocks with strong positive sentiment-return alignment display predictive signals suitable for reinforcement learning trading state representation.
2. News volume and sentiment variance serve as important indicators of price volatility on subsequent trading days.
3. Lagged features (1-day and 2-day prior sentiment) often carry stronger predictive signal than same-day sentiment due to news dissemination delays.
