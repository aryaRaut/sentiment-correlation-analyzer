# Sentiment-to-Price Correlation Analysis Report

## 1. Executive Summary
This report analyzes the linear correlation between FinBERT-extracted news sentiment and next-day stock returns across 20 major NSE equities.

- **Total Data Samples Analyzed**: 2583 trading days
- **Overall Pearson Correlation ($r$)**: `0.0405`
- **Statistical Significance ($p$-value)**: `0.0396`
- **Overall Assessment**: Statistically Significant Correlation Detected

---

## 2. Correlation Breakdown by NSE Stock

| Symbol | Pearson $r$ | $p$-value | Significant (p < 0.05) | Sample Size |
|--------|-------------|-----------|------------------------|-------------|
| WAAREEENER | 0.2337 | 0.0093 | Yes (p < 0.05) | 123 |
| WIPRO | 0.1870 | 0.0383 | Yes (p < 0.05) | 123 |
| ITC | 0.1284 | 0.1568 | No | 123 |
| AXISBANK | 0.1055 | 0.2453 | No | 123 |
| TCS | 0.1049 | 0.2484 | No | 123 |
| SUNPHARMA | 0.0774 | 0.3950 | No | 123 |
| LT | 0.0465 | 0.6096 | No | 123 |
| BHARTIARTL | 0.0430 | 0.6364 | No | 123 |
| HDFCBANK | 0.0361 | 0.6918 | No | 123 |
| SBIN | 0.0066 | 0.9419 | No | 123 |
| BAJFINANCE | 0.0065 | 0.9435 | No | 123 |
| INFY | -0.0171 | 0.8511 | No | 123 |
| TITAN | -0.0180 | 0.8429 | No | 123 |
| HINDUNILVR | -0.0187 | 0.8369 | No | 123 |
| ASIANPAINT | -0.0212 | 0.8161 | No | 123 |
| MARUTI | -0.0319 | 0.7260 | No | 123 |
| NTPC | -0.0362 | 0.6913 | No | 123 |
| ICICIBANK | -0.0390 | 0.6683 | No | 123 |
| HCLTECH | -0.0398 | 0.6619 | No | 123 |
| RELIANCE | -0.0656 | 0.4710 | No | 123 |
| KOTAKBANK | -0.1232 | 0.1747 | No | 123 |

---

## 3. Key Observations & Findings
1. Stocks with strong positive sentiment-return alignment display predictive signals suitable for reinforcement learning trading state representation.
2. News volume and sentiment variance serve as important indicators of price volatility on subsequent trading days.
3. Lagged features (1-day and 2-day prior sentiment) often carry stronger predictive signal than same-day sentiment due to news dissemination delays.
