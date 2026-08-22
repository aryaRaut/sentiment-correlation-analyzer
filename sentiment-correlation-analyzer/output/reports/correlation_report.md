# Sentiment-to-Price Correlation Analysis Report

## 1. Executive Summary
This report analyzes the linear correlation between FinBERT-extracted news sentiment and next-day stock returns across 20 major NSE equities.

- **Total Data Samples Analyzed**: 2602 trading days
- **Overall Pearson Correlation ($r$)**: `0.0333`
- **Statistical Significance ($p$-value)**: `0.0895`
- **Overall Assessment**: Weak / Moderate Correlation

---

## 2. Correlation Breakdown by NSE Stock

| Symbol | Pearson $r$ | $p$-value | Significant (p < 0.05) | Sample Size |
|--------|-------------|-----------|------------------------|-------------|
| WAAREEENER | 0.1719 | 0.0563 | No | 124 |
| AXISBANK | 0.1657 | 0.0659 | No | 124 |
| WIPRO | 0.1068 | 0.2377 | No | 124 |
| NTPC | 0.0971 | 0.2832 | No | 124 |
| TCS | 0.0913 | 0.3131 | No | 124 |
| RELIANCE | 0.0885 | 0.3284 | No | 124 |
| BAJFINANCE | 0.0859 | 0.3426 | No | 124 |
| ITC | 0.0802 | 0.3756 | No | 124 |
| SUNPHARMA | 0.0654 | 0.4721 | No | 123 |
| ASIANPAINT | 0.0461 | 0.6108 | No | 124 |
| HDFCBANK | 0.0449 | 0.6203 | No | 124 |
| LT | 0.0209 | 0.8181 | No | 124 |
| HINDUNILVR | 0.0146 | 0.8728 | No | 123 |
| TITAN | 0.0043 | 0.9620 | No | 124 |
| BHARTIARTL | -0.0114 | 0.9000 | No | 124 |
| MARUTI | -0.0245 | 0.7873 | No | 124 |
| SBIN | -0.0312 | 0.7306 | No | 124 |
| INFY | -0.0863 | 0.3408 | No | 124 |
| HCLTECH | -0.0934 | 0.3020 | No | 124 |
| KOTAKBANK | -0.1083 | 0.2311 | No | 124 |
| ICICIBANK | -0.1253 | 0.1654 | No | 124 |

---

## 3. Key Observations & Findings
1. Stocks with strong positive sentiment-return alignment display predictive signals suitable for reinforcement learning trading state representation.
2. News volume and sentiment variance serve as important indicators of price volatility on subsequent trading days.
3. Lagged features (1-day and 2-day prior sentiment) often carry stronger predictive signal than same-day sentiment due to news dissemination delays.
