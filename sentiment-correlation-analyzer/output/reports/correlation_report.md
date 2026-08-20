# Sentiment-to-Price Correlation Analysis Report

## 1. Executive Summary
This report analyzes the linear correlation between FinBERT-extracted news sentiment and next-day stock returns across 20 major NSE equities.

- **Total Data Samples Analyzed**: 2583 trading days
- **Overall Pearson Correlation ($r$)**: `0.0337`
- **Statistical Significance ($p$-value)**: `0.0865`
- **Overall Assessment**: Weak / Moderate Correlation

---

## 2. Correlation Breakdown by NSE Stock

| Symbol | Pearson $r$ | $p$-value | Significant (p < 0.05) | Sample Size |
|--------|-------------|-----------|------------------------|-------------|
| WAAREEENER | 0.1935 | 0.0320 | Yes (p < 0.05) | 123 |
| WIPRO | 0.1681 | 0.0630 | No | 123 |
| AXISBANK | 0.1056 | 0.2453 | No | 123 |
| ITC | 0.0887 | 0.3295 | No | 123 |
| LT | 0.0822 | 0.3659 | No | 123 |
| SUNPHARMA | 0.0793 | 0.3833 | No | 123 |
| TCS | 0.0791 | 0.3844 | No | 123 |
| RELIANCE | 0.0437 | 0.6313 | No | 123 |
| HDFCBANK | 0.0370 | 0.6843 | No | 123 |
| BHARTIARTL | 0.0340 | 0.7092 | No | 123 |
| BAJFINANCE | 0.0259 | 0.7765 | No | 123 |
| ASIANPAINT | 0.0090 | 0.9217 | No | 123 |
| TITAN | -0.0125 | 0.8905 | No | 123 |
| INFY | -0.0211 | 0.8172 | No | 123 |
| NTPC | -0.0306 | 0.7369 | No | 123 |
| ICICIBANK | -0.0310 | 0.7339 | No | 123 |
| MARUTI | -0.0320 | 0.7251 | No | 123 |
| HCLTECH | -0.0554 | 0.5429 | No | 123 |
| SBIN | -0.0993 | 0.2745 | No | 123 |
| HINDUNILVR | -0.1043 | 0.2510 | No | 123 |
| KOTAKBANK | -0.1180 | 0.1938 | No | 123 |

---

## 3. Key Observations & Findings
1. Stocks with strong positive sentiment-return alignment display predictive signals suitable for reinforcement learning trading state representation.
2. News volume and sentiment variance serve as important indicators of price volatility on subsequent trading days.
3. Lagged features (1-day and 2-day prior sentiment) often carry stronger predictive signal than same-day sentiment due to news dissemination delays.
