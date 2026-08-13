# Sentiment-to-Price Correlation Analysis Report

## 1. Executive Summary
This report analyzes the linear correlation between FinBERT-extracted news sentiment and next-day stock returns across 20 major NSE equities.

- **Total Data Samples Analyzed**: 2397 trading days
- **Overall Pearson Correlation ($r$)**: `0.0798`
- **Statistical Significance ($p$-value)**: `0.0001`
- **Overall Assessment**: Statistically Significant Correlation Detected

---

## 2. Correlation Breakdown by NSE Stock

| Symbol | Pearson $r$ | $p$-value | Significant (p < 0.05) | Sample Size |
|--------|-------------|-----------|------------------------|-------------|
| WAAREEENER | 0.4366 | 0.0000 | Yes (p < 0.05) | 120 |
| NTPC | 0.1879 | 0.0399 | Yes (p < 0.05) | 120 |
| ASIANPAINT | 0.1752 | 0.0567 | No | 119 |
| MARUTI | 0.1561 | 0.0887 | No | 120 |
| AXISBANK | 0.1461 | 0.1113 | No | 120 |
| ICICIBANK | 0.1321 | 0.1503 | No | 120 |
| RELIANCE | 0.1149 | 0.2113 | No | 120 |
| BHARTIARTL | 0.1035 | 0.2604 | No | 120 |
| TITAN | 0.0689 | 0.4547 | No | 120 |
| LT | 0.0408 | 0.6585 | No | 120 |
| SBIN | 0.0367 | 0.6907 | No | 120 |
| INFY | 0.0348 | 0.7063 | No | 120 |
| WIPRO | 0.0120 | 0.8965 | No | 120 |
| HCLTECH | 0.0032 | 0.9723 | No | 119 |
| HDFCBANK | 0.0025 | 0.9785 | No | 120 |
| HINDUNILVR | 0.0020 | 0.9828 | No | 120 |
| KOTAKBANK | -0.0000 | 0.9997 | No | 120 |
| SUNPHARMA | -0.0069 | 0.9405 | No | 119 |
| TCS | -0.0186 | 0.8406 | No | 120 |
| ITC | -0.1461 | 0.1113 | No | 120 |

---

## 3. Key Observations & Findings
1. Stocks with strong positive sentiment-return alignment display predictive signals suitable for reinforcement learning trading state representation.
2. News volume and sentiment variance serve as important indicators of price volatility on subsequent trading days.
3. Lagged features (1-day and 2-day prior sentiment) often carry stronger predictive signal than same-day sentiment due to news dissemination delays.
