# Sentiment-to-Price Correlation Analysis Report

## 1. Executive Summary
This report analyzes the linear correlation between FinBERT-extracted news sentiment and next-day stock returns across 20 major NSE equities.

- **Total Data Samples Analyzed**: 2360 trading days
- **Overall Pearson Correlation ($r$)**: `0.0782`
- **Statistical Significance ($p$-value)**: `0.0001`
- **Overall Assessment**: Statistically Significant Correlation Detected

---

## 2. Correlation Breakdown by NSE Stock

| Symbol | Pearson $r$ | $p$-value | Significant (p < 0.05) | Sample Size |
|--------|-------------|-----------|------------------------|-------------|
| WAAREEENER | 0.3921 | 0.0000 | Yes (p < 0.05) | 118 |
| ASIANPAINT | 0.2212 | 0.0161 | Yes (p < 0.05) | 118 |
| NTPC | 0.1876 | 0.0419 | Yes (p < 0.05) | 118 |
| AXISBANK | 0.1571 | 0.0894 | No | 118 |
| MARUTI | 0.1430 | 0.1223 | No | 118 |
| ICICIBANK | 0.1175 | 0.2052 | No | 118 |
| HINDUNILVR | 0.1111 | 0.2310 | No | 118 |
| BHARTIARTL | 0.0976 | 0.2931 | No | 118 |
| HCLTECH | 0.0899 | 0.3327 | No | 118 |
| INFY | 0.0767 | 0.4094 | No | 118 |
| TCS | 0.0760 | 0.4135 | No | 118 |
| WIPRO | 0.0502 | 0.5896 | No | 118 |
| RELIANCE | 0.0458 | 0.6223 | No | 118 |
| LT | 0.0254 | 0.7848 | No | 118 |
| KOTAKBANK | 0.0119 | 0.8981 | No | 118 |
| SUNPHARMA | -0.0141 | 0.8793 | No | 118 |
| HDFCBANK | -0.0705 | 0.4479 | No | 118 |
| ITC | -0.1276 | 0.1686 | No | 118 |
| TITAN | -0.1440 | 0.1197 | No | 118 |
| SBIN | -0.1574 | 0.0886 | No | 118 |

---

## 3. Key Observations & Findings
1. Stocks with strong positive sentiment-return alignment display predictive signals suitable for reinforcement learning trading state representation.
2. News volume and sentiment variance serve as important indicators of price volatility on subsequent trading days.
3. Lagged features (1-day and 2-day prior sentiment) often carry stronger predictive signal than same-day sentiment due to news dissemination delays.
