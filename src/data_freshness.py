"""
Data freshness utilities for detecting and reporting stale data.
"""

import datetime
import pandas as pd
import streamlit as st


def get_data_freshness(df, date_col='date'):
    """
    Calculate data freshness metrics.
    
    Returns:
        dict with:
        - latest_date: The most recent date in the data
        - days_stale: Number of days since the latest data
        - is_stale: Boolean indicating if data is stale (>1 day old)
        - is_fresh: Boolean indicating if data is from today or yesterday
        - freshness_label: Human-readable label (Fresh / Slightly Stale / Very Stale)
        - reason: Explanation for staleness (weekend / holiday / yfinance delay)
    """
    if df is None or len(df) == 0:
        return {
            'latest_date': None,
            'days_stale': None,
            'is_stale': True,
            'is_fresh': False,
            'freshness_label': 'No Data',
            'reason': 'No data available in the dataset.'
        }
    
    # Get latest date in the data
    latest_date = pd.to_datetime(df[date_col]).max().date()
    today = datetime.date.today()
    days_stale = (today - latest_date).days
    
    # Determine freshness
    is_stale = days_stale > 1
    is_fresh = days_stale <= 1
    
    # Determine freshness label
    if days_stale == 0:
        freshness_label = "✅ Fresh (Today)"
    elif days_stale == 1:
        freshness_label = "✅ Fresh (Yesterday)"
    elif days_stale <= 3:
        freshness_label = f"⚠️ Slightly Stale ({days_stale} days old)"
    else:
        freshness_label = f"🔴 Very Stale ({days_stale} days old)"
    
    # Determine likely reason
    reason = _get_staleness_reason(latest_date, today, days_stale)
    
    return {
        'latest_date': latest_date,
        'days_stale': days_stale,
        'is_stale': is_stale,
        'is_fresh': is_fresh,
        'freshness_label': freshness_label,
        'reason': reason
    }


def _get_staleness_reason(latest_date, today, days_stale):
    """Determine the likely reason for data staleness."""
    # Check if it's a weekend
    weekday = today.weekday()  # 0=Monday, 6=Sunday
    
    if weekday == 5:  # Saturday
        return "Today is Saturday. Markets are closed. Showing data from the last trading day."
    elif weekday == 6:  # Sunday
        return "Today is Sunday. Markets are closed. Showing data from the last trading day."
    elif weekday == 0 and days_stale >= 2:  # Monday
        return "Today is Monday. Data may not have updated over the weekend yet."
    elif days_stale > 3:
        return "Data has not been refreshed for several days. This could be due to yfinance delay, a market holiday, or the pipeline failing."
    elif days_stale >= 1:
        return "Yahoo Finance's NSE data feed can be delayed by several hours after market close. The data may update later today or tomorrow morning."
    else:
        return "Data is up to date."


def display_freshness_banner(df, date_col='date'):
    """
    Display a freshness banner on the dashboard.
    Returns the freshness dict for further use.
    """
    freshness = get_data_freshness(df, date_col)
    
    if freshness['is_fresh']:
        # Data is fresh — show a subtle success banner
        st.success(
            f"✅ **Data is fresh** — Latest data: {freshness['latest_date']}",
            icon="✅"
        )
    else:
        # Data is stale — show a warning banner with reason
        st.warning(
            f"""
            ⚠️ **Data may be stale** — Latest data: **{freshness['latest_date']}** 
            ({freshness['days_stale']} day{'s' if freshness['days_stale'] != 1 else ''} old)
            
            **Reason:** {freshness['reason']}
            
            💡 **Tip:** NSE data via Yahoo Finance can lag by several hours. 
            If you're sure the market has closed, click "Refresh Data Pipeline" to re-fetch.
            """
        )
    
    return freshness
