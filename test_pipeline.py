"""
Unit tests — no API key or S3 required.
Run with: pytest tests/
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock
from dagster import build_asset_context

from risk_models.us_fundamental.short_interest.assets.short_interest_raw import short_interest_raw
from risk_models.us_fundamental.short_interest.assets.short_interest_factor import (
    short_interest_factor_full,
    short_interest_factor_incremental,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(records):
    df = pd.DataFrame(records)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df

def _universe(*tickers):
    return pd.DataFrame({"ticker": list(tickers), "company": list(tickers)})

def _provider(records):
    p = MagicMock()
    p.fetch.return_value = _df(records)
    return p

ONE_DATE = [
    {"ticker": "AAPL", "settlement_date": "2024-12-31", "short_interest": 100, "avg_daily_volume": 50, "days_to_cover": 2.0},
    {"ticker": "MSFT", "settlement_date": "2024-12-31", "short_interest": 200, "avg_daily_volume": 50, "days_to_cover": 4.0},
    {"ticker": "NVDA", "settlement_date": "2024-12-31", "short_interest": 150, "avg_daily_volume": 50, "days_to_cover": 3.0},
    {"ticker": "TSLA", "settlement_date": "2024-12-31", "short_interest": 600, "avg_daily_volume": 50, "days_to_cover": 12.0},
]

TWO_DATES = ONE_DATE + [
    {"ticker": "AAPL", "settlement_date": "2024-12-15", "short_interest": 120, "avg_daily_volume": 50, "days_to_cover": 2.4},
    {"ticker": "MSFT", "settlement_date": "2024-12-15", "short_interest": 180, "avg_daily_volume": 50, "days_to_cover": 3.6},
    {"ticker": "NVDA", "settlement_date": "2024-12-15", "short_interest": 160, "avg_daily_volume": 50, "days_to_cover": 3.2},
    {"ticker": "TSLA", "settlement_date": "2024-12-15", "short_interest": 550, "avg_daily_volume": 50, "days_to_cover": 11.0},
]


# ---------------------------------------------------------------------------
# short_interest_raw
# ---------------------------------------------------------------------------

def test_raw_sends_correct_tickers():
    p = _provider(ONE_DATE)
    short_interest_raw(build_asset_context(), _universe("AAPL", "MSFT", "NVDA", "TSLA"), p)
    assert set(p.fetch.call_args.kwargs["tickers"]) == {"AAPL", "MSFT", "NVDA", "TSLA"}

def test_raw_columns():
    result = short_interest_raw(build_asset_context(), _universe("AAPL"), _provider(ONE_DATE))
    assert {"ticker", "settlement_date", "days_to_cover"}.issubset(result.columns)


# ---------------------------------------------------------------------------
# short_interest_factor_incremental (live, latest date)
# ---------------------------------------------------------------------------

def test_incremental_zscore():
    result = short_interest_factor_incremental(build_asset_context(), _df(ONE_DATE))
    assert abs(result["si_factor"].mean()) < 1e-10
    assert abs(result["si_factor"].std() - 1.0) < 1e-10

def test_incremental_latest_date_only():
    result = short_interest_factor_incremental(build_asset_context(), _df(TWO_DATES))
    assert (result["settlement_date"] == pd.Timestamp("2024-12-31")).all()
    assert len(result) == 4

def test_incremental_empty():
    raw = pd.DataFrame(columns=["ticker", "settlement_date", "short_interest", "avg_daily_volume", "days_to_cover"])
    result = short_interest_factor_incremental(build_asset_context(), raw)
    assert result.empty and "si_factor" in result.columns


# ---------------------------------------------------------------------------
# short_interest_factor_full
# ---------------------------------------------------------------------------

def test_full_covers_all_dates():
    result = short_interest_factor_full(build_asset_context(), _df(TWO_DATES))
    assert result["settlement_date"].nunique() == 2
    assert len(result) == 8

def test_full_zscore_per_date():
    result = short_interest_factor_full(build_asset_context(), _df(TWO_DATES))
    for date, grp in result.groupby("settlement_date"):
        assert abs(grp["si_factor"].mean()) < 1e-10, f"mean != 0 on {date}"
        assert abs(grp["si_factor"].std() - 1.0) < 1e-10, f"std != 1 on {date}"

def test_output_columns():
    result = short_interest_factor_full(build_asset_context(), _df(ONE_DATE))
    assert set(result.columns) == {"ticker", "settlement_date", "days_to_cover", "si_factor"}
