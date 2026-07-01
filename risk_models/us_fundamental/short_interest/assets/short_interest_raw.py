"""
short_interest_raw
==================
Fetches raw FINRA short interest data from the configured provider
for the full S&P 500 universe and writes to S3.
Fetch only — no factor construction here.
"""

from datetime import datetime, timedelta

import dagster as dg
import pandas as pd

from ..config import (
    API_LIMIT,
    COVERAGE_T1_RATIO,
    LOOKBACK_DAYS,
    NAN_T1_RATIO,
)
from ..infra.base import ShortInterestProvider
from ..validation import required_columns_nan_ratio, worst_coverage_ratio

REQUIRED_COLUMNS = ["ticker", "settlement_date", "days_to_cover"]


def _tier1_guard(df: pd.DataFrame, universe_size: int, context: dg.AssetExecutionContext) -> None:
    """
    Tier 1 — catastrophic-failure guard, evaluated before short_interest_raw
    returns anything. This is intentionally NOT a @dg.asset_check: an
    asset_check only runs after materialization, so it cannot prevent a
    broken dataset from being written. Raising dg.Failure here means the
    run fails and nothing is ever handed to the IO manager.
    """
    coverage = worst_coverage_ratio(df, universe_size)
    if coverage < COVERAGE_T1_RATIO:
        raise dg.Failure(
            description=(
                f"short_interest_raw: universe coverage {coverage:.1%} is below the "
                f"Tier 1 floor of {COVERAGE_T1_RATIO:.0%}. This looks like a partial "
                f"materialization or an upstream API failure, not a normal data gap."
            ),
            metadata={"coverage_ratio": coverage, "universe_size": universe_size, "rows": len(df)},
        )

    nan_ratio = required_columns_nan_ratio(df, REQUIRED_COLUMNS)
    if nan_ratio > NAN_T1_RATIO:
        raise dg.Failure(
            description=(
                f"short_interest_raw: {nan_ratio:.1%} of required columns "
                f"{REQUIRED_COLUMNS} are NaN, above the Tier 1 ceiling of "
                f"{NAN_T1_RATIO:.0%}."
            ),
            metadata={"nan_ratio": nan_ratio, "required_columns": REQUIRED_COLUMNS},
        )


@dg.asset(
    group_name="short_interest",
    description=(
        "Raw FINRA short interest from Massive API. "
        "One row per ticker × settlement_date. Written to S3."
    ),
)
def short_interest_raw(
    context: dg.AssetExecutionContext,
    sp500_universe: pd.DataFrame,
    si_provider: dg.ResourceParam[ShortInterestProvider],
) -> pd.DataFrame:
    """
    Output schema
    -------------
    ticker, settlement_date, short_interest, avg_daily_volume, days_to_cover
    """
    tickers = sp500_universe["ticker"].tolist()
    cutoff  = (datetime.today() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    context.log.info(f"Fetching SI for {len(tickers)} tickers from {cutoff}")

    df = si_provider.fetch(
        tickers             = tickers,
        settlement_date_gte = cutoff,
        limit               = API_LIMIT,
    )

    context.log.info(
        f"Fetched {len(df)} records, "
        f"{df['ticker'].nunique() if not df.empty else 0} tickers, "
        f"{df['settlement_date'].nunique() if not df.empty else 0} settlement dates"
    )

    _tier1_guard(df, universe_size=len(sp500_universe), context=context)

    return df

