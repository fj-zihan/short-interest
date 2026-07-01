"""
short_interest_factor assets
=============================
Two assets with explicit IO manager routing:

  short_interest_factor_full        → backfill_io_manager (all dates, versioned S3)
  short_interest_factor_incremental → io_manager           (live, latest date only)
"""

import dagster as dg
import pandas as pd

from ..config import WINSOR_HIGH, WINSOR_LOW


# ---------------------------------------------------------------------------
# Shared compute function
# ---------------------------------------------------------------------------

def _compute_factor(cross_section: pd.DataFrame) -> pd.DataFrame:
    """Winsorize + cross-sectional z-score for a single settlement date."""
    df = cross_section.copy()

    p_low  = df["days_to_cover"].quantile(WINSOR_LOW)
    p_high = df["days_to_cover"].quantile(WINSOR_HIGH)
    df["days_to_cover_w"] = df["days_to_cover"].clip(lower=p_low, upper=p_high)

    mu    = df["days_to_cover_w"].mean()
    sigma = df["days_to_cover_w"].std()
    df["si_factor"] = 0.0 if sigma == 0 else (df["days_to_cover_w"] - mu) / sigma

    return df[["ticker", "settlement_date", "days_to_cover", "si_factor"]]


def _factor_panel(raw: pd.DataFrame, context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Apply _compute_factor independently per settlement_date (for full backfill)."""
    dates = sorted(raw["settlement_date"].unique())
    context.log.info(f"Computing factor for {len(dates)} settlement date(s)")
    return pd.concat(
        [_compute_factor(raw[raw["settlement_date"] == d]) for d in dates],
        ignore_index=True,
    )


def _defensive_guard(short_interest_raw: pd.DataFrame, result: pd.DataFrame,
                      asset_name: str) -> None:
    """
    Tier 1, defense-in-depth only. short_interest_raw already enforces the
    real coverage/NaN floors before it is allowed to materialize, so this
    should not fire under normal operation — it exists to catch a bug in
    _compute_factor / _factor_panel that silently drops rows, rather than
    re-deriving universe coverage (this asset does not have sp500_universe
    as an input, on purpose — duplicating that dependency here would be a
    bigger DAG change for marginal benefit).
    """
    if not short_interest_raw.empty and result.empty:
        raise dg.Failure(
            description=(
                f"{asset_name}: input short_interest_raw had "
                f"{len(short_interest_raw)} rows but factor computation "
                f"produced 0 rows. This points to a bug in the compute "
                f"step, not a data quality issue — short_interest_raw's "
                f"own Tier 1 guard already passed."
            ),
            metadata={"input_rows": len(short_interest_raw)},
        )


# ---------------------------------------------------------------------------
# Asset 1: full backfill (all dates) — uses backfill_io_manager → lineage=backfill
# ---------------------------------------------------------------------------

@dg.asset(
    group_name="short_interest",
    io_manager_key="backfill_io_manager",
    description="Full backfill: SI factor for ALL settlement dates. Written to S3 lineage=backfill.",
)
def short_interest_factor_full(
    context: dg.AssetExecutionContext,
    short_interest_raw: pd.DataFrame,
) -> pd.DataFrame:
    if short_interest_raw.empty:
        return pd.DataFrame(columns=["ticker", "settlement_date", "days_to_cover", "si_factor"])

    result = _factor_panel(short_interest_raw, context)
    _defensive_guard(short_interest_raw, result, "short_interest_factor_full")
    context.log.info(
        f"Full backfill: {len(result)} rows, "
        f"{result['settlement_date'].nunique()} dates, "
        f"{result['ticker'].nunique()} tickers"
    )
    return result


# ---------------------------------------------------------------------------
# Asset 2: incremental live update (latest date) — uses io_manager → lineage=live
# ---------------------------------------------------------------------------

@dg.asset(
    group_name="short_interest",
    io_manager_key="io_manager",
    description="Incremental live update: SI factor for latest settlement date only. lineage=live.",
)
def short_interest_factor_incremental(
    context: dg.AssetExecutionContext,
    short_interest_raw: pd.DataFrame,
) -> pd.DataFrame:
    if short_interest_raw.empty:
        return pd.DataFrame(columns=["ticker", "settlement_date", "days_to_cover", "si_factor"])

    latest = short_interest_raw["settlement_date"].max()
    cross  = short_interest_raw[short_interest_raw["settlement_date"] == latest]
    result = _compute_factor(cross)
    _defensive_guard(short_interest_raw, result, "short_interest_factor_incremental")
    context.log.info(f"Incremental [{latest.date()}] — {len(result)} tickers")
    return result
