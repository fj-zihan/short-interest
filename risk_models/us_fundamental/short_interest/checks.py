"""
checks
======
Tier 2 (ERROR, blocking) and Tier 3 (WARN, non-blocking) asset validation.

Tier 1 is NOT here — Dagster asset_checks only run after an asset has
materialized, so they cannot stop a broken dataset from ever being
written. The catastrophic-failure guards live inline inside the asset
compute functions instead:
  * short_interest_raw.py   — universe coverage, required-column NaN ratio
  * short_interest_factor.py — defensive input/output row-count guard

Everything below is Tier 2/3, implemented as @dg.asset_check. `blocking=True`
is what actually enforces ERROR-blocks-downstream / WARN-does-not; the
`severity` field is UI classification, not the blocking mechanism.

Exposure stability (Δ si_factor between settlement dates) is diagnostic
only (Tier 3, non-blocking) and is only wired to short_interest_factor_full.
short_interest_factor_incremental carries a single settlement_date per
materialization, so a same-run Δ is not available; reading the prior
period back from S3 to compute it would make this check depend on S3 as a
state store, which we're deliberately avoiding — see design notes.
"""

import dagster as dg
import pandas as pd

from . import validation as v
from .assets.short_interest_factor import short_interest_factor_full, short_interest_factor_incremental
from .assets.short_interest_raw import REQUIRED_COLUMNS, short_interest_raw
from .config import (
    FACTOR_DELTA_T3_ABS,
    FACTOR_MEAN_T2_ABS,
    FACTOR_MEAN_T3_ABS,
    FACTOR_STD_T2_ABS,
    FACTOR_STD_T3_ABS,
    NAN_T2_RATIO,
    NAN_T3_RATIO,
)

RAW_KEY_COLS    = ["ticker", "settlement_date"]
FACTOR_COLUMNS  = {"ticker", "settlement_date", "days_to_cover", "si_factor"}


# ---------------------------------------------------------------------------
# short_interest_raw — Tier 2 (schema + duplicates are binary, no T3 needed)
# ---------------------------------------------------------------------------

@dg.asset_check(asset=short_interest_raw, blocking=True)
def raw_schema_check(short_interest_raw: pd.DataFrame) -> dg.AssetCheckResult:
    issues = v.schema_issues(short_interest_raw, set(REQUIRED_COLUMNS))
    return dg.AssetCheckResult(
        passed=not issues,
        severity=dg.AssetCheckSeverity.ERROR,
        description="; ".join(issues) if issues else "all required columns present",
    )


@dg.asset_check(asset=short_interest_raw, blocking=True)
def raw_duplicate_check(short_interest_raw: pd.DataFrame) -> dg.AssetCheckResult:
    dupes = v.duplicate_key_count(short_interest_raw, RAW_KEY_COLS)
    return dg.AssetCheckResult(
        passed=dupes == 0,
        severity=dg.AssetCheckSeverity.ERROR,
        metadata={"duplicate_rows": dupes},
        description=(
            f"{dupes} rows share a duplicate (ticker, settlement_date) key"
            if dupes else "no duplicate (ticker, settlement_date) keys"
        ),
    )


@dg.asset_check(asset=short_interest_raw, blocking=True)
def raw_nan_check_t2(short_interest_raw: pd.DataFrame) -> dg.AssetCheckResult:
    ratio = v.required_columns_nan_ratio(short_interest_raw, REQUIRED_COLUMNS)
    return dg.AssetCheckResult(
        passed=ratio <= NAN_T2_RATIO,
        severity=dg.AssetCheckSeverity.ERROR,
        metadata={"nan_ratio": ratio, "threshold": NAN_T2_RATIO},
    )


@dg.asset_check(asset=short_interest_raw, blocking=False)
def raw_nan_check_t3(short_interest_raw: pd.DataFrame) -> dg.AssetCheckResult:
    ratio = v.required_columns_nan_ratio(short_interest_raw, REQUIRED_COLUMNS)
    return dg.AssetCheckResult(
        passed=ratio <= NAN_T3_RATIO,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={"nan_ratio": ratio, "threshold": NAN_T3_RATIO},
    )


# ---------------------------------------------------------------------------
# Factor assets — shared check bodies, wired to both _full and _incremental
# ---------------------------------------------------------------------------

def _schema_result(df: pd.DataFrame) -> dg.AssetCheckResult:
    issues = v.schema_issues(df, FACTOR_COLUMNS)
    return dg.AssetCheckResult(
        passed=not issues,
        severity=dg.AssetCheckSeverity.ERROR,
        description="; ".join(issues) if issues else "all expected columns present",
    )


def _duplicate_result(df: pd.DataFrame) -> dg.AssetCheckResult:
    dupes = v.duplicate_key_count(df, RAW_KEY_COLS)
    return dg.AssetCheckResult(
        passed=dupes == 0,
        severity=dg.AssetCheckSeverity.ERROR,
        metadata={"duplicate_rows": dupes},
    )


def _mean_result(df: pd.DataFrame, threshold: float, severity: dg.AssetCheckSeverity) -> dg.AssetCheckResult:
    dev = v.worst_mean_abs_deviation(df)
    return dg.AssetCheckResult(
        passed=dev <= threshold,
        severity=severity,
        metadata={"worst_abs_mean": dev, "threshold": threshold},
    )


def _std_result(df: pd.DataFrame, threshold: float, severity: dg.AssetCheckSeverity) -> dg.AssetCheckResult:
    dev = v.worst_std_abs_deviation(df)
    return dg.AssetCheckResult(
        passed=dev <= threshold,
        severity=severity,
        metadata={"worst_abs_std_deviation": dev, "threshold": threshold},
    )


def _inf_result(df: pd.DataFrame) -> dg.AssetCheckResult:
    n = v.inf_count(df)
    return dg.AssetCheckResult(
        passed=n == 0,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={"inf_count": n},
    )


def _delta_result(df: pd.DataFrame) -> dg.AssetCheckResult:
    delta = v.max_abs_delta_last_two_dates(df)
    if delta is None:
        # Fewer than two settlement_dates in this materialization — not
        # applicable, not a failure. Expected every run for _incremental.
        return dg.AssetCheckResult(
            passed=True,
            severity=dg.AssetCheckSeverity.WARN,
            description="fewer than two settlement_dates in this materialization — skipped",
        )
    return dg.AssetCheckResult(
        passed=delta <= FACTOR_DELTA_T3_ABS,
        severity=dg.AssetCheckSeverity.WARN,
        metadata={"max_abs_delta": delta, "threshold": FACTOR_DELTA_T3_ABS},
    )


def _build_factor_checks(asset_def, check_prefix: str) -> list:
    """
    Attach the full Tier 2/3 check set to a factor asset. Both
    short_interest_factor_full and short_interest_factor_incremental get
    the identical set — same schema, same standardization contract.

    Each inner check function takes exactly one parameter (`df`). Dagster's
    asset_check decorator binds a single-parameter function to its `asset=`
    target positionally — the parameter name does not need to match the
    asset's own name, so `df` is fine here and lets us reuse one factory
    for both assets without generating differently-named functions.
    """

    @dg.asset_check(asset=asset_def, name=f"{check_prefix}_schema", blocking=True)
    def schema_check(df: pd.DataFrame) -> dg.AssetCheckResult:
        return _schema_result(df)

    @dg.asset_check(asset=asset_def, name=f"{check_prefix}_duplicates", blocking=True)
    def duplicate_check(df: pd.DataFrame) -> dg.AssetCheckResult:
        return _duplicate_result(df)

    @dg.asset_check(asset=asset_def, name=f"{check_prefix}_mean_t2", blocking=True)
    def mean_check_t2(df: pd.DataFrame) -> dg.AssetCheckResult:
        return _mean_result(df, FACTOR_MEAN_T2_ABS, dg.AssetCheckSeverity.ERROR)

    @dg.asset_check(asset=asset_def, name=f"{check_prefix}_mean_t3", blocking=False)
    def mean_check_t3(df: pd.DataFrame) -> dg.AssetCheckResult:
        return _mean_result(df, FACTOR_MEAN_T3_ABS, dg.AssetCheckSeverity.WARN)

    @dg.asset_check(asset=asset_def, name=f"{check_prefix}_std_t2", blocking=True)
    def std_check_t2(df: pd.DataFrame) -> dg.AssetCheckResult:
        return _std_result(df, FACTOR_STD_T2_ABS, dg.AssetCheckSeverity.ERROR)

    @dg.asset_check(asset=asset_def, name=f"{check_prefix}_std_t3", blocking=False)
    def std_check_t3(df: pd.DataFrame) -> dg.AssetCheckResult:
        return _std_result(df, FACTOR_STD_T3_ABS, dg.AssetCheckSeverity.WARN)

    @dg.asset_check(asset=asset_def, name=f"{check_prefix}_inf", blocking=False)
    def inf_check(df: pd.DataFrame) -> dg.AssetCheckResult:
        return _inf_result(df)

    @dg.asset_check(asset=asset_def, name=f"{check_prefix}_delta_stability", blocking=False)
    def delta_check(df: pd.DataFrame) -> dg.AssetCheckResult:
        return _delta_result(df)

    return [schema_check, duplicate_check, mean_check_t2, mean_check_t3,
            std_check_t2, std_check_t3, inf_check, delta_check]


factor_full_checks        = _build_factor_checks(short_interest_factor_full, "short_interest_factor_full")
factor_incremental_checks = _build_factor_checks(short_interest_factor_incremental, "short_interest_factor_incremental")

raw_checks = [raw_schema_check, raw_duplicate_check, raw_nan_check_t2, raw_nan_check_t3]

all_checks = raw_checks + factor_full_checks + factor_incremental_checks
