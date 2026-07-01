"""
validation
==========
Pure metric functions for asset validation. No Dagster imports here on
purpose — these are plain pandas functions so they can be unit tested
without spinning up an asset context, and so the same metric computation
can back both:

  * Tier 1 (inline `raise dg.Failure(...)` inside an asset compute function,
    before `return` — see short_interest_raw.py)
  * Tier 2 / Tier 3 (`@dg.asset_check` — see checks.py)

Each function returns a plain number (or dict of numbers) with no opinion
about severity. Severity thresholds live in config.py; the raise-or-warn
decision lives in short_interest_raw.py / checks.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Universe coverage
# ---------------------------------------------------------------------------

def coverage_ratio_by_date(df: pd.DataFrame, universe_size: int,
                            date_col: str = "settlement_date") -> dict:
    """Ticker count / universe_size for each settlement_date present."""
    if universe_size <= 0 or df.empty or date_col not in df.columns:
        return {}
    counts = df.groupby(date_col).size()
    return (counts / universe_size).to_dict()


def worst_coverage_ratio(df: pd.DataFrame, universe_size: int,
                          date_col: str = "settlement_date") -> float:
    """
    Worst-case coverage across all settlement_dates in the panel.
    Empty input (or zero-size universe) is coverage 0.0 — that is
    intentional: an empty raw fetch must read as a catastrophic failure,
    not as "no dates to check".
    """
    ratios = coverage_ratio_by_date(df, universe_size, date_col)
    if not ratios:
        return 0.0
    return min(ratios.values())


# ---------------------------------------------------------------------------
# Missing-value / structural integrity
# ---------------------------------------------------------------------------

def required_columns_nan_ratio(df: pd.DataFrame, columns: list[str]) -> float:
    """
    Worst-case NaN ratio across `columns`. A column missing from `df`
    entirely counts as 100% NaN — that is deliberate, a dropped required
    column is at least as bad as a column full of NaNs.
    """
    if df.empty:
        return 1.0
    worst = 0.0
    for col in columns:
        if col not in df.columns:
            worst = max(worst, 1.0)
            continue
        ratio = df[col].isna().mean()
        worst = max(worst, ratio)
    return worst


# ---------------------------------------------------------------------------
# Schema / duplicate structural checks (Tier 2, binary)
# ---------------------------------------------------------------------------

def schema_issues(df: pd.DataFrame, expected_columns: set[str]) -> list[str]:
    issues = []
    missing = expected_columns - set(df.columns)
    if missing:
        issues.append(f"missing columns: {sorted(missing)}")
    return issues


def duplicate_key_count(df: pd.DataFrame, key_cols: list[str]) -> int:
    if df.empty:
        return 0
    return int(df.duplicated(subset=key_cols, keep=False).sum())


# ---------------------------------------------------------------------------
# Factor standardization (Tier 2 / Tier 3)
# ---------------------------------------------------------------------------

def standardization_by_date(df: pd.DataFrame, value_col: str = "si_factor",
                             date_col: str = "settlement_date") -> pd.DataFrame:
    """Per-settlement_date mean and std of `value_col`."""
    if df.empty:
        return pd.DataFrame(columns=[date_col, "mean", "std"])
    grp = df.groupby(date_col)[value_col]
    return grp.agg(mean="mean", std="std").reset_index()


def worst_mean_abs_deviation(df: pd.DataFrame, value_col: str = "si_factor",
                              date_col: str = "settlement_date") -> float:
    stats = standardization_by_date(df, value_col, date_col)
    if stats.empty:
        return 0.0
    return float(stats["mean"].abs().max())


def worst_std_abs_deviation(df: pd.DataFrame, value_col: str = "si_factor",
                             date_col: str = "settlement_date") -> float:
    stats = standardization_by_date(df, value_col, date_col)
    if stats.empty:
        return 0.0
    return float((stats["std"] - 1.0).abs().max())


# ---------------------------------------------------------------------------
# Inf check (Tier 3, binary)
# ---------------------------------------------------------------------------

def inf_count(df: pd.DataFrame, col: str = "si_factor") -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(np.isinf(df[col]).sum())


# ---------------------------------------------------------------------------
# Exposure stability (Tier 3, diagnostic only — see checks.py docstring)
# ---------------------------------------------------------------------------

def max_abs_delta_last_two_dates(df: pd.DataFrame, value_col: str = "si_factor",
                                  date_col: str = "settlement_date",
                                  ticker_col: str = "ticker") -> float | None:
    """
    Max |Δ value_col| per ticker between the two most recent settlement_dates
    present in `df`. Returns None if fewer than two dates are present (this
    is expected for `short_interest_factor_incremental`, which only ever
    carries one date per materialization — see checks.py for how that
    asset's check handles this).
    """
    if df.empty or date_col not in df.columns:
        return None
    dates = sorted(df[date_col].unique())
    if len(dates) < 2:
        return None

    prev_date, curr_date = dates[-2], dates[-1]
    prev = df[df[date_col] == prev_date].set_index(ticker_col)[value_col]
    curr = df[df[date_col] == curr_date].set_index(ticker_col)[value_col]
    common = prev.index.intersection(curr.index)
    if common.empty:
        return None

    delta = (curr.loc[common] - prev.loc[common]).abs()
    return float(delta.max())
