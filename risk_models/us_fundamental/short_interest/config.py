"""
Central configuration — no magic numbers anywhere else.
"""

# S3
S3_BUCKET      = "risk-models"
S3_MODEL       = "us_fundamental"
S3_FREQUENCY   = "biweekly"

# Universe
UNIVERSE_NAME  = "sp500"

# API
LOOKBACK_DAYS  = 90
API_LIMIT      = 50_000

# Factor construction
WINSOR_LOW     = 0.01
WINSOR_HIGH    = 0.99

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
# Each metric has up to three severities. Smaller deviation → lower severity.
#   T1 = inline raise inside asset compute (nothing materialized)
#   T2 = @dg.asset_check, ERROR   (artifact kept, downstream blocked)
#   T3 = @dg.asset_check, WARN    (artifact kept, pipeline continues)

# Universe coverage: len(df) / len(sp500_universe) per settlement_date.
# For multi-date panels, the worst settlement_date decides the severity.
COVERAGE_T1_RATIO = 0.50   # inline raise in short_interest_raw
COVERAGE_T2_RATIO = 0.80   # asset_check ERROR
COVERAGE_T3_RATIO = 0.90   # asset_check WARN

# Required-column NaN ratio (ticker, settlement_date, days_to_cover on raw;
# si_factor on factor output).
NAN_T1_RATIO = 0.50        # inline raise in short_interest_raw
NAN_T2_RATIO = 0.05        # asset_check ERROR
NAN_T3_RATIO = 0.01        # asset_check WARN

# si_factor cross-sectional standardization, evaluated per settlement_date.
FACTOR_MEAN_T3_ABS = 0.05
FACTOR_MEAN_T2_ABS = 0.10
FACTOR_STD_T3_ABS  = 0.10
FACTOR_STD_T2_ABS  = 0.20

# Exposure stability: |Δ si_factor| for a given ticker between the two most
# recent settlement dates present in the same materialization. Diagnostic
# only (T3) — see checks.py docstring for why this is not enforced at T2.
FACTOR_DELTA_T3_ABS = 3.0

