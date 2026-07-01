import dagster as dg
import os
from dotenv import load_dotenv

load_dotenv()

from .assets.short_interest_raw import short_interest_raw
from .assets.short_interest_factor import (
    short_interest_factor_full,
    short_interest_factor_incremental,
)
from .checks import all_checks
from .data.universe import sp500_universe
from .infra.massive import MassiveProvider
from .infra.io_manager import S3RiskModelIOManager

# -------------------------
# Resources
# -------------------------
resources = {
    "io_manager":         S3RiskModelIOManager(lineage="live"),
    "backfill_io_manager": S3RiskModelIOManager(lineage="backfill"),
    "si_provider": MassiveProvider(
        api_key=os.getenv("MASSIVE_API_KEY")
    ),
}

# -------------------------
# Jobs
# -------------------------
full_backfill_job = dg.define_asset_job(
    name="short_interest_full_backfill_job",
    selection=[
        "sp500_universe",
        "short_interest_raw",
        "short_interest_factor_full",
    ],
)

incremental_job = dg.define_asset_job(
    name="short_interest_incremental_job",
    selection=[
        "sp500_universe",
        "short_interest_raw",
        "short_interest_factor_incremental",
    ],
)

# -------------------------
# Definitions
# -------------------------
defs = dg.Definitions(
    assets=[
        sp500_universe,
        short_interest_raw,
        short_interest_factor_full,
        short_interest_factor_incremental,
    ],
    asset_checks=all_checks,
    jobs=[
        full_backfill_job,
        incremental_job,
    ],
    resources=resources,
)
