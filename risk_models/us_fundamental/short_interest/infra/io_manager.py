"""
S3 IO Manager — live and backfill lineage modes.
Supports LocalStack via LOCALSTACK_ENDPOINT env var.

S3 path schema
--------------
LIVE      s3://{bucket}/model=us_fundamental/dataset={dataset}/lineage=live/latest.parquet
BACKFILL  s3://{bucket}/model=us_fundamental/dataset={dataset}/lineage=backfill/build={build_ts}/data.parquet
"""

from __future__ import annotations
import os

import boto3
import dagster as dg
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO

from .. import config


def _s3_client():
    endpoint = os.getenv("LOCALSTACK_ENDPOINT")
    if endpoint:
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
    return boto3.client("s3")


class S3RiskModelIOManager(dg.ConfigurableIOManager):
    """
    Parameters
    ----------
    bucket  : S3 bucket name (default from config.S3_BUCKET)
    lineage : "live" | "backfill"
    """

    bucket:  str = config.S3_BUCKET
    lineage: str = "live"

    def _key(self, context: dg.OutputContext | dg.InputContext) -> str:
        dataset = context.asset_key.path[-1]
        base = f"model={config.S3_MODEL}/dataset={dataset}"

        if self.lineage == "live":
            return f"{base}/lineage=live/latest.parquet"

        build_ts = self._resolve_run_id(context)[:8]
        return f"{base}/lineage=backfill/build={build_ts}/data.parquet"

    @staticmethod
    def _resolve_run_id(context: dg.OutputContext | dg.InputContext) -> str:
        """
        OutputContext exposes .run_id directly (used from handle_output, when
        this asset is the one being written). InputContext does not — it
        only knows about the OutputContext that produced the value it's
        loading (used from load_input, e.g. when an asset_check reads this
        asset's just-materialized output). Dagster asset_checks execute in
        the same run as the asset materialization they check, so
        upstream_output.run_id resolves to the exact build_ts handle_output
        just wrote to.
        """
        if isinstance(context, dg.InputContext):
            if context.upstream_output is None:
                raise dg.DagsterInvariantViolationError(
                    "Cannot resolve run_id: InputContext.upstream_output is not set. "
                    "This IO manager's backfill lineage mode requires reading an input "
                    "whose producing run is known."
                )
            return context.upstream_output.run_id
        return context.run_id

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame) -> None:
        if obj is None or obj.empty:
            context.log.warning(f"Empty DataFrame for {context.asset_key} — skipping write")
            return

        key = self._key(context)
        context.log.info(f"[{self.lineage.upper()}] write → s3://{self.bucket}/{key}")

        buf = BytesIO()
        obj.to_parquet(buf, index=False)
        buf.seek(0)

        s3 = _s3_client()
        s3.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())

    def load_input(self, context: dg.InputContext) -> pd.DataFrame:
        key = self._key(context)
        context.log.info(f"[{self.lineage.upper()}] read ← s3://{self.bucket}/{key}")

        s3 = _s3_client()
        obj = s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
