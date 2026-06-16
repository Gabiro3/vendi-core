"""Small helpers for converting between DataFrames and Parquet bytes.

Used by the datasets router (write the processed dataset) and the worker
(read it back for each ML module).
"""

from __future__ import annotations

import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(df, preserve_index=False)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def parquet_bytes_to_df(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data))
