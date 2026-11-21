"""Convert an IBM AML CSV to the parquet the pipeline reads, keeping rows before a cut date.

  python src/prep_data.py data/HI-Small_Trans.csv 2022-09-11 data/hi_sep10.parquet
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import polars as pl
from datetime import datetime
from load import load

src, cut, out = sys.argv[1], datetime.fromisoformat(sys.argv[2]), sys.argv[3]
tmp = out + ".tmp"
(load(src).filter(pl.col("ts") < cut)
    .select("ts", "src", "dst", amt="amt_paid", fmt=pl.col("fmt").cast(pl.Categorical), is_laundering="is_laundering")
    .sink_parquet(tmp))
df = pl.read_parquet(tmp).sort("ts"); df.write_parquet(out); os.remove(tmp)
print(out, df.height, "rows,", df["is_laundering"].sum(), "laundering,", df["ts"].min(), "to", df["ts"].max())
