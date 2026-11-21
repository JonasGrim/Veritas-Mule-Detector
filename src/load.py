"""Load IBM AML synthetic transactions (Altman et al., NeurIPS 2023; Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml).

Files: {HI,LI}-{Small,Medium,Large}_Trans.csv. HI = higher illicit ratio, LI = lower.
Raw columns: Timestamp, From Bank, Account, To Bank, Account.1, Amount Received,
Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering
"""
import polars as pl

RENAME = {
    "Timestamp": "ts", "From Bank": "src_bank", "Account": "src_acct",
    "To Bank": "dst_bank", "Account.1": "dst_acct",
    "Amount Received": "amt_recv", "Receiving Currency": "cur_recv",
    "Amount Paid": "amt_paid", "Payment Currency": "cur_paid",
    "Payment Format": "fmt", "Is Laundering": "is_laundering",
}


def load(path: str) -> pl.LazyFrame:
    # Header repeats "Account"; rename positionally and force IDs to strings
    cols = list(RENAME.values())
    lf = pl.scan_csv(path, new_columns=cols,
                     schema_overrides={c: pl.Utf8 for c in ("src_bank", "src_acct", "dst_bank", "dst_acct")})
    return (
        lf.with_columns(
            pl.col("ts").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M"),
            # Account IDs are only unique within a bank
            (pl.col("src_bank") + ":" + pl.col("src_acct")).alias("src"),
            (pl.col("dst_bank") + ":" + pl.col("dst_acct")).alias("dst"),
            pl.col("is_laundering").cast(pl.Int8),
        )
        .pipe(_to_usd)
        .with_columns(self_txn=(pl.col("src") == pl.col("dst")),
                      cross_bank=(pl.col("src_bank") != pl.col("dst_bank")),
                      fx=(pl.col("cur_recv") != pl.col("cur_paid")))
    )


def _to_usd(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize amounts to USD using exchange rates implied by the data's own USD<->X transactions."""
    rates = (lf.filter((pl.col("cur_recv") == "US Dollar") & (pl.col("cur_paid") != "US Dollar"))
             .group_by("cur_paid").agg((pl.col("amt_recv") / pl.col("amt_paid")).median().alias("r"))
             .collect())
    m = dict(zip(rates["cur_paid"], rates["r"])) | {"US Dollar": 1.0}
    return lf.with_columns(
        amt_paid=pl.col("amt_paid") * pl.col("cur_paid").replace_strict(m, default=None),
        amt_recv=pl.col("amt_recv") * pl.col("cur_recv").replace_strict(m, default=None),
    )
