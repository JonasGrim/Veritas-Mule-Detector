"""Account features from causal 72h snapshots.

build_world: integer account ids, hours since Sept 1, first-seen time per account and per counterparty pair.
snap(d_end): accounts with both inflow and outflow in [d_end - 72h, d_end). Uses only transactions before d_end.
"""
import polars as pl
from datetime import datetime

START = datetime(2022, 9, 1)
W = 72.0  # snapshot window, hours


def build_world(base):
    b = base.select("ts", "src", "dst", "amt", "fmt", lab="is_laundering").sort("ts")
    ids = pl.concat([b["src"], b["dst"]]).unique().sort().to_frame("a").with_row_index("i")
    b = (b.join(ids.rename({"a": "src", "i": "sid"}), on="src").join(ids.rename({"a": "dst", "i": "did"}), on="dst")
         .with_columns(t_h=((pl.col("ts") - START).dt.total_seconds() / 3600).cast(pl.Float32), amt=pl.col("amt").cast(pl.Float32))
         .select("t_h", "sid", "did", "amt", "fmt", "lab"))
    ns = b.filter(pl.col("sid") != pl.col("did"))
    pairs = ns.group_by("sid", "did").agg(pf=pl.col("t_h").min())
    b = b.join(pairs, on=["sid", "did"], how="left")
    legs = pl.concat([b.select(acct="sid", t="t_h"), b.select(acct="did", t="t_h")])
    fs = legs.group_by("acct").agg(first=pl.col("t").min())
    return b, pairs, fs, ids


def snap(b, pairs, fs, d_end):
    ws = d_end - W
    w = b.filter((pl.col("t_h") >= ws) & (pl.col("t_h") < d_end) & (pl.col("sid") != pl.col("did")))
    hubs = pairs.filter(pl.col("pf") < d_end).group_by("did").agg(ns=pl.len()).filter(pl.col("ns") >= 20).select("did", hub=pl.lit(1.0, pl.Float32))
    w = w.join(hubs, on="did", how="left").with_columns(pl.col("hub").fill_null(0.0))
    new = (pl.col("pf") >= ws)
    ins = w.group_by(acct=pl.col("did")).agg(
        in_usd=pl.col("amt").sum(), in_n=pl.len(), in_snd=pl.col("sid").n_unique(),
        in_cent=(pl.col("amt") * pl.col("t_h")).sum(),
        new_usd=pl.col("amt").filter(new).sum(), new_snd=pl.col("sid").filter(new).n_unique(),
        new_small=pl.col("sid").filter(new & (pl.col("amt") < 3000)).n_unique(),
        lab_in=pl.col("lab").max())
    outs = w.group_by(acct=pl.col("sid")).agg(
        out_usd=pl.col("amt").sum(), out_n=pl.len(), out_pay=pl.col("did").n_unique(),
        out_cent=(pl.col("amt") * pl.col("t_h")).sum(), hub_usd=(pl.col("amt") * pl.col("hub")).sum(),
        lab_out=pl.col("lab").max())
    a = ins.join(outs, on="acct", how="inner")
    a = a.with_columns(
        lag_h=((pl.col("out_cent") / pl.col("out_usd")) - (pl.col("in_cent") / pl.col("in_usd"))).abs(),
        passthru=pl.min_horizontal("in_usd", "out_usd") / pl.max_horizontal("in_usd", "out_usd"),
        keep=((pl.col("in_usd") - pl.col("out_usd")) / pl.col("in_usd")),
        new_share=pl.col("new_usd") / pl.col("in_usd"),
        hub_share=pl.col("hub_usd") / pl.col("out_usd"))
    # candidates for shared-counterparty signals
    cand = a.filter((pl.min_horizontal("in_usd", "out_usd") >= 2000) & (pl.col("passthru") >= 0.5)).select("acct").with_columns(c=pl.lit(1))
    # shared senders: senders paying >= 2 candidates
    wc = w.join(cand.rename({"acct": "did"}), on="did", how="inner")
    sc = wc.group_by("sid").agg(nc=pl.col("did").n_unique())
    shv = wc.join(sc, on="sid").filter(pl.col("nc") >= 2).group_by(acct=pl.col("did")).agg(sh_vic=pl.col("sid").n_unique())
    # shared payees: payees paid by >= 2 candidates
    wo = w.join(cand.rename({"acct": "sid"}), on="sid", how="inner")
    pc = wo.group_by("did").agg(nc=pl.col("sid").n_unique())
    sho = wo.join(pc, on="did").filter(pl.col("nc") >= 2).group_by(acct=pl.col("sid")).agg(sh_exit=pl.col("nc").max() - 1)
    a = a.join(shv, on="acct", how="left").join(sho, on="acct", how="left").with_columns(pl.col("sh_vic").fill_null(0), pl.col("sh_exit").fill_null(0))
    # prior history
    pr = b.filter((pl.col("t_h") < ws) & (pl.col("sid") != pl.col("did")))
    prior = pl.concat([pr.select(acct="sid", amt="amt"), pr.select(acct="did", amt="amt")]).group_by("acct").agg(prior_n=pl.len(), prior_usd=pl.col("amt").sum())
    a = a.join(prior, on="acct", how="left").join(fs, on="acct", how="left").with_columns(
        pl.col("prior_n").fill_null(0), pl.col("prior_usd").fill_null(0.0), age_h=d_end - pl.col("first"))
    a = a.with_columns(surge=(pl.col("in_usd") + pl.col("out_usd")) / ((pl.col("prior_usd") / max(ws / 24, 1.0)) * 3 + 1000.0))
    # training label: laundering both in and out within the window
    a = a.with_columns(ibm_mule=((pl.col("lab_in") == 1) & (pl.col("lab_out") == 1)).cast(pl.Int8),
                       ibm_any=((pl.col("lab_in") == 1) | (pl.col("lab_out") == 1)).cast(pl.Int8), d=pl.lit(int(round(d_end / 24)), pl.Int8))
    return a.drop("in_cent", "out_cent", "hub_usd", "new_usd", "lab_in", "lab_out", "first")
