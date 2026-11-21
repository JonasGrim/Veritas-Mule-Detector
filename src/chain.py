"""Neighbour-score features: how suspicious an account's senders and payees look (1 and 2 hops), from the previous stage's score."""
import polars as pl
import features

CH = ["ps_max", "ps_w", "ps_nhi", "pd_max", "pd_w", "pd_nhi", "ps2", "pd2", "max_in_e", "max_out_e", "e_ratio", "n_snd", "n_rcv", "ps_hi5", "pd_hi5", "ps2_hi", "pd2_hi"]


def chain_feats(b, d_end, a, W=features.W):
    ws = d_end - W
    w = b.filter((pl.col("t_h") >= ws) & (pl.col("t_h") < d_end) & (pl.col("sid") != pl.col("did")))
    e = w.group_by("sid", "did").agg(amt=pl.col("amt").sum())
    a = a.with_columns(pr=pl.col("p0").rank() / pl.len())  # percentile of the stage score within the snapshot
    P = a.select("acct", "pr")
    e = (e.join(P.rename({"acct": "sid", "pr": "p_s"}), on="sid", how="left").join(P.rename({"acct": "did", "pr": "p_d"}), on="did", how="left")
         .with_columns(pl.col("p_s").fill_null(0.0), pl.col("p_d").fill_null(0.0)))
    up = e.group_by(acct=pl.col("did")).agg(ps_max=pl.col("p_s").max(), ps_w=(pl.col("amt") * pl.col("p_s")).sum() / pl.col("amt").sum(),
        ps_nhi=(pl.col("p_s") > 0.99).sum(), ps_hi5=(pl.col("p_s") > 0.95).sum(), max_in_e=pl.col("amt").max(), n_snd=pl.len())
    dn = e.group_by(acct=pl.col("sid")).agg(pd_max=pl.col("p_d").max(), pd_w=(pl.col("amt") * pl.col("p_d")).sum() / pl.col("amt").sum(),
        pd_nhi=(pl.col("p_d") > 0.99).sum(), pd_hi5=(pl.col("p_d") > 0.95).sum(), max_out_e=pl.col("amt").max(), n_rcv=pl.len())
    q_up = up.select(sid="acct", q=pl.col("ps_max")); q_dn = dn.select(did="acct", q=pl.col("pd_max"))
    up2 = e.join(q_up, on="sid", how="left").group_by(acct=pl.col("did")).agg(ps2=pl.col("q").max(), ps2_hi=(pl.col("q") > 0.99).sum())
    dn2 = e.join(q_dn, on="did", how="left").group_by(acct=pl.col("sid")).agg(pd2=pl.col("q").max(), pd2_hi=(pl.col("q") > 0.99).sum())
    out = a.drop("pr").join(up, on="acct", how="left").join(dn, on="acct", how="left").join(up2, on="acct", how="left").join(dn2, on="acct", how="left")
    out = out.with_columns(e_ratio=pl.min_horizontal("max_in_e", "max_out_e") / pl.max_horizontal("max_in_e", "max_out_e"))
    return out.with_columns(pl.col(CH).fill_null(0.0))
