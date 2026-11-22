"""Train and score the mule detector (full_r3) and baselines on one dataset.

  python src/pipeline.py --data data/himed_sep10.parquet --last-day 10 --test 8 9 10 --mode full
  python src/pipeline.py --data data/himed_sep16.parquet --last-day 16 --test 11 12 13 14 15 16 --mode full --scale 4.0144

Protocol: daily causal 72h snapshots; train on labels from days 1-3, early stop on days 6-7, test on --test days.
Daily top-K alerts, accounts already alerted are skipped. Positive = account that both sent and received a
laundering transaction during the test days.
"""
import sys, os, time, argparse, json, numpy as np, polars as pl
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as feat, chain
from model import prep, FE, fit, simulate, rule, KS as KS0

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--last-day", type=int, default=10)
ap.add_argument("--test", type=int, nargs="+", default=[8, 9, 10]); ap.add_argument("--mode", default="full", choices=["full", "baselines"])
ap.add_argument("--scale", type=float, default=None, help="candidate ratio vs HI-Small; adds scaled budgets")
ap.add_argument("--scale-ref", default=None, help="HI-Small parquet: compute the ratio from mean test-day candidates")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default="results")
ap.add_argument("--save-alerts", action="store_true", help="write each detector's alert list (account, day, score) to <out>/alerts/")
A = ap.parse_args()
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
DAYS = range(1, A.last_day + 1); TEST = A.test
assert min(TEST) > 7 and max(TEST) <= A.last_day

base = pl.read_parquet(A.data)
assert base["ts"].max() < feat.START + timedelta(days=A.last_day), "data runs past --last-day"
b, pairs, fs, ids = feat.build_world(base); log("world", b.height)
allf = pl.concat([prep(feat.snap(b, pairs, fs, 24.0 * d)) for d in DAYS])
log("snaps", allf.height, "positive train snapshots", allf.filter(pl.col("d") <= 3)["ibm_mule"].sum())

KS = dict(KS0)
if A.scale_ref:
    rb = pl.read_parquet(A.scale_ref); rb_, rp, rf, _ = feat.build_world(rb)
    ns = [feat.snap(rb_, rp, rf, 24.0 * d).height for d in TEST]; nm = [allf.filter(pl.col("d") == d).height for d in TEST]
    A.scale = (sum(nm) / len(nm)) / (sum(ns) / len(ns)); log("candidates/test day ref", ns, "this", nm); del rb, rb_, rp, rf
if A.scale:
    KS.update(low_s=round(180 * A.scale), high_s=round(300 * A.scale)); log("scale", round(A.scale, 4), "budgets/day", KS)


def Xm(f, cols): return f.select(pl.col(cols).cast(pl.Float32)).to_numpy()


def oof(f, cols, seed=0):
    """Stage scores: out-of-fold (5 folds by account) on training days, train-day model elsewhere."""
    d_ = f["d"].to_numpy(); acct = f["acct"].to_numpy(); Xa = Xm(f, cols); y = f["ibm_mule"].to_numpy()
    istr = d_ <= 3; p = np.zeros(f.height); fold = acct % 5
    for k in range(5):
        idx = np.where(istr & (fold != k))[0]; pos = idx[y[idx] == 1]; neg = idx[y[idx] == 0]
        neg = np.random.default_rng(k + 10 * seed).choice(neg, size=int(len(neg) * 0.1), replace=False)
        sel = np.concatenate([pos, neg]); m = fit(Xa[sel], y[sel], seed=seed)
        te = np.where(istr & (fold == k))[0]; p[te] = m.predict_proba(Xa[te])[:, 1]
    idx = np.where(istr)[0]; pos = idx[y[idx] == 1]; neg = np.random.default_rng(99 + seed).choice(idx[y[idx] == 0], size=int((~(y[idx] == 1)).sum() * 0.1), replace=False)
    sel = np.concatenate([pos, neg]); tu = np.where(np.isin(d_, [6, 7]))[0]
    m = fit(Xa[sel], y[sel], Xa[tu], y[tu], seed=seed); p[~istr] = m.predict_proba(Xa[~istr])[:, 1]
    return p


# ---------- per-day graph features: neighbour raw features + motifs
RAW = ["up_pt_n", "up_pt_amt", "dn_pt_n", "dn_pt_amt", "up_pass_max", "dn_pass_max", "up_new_w", "dn_age_min", "up_age_min"]
MOT = ["cyc2", "cyc3", "sg_max", "match_n", "match_share", "match_amt"]
def graph_feats(d):
    d_end = 24.0 * d; ws = d_end - feat.W
    w = b.filter((pl.col("t_h") >= ws) & (pl.col("t_h") < d_end) & (pl.col("sid") != pl.col("did")))
    a = allf.filter(pl.col("d") == d).select("acct", "passthru", "new_share", "age_h", "in_n", "out_n")
    e = w.group_by("sid", "did").agg(amt=pl.col("amt").sum())
    P = a.select("acct", "passthru", "new_share", "age_h")
    eu = e.join(P.rename({"acct": "sid", "passthru": "q", "new_share": "nsh", "age_h": "ag"}), on="sid", how="left")
    up = eu.group_by(acct=pl.col("did")).agg(up_pt_n=pl.col("q").is_not_null().sum(), up_pt_amt=pl.col("amt").filter(pl.col("q").is_not_null()).sum() / pl.col("amt").sum(),
        up_pass_max=pl.col("q").max(), up_new_w=(pl.col("nsh") * pl.col("amt")).sum() / pl.col("amt").sum(), up_age_min=pl.col("ag").min())
    ed = e.join(P.rename({"acct": "did", "passthru": "q", "age_h": "ag"}).drop("new_share"), on="did", how="left")
    dn = ed.group_by(acct=pl.col("sid")).agg(dn_pt_n=pl.col("q").is_not_null().sum(), dn_pt_amt=pl.col("amt").filter(pl.col("q").is_not_null()).sum() / pl.col("amt").sum(),
        dn_pass_max=pl.col("q").max(), dn_age_min=pl.col("ag").min())
    # motifs on accounts with <= 60 in and out transactions
    small = a.filter((pl.col("in_n") <= 60) & (pl.col("out_n") <= 60)).select("acct")
    es = e.select("sid", "did")
    c2 = es.join(es.rename({"sid": "did", "did": "sid"}), on=["sid", "did"], how="inner").group_by(acct=pl.col("sid")).agg(cyc2=pl.len())
    e1 = es.join(small.rename({"acct": "did"}), on="did", how="inner")
    p2 = e1.join(es.rename({"sid": "did", "did": "c"}), on="did", how="inner").filter(pl.col("c") != pl.col("sid")).rename({"sid": "A", "did": "B", "c": "C"})
    cyc3 = p2.join(es.rename({"sid": "C", "did": "A"}), on=["C", "A"], how="inner").group_by(acct=pl.col("B")).agg(cyc3=pl.len())
    sg = p2.group_by("A", "C").agg(n=pl.col("B").n_unique()).filter(pl.col("n") >= 2)
    sgm = p2.join(sg, on=["A", "C"], how="inner").group_by(acct=pl.col("B")).agg(sg_max=pl.col("n").max())
    # matched forwarding: inflow followed within 48h by an outflow of 80-100% of the amount
    ti = w.join(small.rename({"acct": "did"}), on="did", how="inner").select(acct="did", ti="t_h", ai="amt").with_row_index("k")
    to = w.join(small.rename({"acct": "sid"}), on="sid", how="inner").select(acct="sid", to="t_h", ao="amt")
    m = ti.join(to, on="acct", how="inner").filter((pl.col("to") > pl.col("ti")) & (pl.col("to") <= pl.col("ti") + 48) & ((pl.col("ao") / pl.col("ai")).is_between(0.8, 1.0)))
    mt = m.group_by("acct").agg(match_n=pl.col("k").n_unique(), match_amt=pl.col("ai").max())
    out = a.select("acct", "in_n").join(up, on="acct", how="left").join(dn, on="acct", how="left").join(c2, on="acct", how="left").join(cyc3, on="acct", how="left") \
        .join(sgm, on="acct", how="left").join(mt, on="acct", how="left").with_columns(match_share=pl.col("match_n") / pl.col("in_n"))
    return out.drop("in_n").with_columns(pl.col(RAW + MOT).fill_null(0.0).cast(pl.Float32), d=pl.lit(d, pl.Int8))
gf = pl.concat([graph_feats(d) for d in DAYS]); log("graph feats")
allf = allf.join(gf, on=["acct", "d"], how="left").sort(["d", "acct"]); del gf

# ---------- evaluation
t0 = feat.START + timedelta(days=min(TEST) - 1); t1 = feat.START + timedelta(days=max(TEST))
bt = base.filter((pl.col("is_laundering") == 1) & (pl.col("ts") >= t0) & (pl.col("ts") < t1))
s2i = dict(zip(ids["a"], ids["i"])); pos = set(bt["src"].replace_strict(s2i).to_list()) & set(bt["dst"].replace_strict(s2i).to_list())
del base, bt
log("test days", TEST, f"[{t0:%b %d}, {t1:%b %d})", "positives", len(pos), "test accounts", allf.filter(pl.col("d").is_in(TEST))["acct"].n_unique())


TAG = f"{os.path.splitext(os.path.basename(A.data))[0]}_d{min(TEST)}-{max(TEST)}"
i2s = ids.sort("i")["a"]


def save_alerts(f, col, al, name, kn):
    """Alert list for one detector and budget: original account id, day alerted, score, rank within the day, true label."""
    d = pl.DataFrame({"acct": list(al), "d": list(al.values())}, schema={"acct": f["acct"].dtype, "d": pl.Int8})
    d = d.join(f.select("acct", "d", score=pl.col(col).cast(pl.Float64)), on=["acct", "d"], how="left")
    d = d.with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("d").cast(pl.Int32),
                       mule=pl.col("acct").is_in(list(pos)).cast(pl.Int8),
                       account=pl.col("acct").map_elements(lambda i: i2s[int(i)], return_dtype=pl.Utf8))
    os.makedirs(f"{A.out}/alerts", exist_ok=True)
    d.select("account", "d", "rank", "score", "mule").sort(["d", "rank"]).write_csv(f"{A.out}/alerts/{TAG}_{name}_{kn}.csv")


def evaluate(f, col, name=None):
    r = {}
    for kn, K in KS.items():
        al = simulate(f, col, K, TEST); hit = set(al) & pos
        r[kn] = len(hit) / len(pos); r[kn + "_prec"] = len(hit) / len(al); r[kn + "_alerts"] = len(al)
        if A.save_alerts and name: save_alerts(f, col, al, name, kn)
    return r


def with_chain(f, score, prefix=""):
    parts = [chain.chain_feats(b, 24.0 * d, f.filter(pl.col("d") == d).select("acct", "d").with_columns(p0=pl.Series(score[(f["d"] == d).to_numpy()])))
             for d in DAYS]
    cf = pl.concat(parts).select(["acct", "d"] + chain.CH)
    if prefix: cf = cf.rename({c: prefix + c for c in chain.CH})
    return f.join(cf, on=["acct", "d"], how="left").sort(["d", "acct"])


def final(f, cols, seed, name=None):
    tr = f.filter(pl.col("d") <= 3); tu = f.filter(pl.col("d").is_in([6, 7]))
    rng = np.random.default_rng(seed); y = tr["ibm_mule"].to_numpy(); keep = (y == 1) | (rng.random(len(y)) < 0.1); trs = tr.filter(pl.Series(keep))
    m = fit(Xm(trs, cols), trs["ibm_mule"].to_numpy(), Xm(tu, cols), tu["ibm_mule"].to_numpy(), seed=seed)
    te = f.filter(pl.col("d").is_in(TEST)); te = te.with_columns(s=pl.Series(m.predict_proba(Xm(te, cols))[:, 1]))
    return evaluate(te, "s", name)


FULL = FE + chain.CH + RAW + MOT
def pipeline(cols2, rounds, seed, f0, name=None):
    """full_r3: stage 1 on base features, then chain features + 2 more propagation rounds."""
    p = oof(f0, FE, seed); f = with_chain(f0, p); cols = list(cols2)
    for r in range(rounds):
        p = oof(f, cols, seed); pre = f"r{r+2}_"; f = with_chain(f, p, pre); cols = cols + [pre + c for c in chain.CH]
    return final(f, cols, seed, name)


res = []
short = lambda r: {k: round(v, 3) for k, v in r.items() if k in KS}
if A.mode == "full":
    for seed in range(A.seeds):
        r = pipeline(FULL, 2, seed, allf, f"full_r3_s{seed}"); r.update(variant="full_r3", seed=seed); res.append(r); log("full_r3", seed, short(r))
else:
    te = allf.filter(pl.col("d").is_in(TEST))
    te = te.with_columns(rand=pl.Series(np.random.default_rng(0).random(te.height)), vol=pl.min_horizontal("in_usd", "out_usd"), rule=rule(te))
    for c in ("rand", "vol", "rule"):
        r = evaluate(te, c, c); r.update(variant=c, seed=0); res.append(r); log(c, short(r))
    # best single feature, picked on the test days (oracle choice, favours the baseline)
    best = None
    for c in FE + RAW + MOT:
        for sgn in (1, -1):
            t = te.with_columns(z=(pl.col(c).cast(pl.Float64).fill_null(-1e18 if sgn == 1 else 1e18) * sgn))
            v = len(set(simulate(t, "z", 300, TEST)) & pos) / len(pos)
            if best is None or v > best[0]: best = (v, c, sgn)
    _, c, sgn = best; t = te.with_columns(z=(pl.col(c).cast(pl.Float64).fill_null(-1e18 if sgn == 1 else 1e18) * sgn))
    r = evaluate(t, "z"); r.update(variant=f"best_single:{c}", seed=0); res.append(r); log("best single", c, short(r))
    for seed in range(A.seeds):
        r = final(allf, FE, seed, f"base_model_s{seed}"); r.update(variant="base_model", seed=seed); res.append(r); log("base", seed, short(r))
    rng = np.random.default_rng(7); y = allf["ibm_mule"].to_numpy().copy(); d_ = allf["d"].to_numpy()
    for d in range(1, 8):
        i = np.where(d_ == d)[0]; y[i] = rng.permutation(y[i])
    r = pipeline(FULL, 2, 0, allf.with_columns(ibm_mule=pl.Series(y, dtype=pl.Int8))); r.update(variant="placebo_full_r3", seed=0); res.append(r); log("placebo", short(r))

R = pl.DataFrame(res); os.makedirs(A.out, exist_ok=True)
tag = f"{TAG}_{A.mode}"
R.write_csv(f"{A.out}/{tag}.csv")
num = [c for c in R.columns if R[c].dtype in (pl.Float64, pl.Int64) and c != "seed"]
pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
print(R.group_by("variant", maintain_order=True).agg([pl.col(c).mean().round(3) for c in num]))
