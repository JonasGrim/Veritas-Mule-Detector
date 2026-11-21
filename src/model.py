"""Model, base features, hand rule and alert simulation."""
import numpy as np, polars as pl, lightgbm as lgb

KS = {"low": 180, "high": 300}  # alerts/day = 6 analysts x 30 / x 50
FE = ["log_in", "log_out", "passthru", "keep", "lag_h", "new_share", "new_snd", "new_small", "hub_share", "age_h", "log_prior",
      "log_surge", "sh_vic", "sh_exit", "in_snd", "out_pay", "in_n", "out_n", "log_mean_in"]


def prep(f):
    return f.with_columns(log_in=(pl.col("in_usd") + 1).log10(), log_out=(pl.col("out_usd") + 1).log10(),
                          log_prior=(pl.col("prior_n") + 1).log10(), log_surge=(pl.col("surge") + 1e-3).log10(),
                          log_mean_in=((pl.col("in_usd") / pl.col("in_n")) + 1).log10()).with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))


def rule(f):
    """Hand-written pass-through rule (baseline)."""
    vol = pl.min_horizontal("in_usd", "out_usd")
    s_vol = ((vol / 1000).log10() / 1.5).clip(0, 1); s_pass = ((pl.col("passthru") - 0.5) / 0.5).clip(0, 1)
    s_lag = (-pl.col("lag_h") / 24).exp(); s_new = (pl.col("new_snd") / 5).clip(0, 1) * pl.col("new_share")
    s_sh = ((pl.col("sh_vic") + pl.col("sh_exit")) / 4).tanh()
    return (s_vol * s_pass * (0.4 + 0.2 * s_lag + 0.15 * s_new + 0.1 * s_sh + 0.1 * pl.col("hub_share") + 0.05 * (pl.col("age_h") <= 96).cast(pl.Float64))).fill_null(0.0)


def fit(X, y, Xv=None, yv=None, seed=0):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, random_state=seed, n_jobs=2)
    if Xv is not None and yv.sum() > 0:
        m.fit(X, y, eval_set=[(Xv, yv)], eval_metric="average_precision", callbacks=[lgb.early_stopping(40, verbose=False)])
    else:
        m.fit(X, y)
    return m


def simulate(f, col, K, days):
    """Daily top-K alerts; an account already alerted is skipped on later days. Returns {acct: day alerted}."""
    seen = {}
    for d in days:
        s = f.filter(pl.col("d") == d)
        if seen: s = s.filter(~pl.col("acct").is_in(list(seen)))
        for a in s.top_k(K, by=col)["acct"].to_list(): seen[a] = d
    return seen
