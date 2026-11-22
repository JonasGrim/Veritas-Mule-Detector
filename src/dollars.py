"""Dollar view of the alert lists written by `pipeline.py --save-alerts`.

  python src/dollars.py --data data/himed_sep10.parquet --test 8 9 10
  python src/dollars.py --data data/himed_sep16.parquet --test 11 12 13 14 15 16

For each detector and daily budget, sums the laundering money that mule accounts sent on during the test days:
  - moved by all mules: every laundering outflow from a positive account (the total at stake);
  - moved by caught mules: the same, for mules the detector alerted on;
  - after the alert: only outflows sent after the account was alerted, i.e. money that could have been
    stopped if the account had been restricted the moment it was alerted. This is the conservative number.
An alert for day d is issued at the end of day d (midnight). Amounts are USD. A dollar that passes through
three mules counts three times, once per mule that sent it on.
"""
import argparse, glob, os, re
from datetime import timedelta
import polars as pl
import features as feat

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--test", type=int, nargs="+", required=True)
ap.add_argument("--out", default="results")
A = ap.parse_args()
TEST = A.test
TAG = f"{os.path.splitext(os.path.basename(A.data))[0]}_d{min(TEST)}-{max(TEST)}"
t0 = feat.START + timedelta(days=min(TEST) - 1); t1 = feat.START + timedelta(days=max(TEST))

bt = (pl.scan_parquet(A.data).filter((pl.col("is_laundering") == 1) & (pl.col("ts") >= t0) & (pl.col("ts") < t1))
      .select("ts", "src", "dst", "amt").collect())
pos = set(bt["src"].to_list()) & set(bt["dst"].to_list())  # same positive definition as pipeline.py
out = bt.filter(pl.col("src").is_in(list(pos)) & (pl.col("src") != pl.col("dst"))).rename({"src": "account"})
total = out["amt"].sum()
per = out.group_by("account").agg(usd=pl.col("amt").sum()).sort("usd", descending=True)
print(f"{TAG}: {len(pos)} mules sent on ${total:,.0f} of laundering money during the test days")
print(f"  concentration: top 10 mules = {per['usd'].head(10).sum() / total:.0%} of it; median mule ${per['usd'].median():,.0f}."
      " Totals are dominated by a few huge accounts, so compare detectors on the median and on money moved after the alert.")

rows = []
for p in sorted(glob.glob(f"{A.out}/alerts/{TAG}_*.csv")):
    m = re.match(rf"{re.escape(TAG)}_(.+?)(?:_s(\d+))?_(low_s|high_s|low|high)\.csv$", os.path.basename(p))
    if not m: continue
    det, seed, kn = m.group(1), int(m.group(2) or 0), m.group(3)
    al = pl.read_csv(p, schema_overrides={"account": pl.Utf8})
    caught = al.filter(pl.col("mule") == 1).select("account", alert_ts=pl.lit(feat.START) + pl.duration(days=pl.col("d")))
    o = out.join(caught, on="account", how="inner")
    rows.append(dict(detector=det, seed=seed, budget=kn, alerts=al.height, mules_caught=caught.height,
                     usd_caught=o["amt"].sum(), usd_after_alert=o.filter(pl.col("ts") >= pl.col("alert_ts"))["amt"].sum(),
                     median_usd_per_caught_mule=o.group_by("account").agg(u=pl.col("amt").sum())["u"].median() or 0.0))
if not rows:
    raise SystemExit(f"no alert files for {TAG} in {A.out}/alerts; run pipeline.py with --save-alerts first")

R = pl.DataFrame(rows)
S = (R.group_by("detector", "budget", maintain_order=True).agg(pl.len().alias("seeds"), pl.col("alerts").mean(),
     pl.col("mules_caught").mean(), pl.col("usd_caught").mean(), pl.col("usd_after_alert").mean(), pl.col("median_usd_per_caught_mule").mean())
     .with_columns(share_caught=pl.col("usd_caught") / total, share_after_alert=pl.col("usd_after_alert") / total,
                   usd_after_per_alert=pl.col("usd_after_alert") / pl.col("alerts"), usd_total=pl.lit(total)))
S.write_csv(f"{A.out}/dollars_{TAG}.csv")
pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(3)
print(S.select("detector", "budget", "seeds", "alerts", "mules_caught", "median_usd_per_caught_mule", "usd_caught",
               "usd_after_alert", "share_after_alert", "usd_after_per_alert"))
