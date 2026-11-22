"""Veritas in about two minutes: one seed on HI-Small, then the dollar view and three example alerts.

  pip install -r requirements.txt
  python demo.py

Downloads the prepared HI-Small file (about 90 MB, IBM AMLworld, CDLA-Sharing-1.0) on first run.
HI-Small is the development dataset, so recall here is optimistic; the clean results are in README.md.
"""
import os, subprocess, sys, time, urllib.request
from datetime import timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "hi_sep10.parquet")
URL = "https://github.com/JonasGrim/Veritas-Mule-Detector/releases/download/data-v1/hi_sep10.parquet"
OUT = os.path.join(ROOT, "results", "demo")
T0 = time.time()


def step(msg): print(f"\n[{time.time() - T0:5.0f}s] {msg}", flush=True)


def fetch():
    if os.path.exists(DATA): return
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    step(f"Downloading HI-Small (about 90 MB) from {URL}")
    try:
        urllib.request.urlretrieve(URL, DATA + ".part"); os.replace(DATA + ".part", DATA)
    except Exception as e:
        sys.exit(f"Download failed ({e}).\nInstead, download HI-Small_Trans.csv from Kaggle "
                 "(ealtman2019/ibm-transactions-for-anti-money-laundering-aml) into data/ and run:\n"
                 "  python src/prep_data.py data/HI-Small_Trans.csv 2022-09-11 data/hi_sep10.parquet")


def run(args):
    r = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True)
    if r.returncode: sys.exit(r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def explain(acct, day):
    """Plain-language evidence for one alert, from the 72 hours before it fired."""
    import polars as pl
    sys.path.insert(0, os.path.join(ROOT, "src")); import features as feat
    end = feat.START + timedelta(days=day); start = end - timedelta(hours=72)
    w = (pl.scan_parquet(DATA).filter((pl.col("ts") >= start) & (pl.col("ts") < end) & (pl.col("src") != pl.col("dst"))
         & ((pl.col("src") == acct) | (pl.col("dst") == acct))).collect())
    i = w.filter(pl.col("dst") == acct); o = w.filter(pl.col("src") == acct)
    lines = [f"received ${i['amt'].sum():,.0f} from {i['src'].n_unique()} sender(s) and sent ${o['amt'].sum():,.0f} "
             f"to {o['dst'].n_unique()} payee(s) in the 72h before the alert"]
    m = (i.select(ti="ts", ai="amt", frm="src").join(o.select(to="ts", ao="amt", to_acct="dst"), how="cross")
         .filter((pl.col("to") > pl.col("ti")) & (pl.col("to") <= pl.col("ti") + timedelta(hours=48))
                 & (pl.col("ao") / pl.col("ai")).is_between(0.8, 1.0))
         .with_columns(h=(pl.col("to") - pl.col("ti")).dt.total_minutes() / 60).sort("h"))
    if m.height:
        r = m.row(0, named=True)
        lines.append(f"forwarded {r['ao'] / r['ai']:.0%} of a ${r['ai']:,.0f} inflow {r['h']:.1f}h after receiving it "
                     f"({m.height} matched in-then-out pair{'s' if m.height > 1 else ''})")
    return lines


fetch()
step("Training and scoring: daily 72h snapshots, graph features, 3 rounds of score propagation (one seed)")
log = run(["src/pipeline.py", "--data", DATA, "--mode", "full", "--seeds", "1", "--save-alerts", "--out", OUT])
step("Dollar view")
dol = run(["src/dollars.py", "--data", DATA, "--test", "8", "9", "10", "--out", OUT])

import polars as pl
res = pl.read_csv(os.path.join(OUT, "hi_sep10_d8-10_full.csv")).row(0, named=True)
d = pl.read_csv(os.path.join(OUT, "dollars_hi_sep10_d8-10.csv")).filter(pl.col("budget") == "high").row(0, named=True)
al = pl.read_csv(os.path.join(OUT, "alerts", "hi_sep10_d8-10_full_r3_s0_high.csv"), schema_overrides={"account": pl.Utf8})

print("\n" + "=" * 72)
print("VERITAS DEMO: HI-Small, Sept 8-10, 6 analysts x 50 alerts/day = 900 alerts")
print("=" * 72)
n_mules = int(dol.splitlines()[0].split(": ", 1)[1].split()[0])
print(f"Mules caught:     {d['mules_caught']:.0f} of {n_mules} ({res['high']:.0%}), precision {res['high_prec']:.0%}, "
      f"{1 / res['high_prec']:.1f} alerts per mule found")
print(f"Dollars:          {dol.splitlines()[0].split(': ', 1)[1]}")
print(f"                  money these caught mules sent on AFTER being alerted: ${d['usd_after_alert']:,.0f}")
print("                  (most mules forward within hours, so a daily alert mostly arrives after the money moved)")
print("\nTop 3 alerts on Sept 8, with the evidence an analyst would see:")
for r in al.filter(pl.col("d") == 8).sort("rank").head(3).iter_rows(named=True):
    print(f"\n  #{r['rank']}  account {r['account']}  score {r['score']:.3f}  "
          f"[IBM label: {'mule' if r['mule'] else 'not a mule'}]")
    for line in explain(r["account"], r["d"]): print(f"      - {line}")
print(f"\nDone in {time.time() - T0:.0f}s. Alert lists and dollar table: results/demo/")
print("HI-Small is the development set (optimistic). The pre-registered HI-Medium results are in README.md.")
