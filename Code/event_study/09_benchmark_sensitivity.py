"""
09_benchmark_sensitivity.py — the twelve-day war's contaminated benchmark, on record.
====================================================================================
WHY. The war's [-140,-21] estimation window (2024-11-29 to 2025-05-15, 120 trading
days) contains all 41 trading days of the Liberation Day [-20,+20] event window
(2025-03-05 to 2025-04-30). Liberation Day is a mean shift inside the war's benchmark
window, so the constant-mean benchmark is contaminated: its mean is dragged down and
its standard deviation inflated. In the difference Liberation Day - war, Liberation
Day's abnormal returns enter twice, directly through the Liberation Day CAR and again
through the war's benchmark mean, and the printed gap is overstated.

The thesis therefore reports the war descriptively (Table 6.1 row, Table 6.2 column,
Section 6.2) with the caveat of Section 6.5, computes the Liberation Day - war pair in
05_cross_event_tests.py for the record (cross_event_diff.csv) and does not print it in
Table 6.3 or in the difference figure. This script reproduces the numbers quoted in
Section 6.5 and shows the pair's row under three benchmarks. The thesis prints none of
the cleaned rows. They are a sensitivity record, not a corrected estimate.

BENCHMARKS (broad dollar index r_DTWEXBGS, constant-mean model, conventions of
03/05: sample standard deviation, L = h+1, normal critical values 1.65/1.96/2.58).
Liberation Day's own CARs always use Liberation Day's own [-140,-21] benchmark.
  pipeline   the war's own [-140,-21] window, 120 days, the benchmark printed everywhere
  excl_ld    the same window without the 41 Liberation Day event-window days, 79 days
  shared     the 120-day window ending 21 trading days before Liberation Day's day 0,
             which is Liberation Day's own [-140,-21] estimation window, so under it
             both events share one benchmark

OUTPUT  Data/processed/event_study/benchmark_sensitivity.csv, 12 rows (3 benchmarks x
        4 horizons), one row per (benchmark, window). Pinned by verify_results.py.
CHECKS  (hard-fail)
  - the pipeline rows equal the pair's rows in cross_event_diff.csv (diff to 3 decimals,
    t to 2 decimals), so this script and 05 agree on the printed convention;
  - the shared benchmark equals Liberation Day's own benchmark;
  - the Section 6.5 receipts reproduce: benchmark mean +0.014 (excl_ld) vs -0.027
    (pipeline) percent per day, war CAR(0,20) -0.50 vs +0.35 percent.

USAGE   python3 Code/event_study/09_benchmark_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from es_common import ES_OUT, EST_WIN_MAIN, EVT_WIN_MAIN, load_data, load_events, stars  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SERIES = "r_DTWEXBGS"
WAR, LD = "iran_12day", "liberation_day"
HORIZONS = [1, 5, 10, 20]
OUTFILE = ES_OUT / "benchmark_sensitivity.csv"


def positions(idx, date):
    """Day-0 position, estimation-window index and event-window index (as in 03/05)."""
    pos = idx.searchsorted(pd.Timestamp(date))
    est = idx[max(0, pos + EST_WIN_MAIN[0]): pos + EST_WIN_MAIN[1] + 1]
    evt = idx[pos + EVT_WIN_MAIN[0]: pos + EVT_WIN_MAIN[1] + 1]
    return pos, est, evt


def benchmark(r, est_idx):
    est = r.loc[est_idx].dropna()
    return {"start": est.index[0].date(), "end": est.index[-1].date(),
            "n": int(len(est)), "mu": float(est.mean()), "sd": float(est.std())}


def car(r, pos, mu, h):
    """CAR(0,h) = sum of (r_t - mu) over days 0..h, exactly as 05_cross_event_tests.py."""
    return float((r.iloc[pos: pos + h + 1] - mu).sum())


def main():
    data = load_data()
    _, dates = load_events()
    r, idx = data[SERIES], data.index

    pos_w, est_w, _ = positions(idx, dates[WAR])
    pos_l, est_l, evt_l = positions(idx, dates[LD])
    inside = est_w.intersection(evt_l)
    assert len(evt_l) == 41 and len(inside) == 41, (len(evt_l), len(inside))

    shared_idx = idx[pos_l + EST_WIN_MAIN[0]: pos_l + EST_WIN_MAIN[1] + 1]
    benches = {"pipeline": benchmark(r, est_w),
               "excl_ld": benchmark(r, est_w.difference(evt_l)),
               "shared": benchmark(r, shared_idx)}
    ld = benchmark(r, est_l)
    assert benches["shared"] == ld, "shared benchmark must equal Liberation Day's own"
    pipe = benches["pipeline"]

    rows = []
    for name, b in benches.items():
        for h in HORIZONS:
            L = h + 1
            cw = car(r, pos_w, b["mu"], h)
            tw = cw / (b["sd"] * np.sqrt(L))
            cl = car(r, pos_l, ld["mu"], h)
            diff = cl - cw
            se = np.sqrt(L * (ld["sd"] ** 2 + b["sd"] ** 2))
            t = diff / se
            rows.append({
                "benchmark": name, "window": f"(0,{h})",
                "est_start": b["start"], "est_end": b["end"], "n_est": b["n"],
                "mu_pct_per_day": round(b["mu"] * 100, 4),
                "sd_pct_per_day": round(b["sd"] * 100, 4),
                "mu_shift_vs_pipeline_pp_per_day": round((b["mu"] - pipe["mu"]) * 100, 4),
                "sd_ratio_pipeline_over_this": round(pipe["sd"] / b["sd"], 4),
                "war_CAR_%": round(cw * 100, 3), "war_t": round(tw, 2), "war_sig": stars(tw),
                "LD_CAR_%": round(cl * 100, 3),
                "diff_%": round(diff * 100, 3), "se_pp": round(se * 100, 3),
                "t": round(t, 2), "sig": stars(t),
            })
    res = pd.DataFrame(rows)

    # ---- checks against the pinned 05 output and the Section 6.5 receipts ----------
    ced = pd.read_csv(ES_OUT / "cross_event_diff.csv")
    pair = ced[(ced.series == SERIES) & (ced.pair == f"{LD}-{WAR}")].set_index("window")
    for h in HORIZONS:
        mine = res[(res.benchmark == "pipeline") & (res.window == f"(0,{h})")].iloc[0]
        ref = pair.loc[f"(0,{h})"]
        assert mine["diff_%"] == ref["diff_%"] and mine["t"] == ref["t"], (h, mine, ref)
    mu_pipe, mu_excl = round(pipe["mu"] * 100, 3), round(benches["excl_ld"]["mu"] * 100, 3)
    car20_pipe = round(car(r, pos_w, pipe["mu"], 20) * 100, 2)
    car20_excl = round(car(r, pos_w, benches["excl_ld"]["mu"], 20) * 100, 2)
    assert (mu_pipe, mu_excl) == (-0.027, 0.014), (mu_pipe, mu_excl)
    assert (car20_pipe, car20_excl) == (0.35, -0.50), (car20_pipe, car20_excl)

    res.to_csv(OUTFILE, index=False)
    print(f"Saved {OUTFILE.relative_to(ROOT)} ({len(res)} rows)\n")
    print("Benchmarks for the twelve-day war, broad dollar index (percent per day):")
    for name, b in benches.items():
        print(f"  {name:9s} {b['start']} to {b['end']}  n={b['n']:3d}  "
              f"mean {b['mu']*100:+.4f}  sd {b['sd']*100:.4f}")
    print(f"  Liberation Day event-window days inside the war's window: {len(inside)}; "
          f"mean shift excl_ld - pipeline {(benches['excl_ld']['mu']-pipe['mu'])*100:+.3f} pp/day; "
          f"sd ratio pipeline/excl_ld {pipe['sd']/benches['excl_ld']['sd']:.3f}")
    print(f"  Section 6.5 receipts: mean {mu_pipe:+.3f} vs {mu_excl:+.3f}, "
          f"war CAR(0,20) {car20_pipe:+.2f} vs {car20_excl:+.2f} percent\n")
    with pd.option_context("display.width", 160):
        print(res[["benchmark", "window", "n_est", "war_CAR_%", "war_t", "war_sig",
                   "diff_%", "se_pp", "t", "sig"]].to_string(index=False))
    print("\nChecks passed: pipeline rows == cross_event_diff.csv, shared == Liberation Day "
          "benchmark, Section 6.5 receipts reproduced. The thesis prints none of the "
          "excl_ld or shared rows.")


if __name__ == "__main__":
    main()
