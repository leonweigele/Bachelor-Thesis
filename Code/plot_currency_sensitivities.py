"""
plot_currency_sensitivities.py — currency classification figure.
================================================================
Scatter of each currency's Dollar-factor loading (x) against the carry signal
that DEFINES the safe/risky buckets -- its average 1-month forward discount (y).
Dashed lines mark the tercile cut-offs, so colour (bucket) and vertical position
agree and the classification is readable directly from the graph:
low forward discount -> safe (funding), high -> risky (carry).

Dollar-factor loading b^DOL is from a time-series regression of each currency's
monthly excess return on the rebuilt Dollar (DOL) and Carry (CARRY) factors.

Inputs : Data/processed/fx_excess_returns_monthly.csv, fx_factors_monthly.csv,
         carry_classification.csv   (all from build_fx_factors.py)
Output : Output/figures/fig_currency_sensitivities.(png|pdf)
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thesis_style import apply_style, save_fig

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "Data/processed"

rx = pd.read_csv(PROC / "fx_excess_returns_monthly.csv", parse_dates=["date"], index_col="date")
fac = pd.read_csv(PROC / "fx_factors_monthly.csv", parse_dates=["date"], index_col="date")
cls = pd.read_csv(PROC / "carry_classification.csv").set_index("currency")

# Dollar-factor loading per currency (for the x-axis)
dollar_beta = {}
for c in cls.index:
    if c not in rx.columns:
        continue
    d = pd.concat([rx[c], fac["DOL"], fac["CARRY"]], axis=1).dropna()
    if len(d) < 24:
        continue
    X = np.column_stack([np.ones(len(d)), d["DOL"], d["CARRY"]])
    coef, *_ = np.linalg.lstsq(X, d[c].values, rcond=None)
    dollar_beta[c] = coef[1]

B = cls.copy()
B["dollar"] = pd.Series(dollar_beta)
B["fd"] = B["avg_fd_ann"] * 100        # annualised forward discount, %
B = B.dropna(subset=["dollar"])

# tercile cut-offs on the sort variable (midpoints between adjacent terciles)
s = B["fd"].sort_values().values
n = len(s)
cut_lo = (s[n // 3 - 1] + s[n // 3]) / 2          # safe | middle
cut_hi = (s[2 * n // 3 - 1] + s[2 * n // 3]) / 2  # middle | risky

apply_style()
MARKERS = {"safe": "o", "mid": "s", "risky": "^"}
LABELS = {"safe": "Safe (low-yield / funding)", "mid": "Middle tercile",
          "risky": "Risky (high-yield / carry)"}
fig, ax = plt.subplots(figsize=(7.6, 6.6))

ax.axhline(cut_lo, color="black", lw=0.5, ls="--", zorder=1)
ax.axhline(cut_hi, color="black", lw=0.5, ls="--", zorder=1)

for b in ("safe", "mid", "risky"):
    grp = B[B["bucket"] == b]
    ax.scatter(grp["dollar"], grp["fd"], s=26, marker=MARKERS[b],
               facecolor="black", edgecolor="black", linewidths=0.5,
               zorder=3, label=LABELS[b])
# Every label hugs its OWN marker. Default is just above-right; labels that
# would collide there are flipped to a free side of their own dot (set beside
# the dot at marker height, or directly below it) so each label clearly
# belongs to one point. (dx, dy, ha, va)
DEFAULT = (0.009, 0.35, "left", "bottom")
OFFSETS = {
    # bottom pair: label beside each dot at its own height
    "JPY": (-0.011, 0.0, "right", "center"),
    "CHF": (0.011, 0.0, "left", "center"),
    # mid-band: drop straight below their dots
    "ILS": (0.0, -0.95, "center", "bottom"),
    "EUR": (0.0, -0.95, "center", "bottom"),
    "THB": (0.012, -0.95, "left", "bottom"),
    "SEK": (0.0, -0.95, "center", "bottom"),   # below the gridline, clear of AUD
    # vertical stack on the right: label to the right of each dot, at its height
    "ZAR": (0.013, 0.0, "left", "center"),
    "HUF": (0.013, 0.0, "left", "center"),
    "PLN": (0.013, 0.0, "left", "center"),
    "CZK": (0.013, 0.0, "left", "center"),     # sits between the two gridlines
    "NZD": (0.0, 0.40, "center", "bottom"),    # above its dot (NOK goes up-right)
}
for c, r in B.iterrows():
    dx, dy, ha, va = OFFSETS.get(c, DEFAULT)
    ax.text(r["dollar"] + dx, r["fd"] + dy, c, fontsize=6.5,
            color="#333333", ha=ha, va=va)

ax.set_xlim(0.22, 1.64)   # a little right margin so NOK's label isn't clipped

for sp in ("left", "bottom"):
    ax.spines[sp].set_visible(True)
    ax.spines[sp].set_linewidth(0.6)
ax.grid(True, color="#e6e6e6", lw=0.6)
ax.set_axisbelow(True)
ax.tick_params(width=0.6, length=3)
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel(r"Dollar-factor loading, $\beta^{\mathrm{DOL}}$")
ax.set_ylabel("Average 1-month forward discount (annualised)")

ax.legend(loc="upper right", frameon=False, fontsize=7,
          handletextpad=0.3, labelspacing=0.3)

save_fig(fig, ROOT / "Output/figures/ch05_methodology/fig_currency_sensitivities")
print(f"saved fig_currency_sensitivities ({len(B)} currencies); "
      f"cut-offs at {cut_lo:.2f}% and {cut_hi:.2f}%")
