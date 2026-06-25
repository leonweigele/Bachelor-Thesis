"""
plot_classification.py — figure of the safe/risky currency classification.
==========================================================================
Horizontal bar chart of each currency's average 1-month forward discount
(annualised) -- the carry signal that defines the buckets. Low (funding) =
safe, high (carry) = risky. Styled via thesis_style for a consistent look.

Input : Data/processed/carry_classification.csv
Output: Output/figures/fig_classification.(png|pdf)
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from thesis_style import apply_style, save_fig

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "Data/processed/carry_classification.csv").sort_values("avg_fd_ann")

apply_style()
COLORS = {"safe": "#2e6da4", "mid": "#9aa0a6", "risky": "#c0392b"}
bar_colors = [COLORS[b] for b in df["bucket"]]

fig, ax = plt.subplots(figsize=(5.0, 5.6))
ax.barh(df["currency"], df["avg_fd_ann"] * 100, color=bar_colors, height=0.72)
ax.axvline(0, color="black", lw=0.6)

ax.set_xlabel("Average 1-month forward discount, annualised")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.xaxis.grid(True, color="#e6e6e6", lw=0.6)
ax.yaxis.grid(False)
ax.tick_params(width=0.6, length=3)
ax.margins(y=0.01)

handles = [mpatches.Patch(color=COLORS[b], label=l) for b, l in
           [("safe", "Safe (low-yield / funding)"),
            ("mid", "Middle tercile"),
            ("risky", "Risky (high-yield / carry)")]]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7)

save_fig(fig, ROOT / "Output/figures/fig_classification")
print("saved Output/figures/fig_classification.(png|pdf)")
