"""Controlled changes with known effects (App. G, summary figure).

(a) Share of held videos and routes flagged by each test when one property of an ideal model is changed by a known
    amount, from the per-item tables perturbations/m_held.csv and m_routes.csv. Decision rules: stride test = held pace
    ratio outside 0.8-1.25 / route not completed within one stride (category != near); Stall = route has a stalled step
    (category == stalled); Direction = some step not in the commanded direction (held: direction_rate < 1, routes:
    step_direction_acc < 1); ATE = position error after fitting one scale or a similarity transform above 10% of the
    path. Held rows use the videos measurable at step 16.
(b) Rigidity offset and VGGT reprojection error under known changes of the rendered scene
    (perturbations/perturb_ga_relative.csv: median change against the rigid twin, as a share of the change at 2 deg of
    smooth deformation).

usage: python3 scoring/perturbation_summary.py DATA_RELEASE_DIR OUT_DIR      writes flag_rates.csv, perturb_summary.pdf
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ITEMS = os.path.join(sys.argv[1], "perturbations")
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "Times New Roman", "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})
BLUE, INK, ORANGE = "#1F5FA8", "#1D2430", "#D9822B"

h = pd.read_csv(os.path.join(ITEMS, "m_held.csv"))
r = pd.read_csv(os.path.join(ITEMS, "m_routes.csv"))
h = h[h.measurable16 == True]  # noqa: E712


def pct(s):
    return 100.0 * s.mean()


def held_row(cond, amp, label):
    q = h[(h.condition == cond) & np.isclose(h.amplitude, amp)]
    return dict(group="held", label=label, n=len(q),
                stride=pct((q.pace_ratio < 0.8) | (q.pace_ratio > 1.25)), stall=np.nan,
                direction=pct(q.direction_rate < 1), ate_scale=pct(q.ate_scale_pct > 10),
                ate_sim3=pct(q.ate_sim3_pct > 10))


def route_row(cond, amp, label):
    q = r[(r.condition == cond) & np.isclose(r.amplitude, amp)]
    return dict(group="route", label=label, n=len(q),
                stride=pct(q.category != "near"), stall=pct(q.category == "stalled"),
                direction=pct(q.step_direction_acc < 1), ate_scale=pct(q.ate_scale_pct > 10),
                ate_sim3=pct(q.ate_sim3_pct > 10))


rows = [held_row("M0", 0.0, "Ideal"),
        held_row("M1", 1.5, "Pace grows to $1.5\\times$"),
        held_row("M1", 2.0, "Pace grows to $2\\times$"),
        route_row("M0", 0.0, "Ideal"),
        route_row("M2", 0.5, "Carry-over of $\\frac{1}{2}$ step"),
        route_row("M2", 1.0, "Carry-over of 1 step"),
        route_row("M4", 0.5, "First step $\\frac{1}{2}$ stride"),
        route_row("M4", 0.25, "First step $\\frac{1}{4}$ stride"),
        route_row("M3", 1.0, "Stall of 1 step"),
        route_row("M3", 2.0, "Stall of 2 steps")]
F = pd.DataFrame(rows)
F.to_csv(os.path.join(OUT, "flag_rates.csv"), index=False, float_format="%.1f")
print(F.round(1).to_string())

cols = [("stride", "Stride\ntest"), ("stall", "Stall"), ("direction", "Direc-\ntion"),
        ("ate_scale", "ATE\nscale"), ("ate_sim3", "ATE\nSim(3)")]
cmap = LinearSegmentedColormap.from_list("w2b", ["#FFFFFF", "#D6E2F3", "#7FA3D4", BLUE])

g = pd.read_csv(os.path.join(ITEMS, "perturb_ga_relative.csv"))

fig = plt.figure(figsize=(7.0, 3.1))
axa = fig.add_axes([0.205, 0.09, 0.30, 0.70])
axb = fig.add_axes([0.745, 0.09, 0.245, 0.70])

# (a) heatmap with a gap between the held and route blocks
ypos, y = [], 0.0
for i, row in F.iterrows():
    if i == 3:
        y += 0.45
    ypos.append(y)
    y += 1.0
ypos = np.array(ypos)
for i, row in F.iterrows():
    for j, (c, _) in enumerate(cols):
        v = row[c]
        if np.isnan(v):
            axa.add_patch(plt.Rectangle((j - 0.5, ypos[i] - 0.5), 1, 1, fc="#F3F4F6", ec="white", lw=1.2))
            axa.text(j, ypos[i], "–", ha="center", va="center", fontsize=7.2, color="#9AA1AC")
            continue
        axa.add_patch(plt.Rectangle((j - 0.5, ypos[i] - 0.5), 1, 1, fc=cmap(min(v, 100) / 100), ec="white", lw=1.2))
        axa.text(j, ypos[i], f"{v:.0f}", ha="center", va="center", fontsize=7.2,
                 color="white" if v >= 60 else INK)
axa.set_xlim(-0.5, len(cols) - 0.5)
axa.set_ylim(ypos[-1] + 0.5, -0.5)
axa.set_yticks(ypos)
axa.set_yticklabels(F.label)
axa.set_xticks(range(len(cols)))
axa.set_xticklabels([c[1] for c in cols], fontsize=7.2, linespacing=0.95)
axa.xaxis.tick_top()
axa.tick_params(length=0, pad=2)
for s in axa.spines.values():
    s.set_visible(False)
axa.axvline(1.5, color=INK, lw=0.8)
axa.text(0.5, -1.75, "stride-based", ha="center", va="center", fontsize=7.4, color=BLUE, fontweight="bold")
axa.text(3.0, -1.75, "existing", ha="center", va="center", fontsize=7.4, color="#6B7380", fontweight="bold")
axa.text(-4.15, ypos[1], "held", rotation=90, ha="center", va="center", fontsize=7.2, color="#6B7380")
axa.text(-4.15, ypos[6], "routes", rotation=90, ha="center", va="center", fontsize=7.2, color="#6B7380")
fig.text(0.005, 0.995, "(a) Share of videos and routes flagged (%)", fontsize=8.5, fontweight="bold", va="top")

# (b) world changes, as in perturb_ga
n = len(g)
ys = np.arange(n)[::-1].astype(float)
hgt = 0.36
for yy, (_, rr) in zip(ys, g.iterrows()):
    axb.barh(yy + hgt / 2 + 0.02, rr.offset, height=hgt, color=BLUE, label="Rigidity offset" if yy == ys[0] else None)
    axb.errorbar(rr.offset, yy + hgt / 2 + 0.02, xerr=[[rr.offset - rr.offset_lo], [rr.offset_hi - rr.offset]],
                 fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
    axb.barh(yy - hgt / 2 - 0.02, rr.vggt, height=hgt, color=ORANGE,
             label="VGGT reprojection" if yy == ys[0] else None)
    axb.errorbar(rr.vggt, yy - hgt / 2 - 0.02, xerr=[[rr.vggt - rr.vggt_lo], [rr.vggt_hi - rr.vggt]],
                 fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.5)
axb.axhline(4.5, color="#9A9A9A", lw=0.6, ls=(0, (3, 2)))
axb.set_yticks(ys)
axb.set_yticklabels(g.condition)
axb.axvline(0, color=INK, lw=0.6)
axb.axvline(100, color="#9A9A9A", lw=0.6, ls=(0, (1, 1.5)))
axb.set_xlabel("Change from the rigid twin\n(% of the change at 2$^\\circ$ smooth deformation)", linespacing=1.0)
axb.set_xlim(-8, 205)
axb.set_ylim(-0.6, n - 0.4)
axb.grid(axis="x", color="#E6E6E6", lw=0.5)
axb.set_axisbelow(True)
axb.text(203, 8.55, "geometry", ha="right", va="center", fontsize=7, color="#6B7380")
axb.text(203, 3.85, "appearance", ha="right", va="center", fontsize=7, color="#6B7380")
axb.legend(loc="lower right", bbox_to_anchor=(1.0, 0.985), ncol=2, frameon=False, handlelength=1.0,
           columnspacing=0.8, fontsize=7.2)
fig.text(0.525, 0.995, "(b) Response to known changes of the world", fontsize=8.5, fontweight="bold", va="top")

fig.savefig(os.path.join(OUT, "perturb_summary.pdf"), bbox_inches="tight", pad_inches=0.02)
print("ok")
