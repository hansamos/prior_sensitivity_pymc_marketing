"""Figures from the analysis CSVs (run `analyze.py` first).

    python -I evidence/common/make_figures.py

error_vs_khat.png            worst-quantity error of the reweighted posterior against Pareto k-hat, one dot per scenario
reweighted_vs_do_nothing.png median error per k-hat bin on the quantities the change moves: unweighted reference vs reweighted
error_vs_ess.png             the same error against the ESS fraction, scenarios with k-hat < 0.7

Errors are in refit posterior SDs; figures 1 and 3 show the worst of the nine posterior quantities of a scenario. Refits that did not
converge are left out (listed in results/SUMMARY.md). Colours: one accent (blue) on neutral grays.
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import summarize  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS, OUT = HERE.parent / "results", HERE.parent / "figures"
OUT.mkdir(exist_ok=True)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WASH = "#e1e0d9", "#c3c2b7", "#f1f0ec"
BLUE = "#2a78d6"
NOISE = 0.009  # median error between two refits of one scenario (results/SUMMARY.md)
TITLES = {"priors": "Prior changes (observational model)", "likelihood": "Likelihood and combined changes (lift-calibrated model)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": AXIS})


def style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(axis=grid, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def header(fig, title, subtitle):
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=INK)
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=9.5, color=INK2)


def dots(ax, x, y, **kw):
    ax.scatter(x, y, s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3, **kw)  # 2px surface ring


def fig_error_vs_khat(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    xmax = 3.0
    for ax, (t, s) in zip(axes, data.items()):
        style(ax)
        ax.axvspan(0.5, 0.7, color=WASH, zorder=0)
        for k in (0.5, 0.7):
            ax.axvline(k, color=AXIS, linewidth=1, zorder=1)
        ax.axhline(NOISE, color=MUTED, linewidth=1, zorder=1)
        inside = s.pareto_k <= xmax
        dots(ax, s.pareto_k[inside], s.err_max[inside])
        if (~inside).any():  # beyond the axis: drawn on the edge as open triangles
            ax.scatter(np.full((~inside).sum(), xmax), s.err_max[~inside], s=46, marker=">", facecolor=SURFACE,
                       edgecolor=BLUE, linewidth=1.5, zorder=3, clip_on=False)
        ax.set_yscale("log")
        ax.set_xlim(-1.0, xmax)
        ax.set_ylim(0.004, 12)
        ax.set_xlabel("Pareto k̂ of the reweighting")
        ax.set_title(f"{TITLES[t]},  n = {len(s)}", loc="left", fontsize=10, color=INK, pad=12)
        ax.text(0.6, 8, "check", ha="center", va="top", fontsize=8.5, color=INK2)
        ax.text(xmax - 0.05, NOISE * 1.12, "refit noise", ha="right", va="bottom", fontsize=8.5, color=INK2)
    axes[0].set_ylabel("error vs. full refit (refit posterior SDs, worst quantity)")
    axes[0].set_yticks([0.1, 1, 10])
    axes[0].set_yticklabels(["0.1", "1", "10"])
    header(fig, "The reweighted posterior is close to the refit while k̂ < 0.5 and drifts away beyond 0.7",
           "One dot per scenario. Open triangles: k̂ above 3, drawn on the edge. Grey band: k̂ 0.5 to 0.7.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "error_vs_khat.png", dpi=200)
    plt.close(fig)


def fig_vs_do_nothing(qdata):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, (t, g) in zip(axes, qdata.items()):
        style(ax, grid="x")
        ys = np.arange(len(g))[::-1]
        for y, (_, r) in zip(ys, g.iterrows()):
            ax.plot([r.err_do_nothing, r.err_reweighted], [y, y], color=AXIS, linewidth=2, zorder=1, solid_capstyle="round")
            ax.scatter([r.err_do_nothing], [y], s=56, color=MUTED, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.scatter([r.err_reweighted], [y], s=56, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=4)
            left = r.err_reweighted < r.err_do_nothing
            label = f"{r.err_reweighted:.2g}" if r.err_reweighted < 1 else f"{r.err_reweighted:.1f}"
            ax.text(r.err_reweighted * (0.8 if left else 1.25), y, label,
                    ha="right" if left else "left", va="center", fontsize=9, color=INK)
        ax.axvline(NOISE, color=MUTED, linewidth=1, zorder=0)
        ax.text(NOISE * 1.1, ys.max() + 0.62, "refit noise", fontsize=8.5, color=INK2, va="bottom")
        ax.set_yticks(ys)
        ax.set_yticklabels([f"k̂ {lab}   (n={int(n)})" for lab, n in zip(g.k_bin, g.n)])
        ax.set_xscale("log")
        ax.set_xlim(0.001, 12)
        ax.set_xticks([0.1, 1, 10])
        ax.set_xticklabels(["0.1", "1", "10"])
        ax.set_ylim(-0.6, ys.max() + 0.95)
        ax.set_xlabel("error vs. full refit (refit posterior SDs, median)")
        ax.set_title(TITLES[t], loc="left", fontsize=10, color=INK, pad=10)
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=MUTED, markeredgecolor=SURFACE, markersize=8,
                      label="unweighted reference posterior (do nothing)"),
               Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor=SURFACE, markersize=8,
                      label="reweighted reference posterior")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.005, 0.885), frameon=False, ncol=2, fontsize=9,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.5)
    header(fig, "Reweighting moves the posterior toward the refit; at k̂ ≥ 1 it stops helping likelihood changes",
           f"Posterior quantities the change moves by more than {summarize.MOVED} refit SD; n = number of such quantities. "
           "Rows with n < 10 are left out.")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(OUT / "reweighted_vs_do_nothing.png", dpi=200)
    plt.close(fig)


def fig_error_vs_ess(data):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (t, s) in zip(axes, data.items()):
        style(ax)
        s = s[s.pareto_k < 0.7]
        ax.axhline(NOISE, color=MUTED, linewidth=1, zorder=1)
        dots(ax, s.ess_fraction, s.err_max)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.01, 1.2)
        ax.set_ylim(0.004, 2)
        ax.set_xticks([0.01, 0.1, 1])
        ax.set_xticklabels(["1%", "10%", "100%"])
        ax.set_xlabel("effective sample size of the weights, share of the draws")
        ax.set_title(f"{TITLES[t]},  n = {len(s)}", loc="left", fontsize=10, color=INK, pad=10)
        ax.text(0.0105, NOISE * 1.12, "refit noise", fontsize=8.5, color=INK2, va="bottom")
    axes[0].set_ylabel("error vs. full refit (refit posterior SDs, worst quantity)")
    axes[0].set_yticks([0.1, 1])
    axes[0].set_yticklabels(["0.1", "1"])
    header(fig, "Among scenarios with k̂ < 0.7, a low ESS still goes with a larger error",
           "One dot per scenario with k̂ below 0.7.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "error_vs_ess.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    data, qdata = {}, {}
    for t in ("priors", "likelihood"):
        s = summarize.load(RESULTS, t)
        data[t] = s[s.refit_ok]
        tq = summarize.by_bin_quantities(summarize.load_quantities(RESULTS, t, s))
        qdata[t] = tq[(tq.quantities == "moved") & (tq.n >= 10)]
    fig_error_vs_khat(data)
    fig_vs_do_nothing(qdata)
    fig_error_vs_ess(data)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
