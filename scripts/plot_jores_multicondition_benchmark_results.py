#!/usr/bin/env python3
"""
Jores 2026 plant multi-condition MPRA — AlphaGenome encoder vs plantGREP.

Three-panel figure (a-c), each per condition, showing Pearson r:

  (a) Held-out TEST set       — AlphaGenome probing (stage 1) and fine-tuned (stage 2),
                                vs the plantGREP baseline.
  (b) Zero-shot PERTURBATION  — the design-validation library, split into TF *addition*
                                (TFBS insertion) and TF *ablation* (TFBS shuffling), for
                                fine-tuned AlphaGenome vs plantGREP.
  (c) Zero-shot DESIGN        — in-silico evolved sequences, fine-tuned AlphaGenome vs
                                plantGREP.

Conditions use the two-row species axis from jores26's plot_pearson_comparison.py: the
four tobacco conditions (light/dark/warm/cold), then the maize "dark" condition.

Values are read from results/jores_multicondition/reference/*.json (the metrics written
by alphagenome-ft-jores26's evaluate_jores*.py), so the figure tracks the committed
numbers. Style mirrors scripts/plot_plant_starrseq_benchmark_results.py (seaborn white,
grouped bars, black bar edges, rotated bold value labels, dashed y-grid, repo palette).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "results" / "jores_multicondition" / "reference"

# Display order + two-row species axis (maize condition shown as "Dark").
CONDITIONS = ["light", "dark", "warm", "cold", "maize"]
CONDITION_DISPLAY = ["Light", "Dark", "Warm", "Cold", "Dark"]
SPECIES_GROUPS = [(0, 3, "Tobacco"), (4, 4, "Maize")]

# Colour is by MODEL, consistent across panels and with the repo's other figures:
# AlphaGenome = blue/navy family, plantGREP = the grey-cream CNN-baseline colour. Within
# AlphaGenome, the paired conditions (probing/fine-tuned in (a), TF addition/ablation in
# (b)) are the dark/light shades of the navy.
AG_DARK, AG_LIGHT = "#394165", "#80A0C7"      # navy / steel blue — AlphaGenome
                                              # (Fine-tuned / Probing), as in the other figures
PG_FILL = "#D6D1C7"                           # grey-cream CNN baseline — the colour the other
                                              # repo figures give the Jores CNN
PG_TEXT = "#7E7869"                           # its dark companion: value labels + trajectory line
MEASURED_COLOR = "#B0413E"                    # brick red — distinct from AG navy / CNN cream

# Each bar: (label, colour, hatch, reference file, nested path above per_condition).
# In (b), addition and ablation share the model colour and differ only by hatch, and are
# kept adjacent within each model so they are easy to compare.
# (letter, title, n_samples, bars). Design (c) uses the by-objective eval: each
# condition scored only on the sequences evolved for it (on-target).
PANELS = [
    ("a", "Held-out test", 33300, [
        ("AG (Probing)",     AG_LIGHT, None, "test_probing.json",   ()),
        ("AG (Fine-tuned)",  AG_DARK,  None, "test_finetuned.json", ()),
        ("plantGREP",        PG_FILL,  None, "pgrep_test.json",      ()),
    ]),
    ("b", "Zero-shot perturbation", 22791, [
        ("AG · TF addition",        AG_DARK, None, "category_eval.json", ("perturbation", "insertion")),
        ("plantGREP · TF addition", PG_FILL, None, "pgrep_category.json", ("perturbation", "insertion")),
        ("AG · TF ablation",        AG_DARK, "///", "category_eval.json", ("perturbation", "shuffling")),
        ("plantGREP · TF ablation", PG_FILL, "///", "pgrep_category.json", ("perturbation", "shuffling")),
    ]),
    ("c", "Zero-shot design", 9665, [
        ("AG (Fine-tuned)", AG_DARK, None, "evolution_by_objective.json", ()),
        ("plantGREP",       PG_FILL, None, "pgrep_evolution_by_objective.json", ()),
    ]),
]

# Darker stand-in text for the lighter bars so value labels stay legible.
TEXT_COLORS = {AG_LIGHT: "#4A6A8C", PG_FILL: PG_TEXT}

# Font sizes, matched to the repo's other figures (seaborn font_scale 1.2, title ~13-14,
# axis labels 12, ticks/legend ~9-10, panel letters bold).
TITLE_FS, AXIS_FS, TICK_FS, LEGEND_FS, LETTER_FS, VALUE_FS = 13, 12, 11, 10, 15, 8


def setup_plot_style():
    sns.set(font_scale=1.2)
    sns.set_style("white")


def _per_condition(fname, nested):
    d = json.loads((REFERENCE_DIR / fname).read_text())
    for key in nested:
        d = d[key]
    per = d["per_condition"]
    return [per[c]["pearsonr"] for c in CONDITIONS]


def _label_bars(ax, bars, vals, color, fontsize=VALUE_FS):
    text_color = TEXT_COLORS.get(color, color)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.2f}", ha="center", va="bottom", fontsize=fontsize,
                fontweight="bold", color=text_color, rotation=90)


def _species_axis(ax):
    """Second label row below the condition ticks: a bracket line + species label."""
    trans = ax.get_xaxis_transform()
    fs = ax.get_xticklabels()[0].get_fontsize()
    line_y, text_y = -0.10, -0.155
    for start, end, label in SPECIES_GROUPS:
        x0, x1 = start - 0.47, end + 0.47
        ax.plot([x0, x1], [line_y, line_y], color="black", linewidth=1,
                transform=trans, clip_on=False)
        ax.text((x0 + x1) / 2, text_y, label, transform=trans, ha="center",
                va="top", fontsize=fs)


def _draw_bar_panel(ax, letter, title, n_samples, bars_spec):
    x = np.arange(len(CONDITIONS))
    n_bars = len(bars_spec)
    width = min(0.26, 0.72 / n_bars)
    for i, (label, color, hatch, fname, nested) in enumerate(bars_spec):
        vals = _per_condition(fname, nested)
        offset = (i - n_bars / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      alpha=0.9, edgecolor="black", linewidth=1, hatch=hatch)
        _label_bars(ax, bars, vals, color)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=LETTER_FS,
            fontweight="bold", va="bottom", ha="right")
    ax.text(0.98, 0.97, f"n = {n_samples:,}", transform=ax.transAxes, fontsize=TICK_FS,
            va="top", ha="right", color="#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_DISPLAY, fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylim([0.3, 1.0])
    ax.grid(axis="y", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)
    _species_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False,
              fontsize=LEGEND_FS, ncol=1 if n_bars <= 2 else 2, columnspacing=1.0,
              handlelength=1.3)


def _draw_trajectory(ax, rows, title, letter=None, show_xlabel=True, show_legend=True):
    """One measured evolved lineage from (c) across its mutation depths (0/6/12):
    measured warm activity vs each model's predicted warm activity. Raw values —
    AlphaGenome is trained on the measured log2-enrichment scale, so its line should
    track the measured one; plantGREP's own scale does not."""
    muts = [int(r["mutations"]) for r in rows]
    series = [
        ("Measured",    "warm_measured", MEASURED_COLOR, "o"),
        ("AlphaGenome", "warm_ag",       AG_DARK,        "s"),
        ("plantGREP",   "warm_pg",       PG_FILL,        "^"),
    ]
    for label, col, color, marker in series:
        # Black-edged markers (matching the black-edged bars) keep the light plantGREP
        # cream line legible against white.
        ax.plot(muts, [float(r[col]) for r in rows], "-", color=color, linewidth=2,
                marker=marker, markersize=7, markeredgecolor="black",
                markeredgewidth=0.8, label=label)

    ax.set_title(title, fontsize=TICK_FS)
    if letter:
        ax.text(-0.02, 1.10, letter, transform=ax.transAxes, fontsize=LETTER_FS,
                fontweight="bold", va="bottom", ha="right")
    if show_xlabel:
        ax.set_xlabel("Mutations from original", fontsize=AXIS_FS)
    ax.set_ylabel("Warm activity", fontsize=AXIS_FS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(muts)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(loc="upper left", frameon=False, fontsize=LEGEND_FS)


def plot_jores(figsize=(14, 11)):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.18)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[1, 0])
    for ax, panel in zip((ax_a, ax_b, ax_c), PANELS):     # a, b, c are the bar panels
        _draw_bar_panel(ax, *panel)
    ax_a.set_ylabel("Pearson's r", fontsize=AXIS_FS)
    ax_c.set_ylabel("Pearson's r", fontsize=AXIS_FS)

    # d: two stacked example lineages where plantGREP is very wrong, beside c.
    traj = list(csv.DictReader((REFERENCE_DIR / "evolution_trajectory_measured.csv").open()))
    lineages = []
    for r in traj:
        if not lineages or lineages[-1][0] != r["lineage"]:
            lineages.append((r["lineage"], []))
        lineages[-1][1].append(r)
    sub = gs[1, 1].subgridspec(2, 1, hspace=0.55)
    for i, (name, rows) in enumerate(lineages[:2]):
        ax = fig.add_subplot(sub[i])
        _draw_trajectory(ax, rows, title=name, letter="d" if i == 0 else None,
                         show_xlabel=(i == 1), show_legend=(i == 0))

    fig.subplots_adjust(bottom=0.10, top=0.95, left=0.07, right=0.97)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--output_dir", type=str,
                        default="results/jores_multicondition/plots")
    parser.add_argument("--output_name", type=str,
                        default="jores_multicondition_benchmark")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    setup_plot_style()
    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_jores()
    png = out_dir / f"{args.output_name}.png"
    fig.savefig(png, format="png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()
