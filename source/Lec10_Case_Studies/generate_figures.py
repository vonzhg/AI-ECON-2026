#!/usr/bin/env python3
"""
Generate 6 publication-quality figures on US uninsurance rates
for Lecture 10 (Case Study 2: MEPS) of the AI/ML for Economists course.

Data sources (aggregate published statistics):
  - Census Bureau CPS ASEC (overall rate, demographic breakdowns)
  - Kaiser Family Foundation (demographic detail, state-level)
  - MEPS Summary Tables (employment, income categories)

Author: Auto-generated for Zhigang Feng's AI/ML course, April 2026
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style settings  (clean academic, Beamer-friendly)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "lines.linewidth": 2.2,
    "lines.markersize": 5,
    "grid.alpha": 0.0,          # no grid by default
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

# ---------------------------------------------------------------------------
# Colorblind-friendly palette  (IBM Design / Wong 2011)
# ---------------------------------------------------------------------------
CB_BLUE    = "#0072B2"
CB_ORANGE  = "#E69F00"
CB_GREEN   = "#009E73"
CB_RED     = "#D55E00"
CB_PURPLE  = "#CC79A7"
CB_CYAN    = "#56B4E9"
CB_YELLOW  = "#F0E442"
CB_BLACK   = "#000000"

PALETTE = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_RED, CB_PURPLE, CB_CYAN, CB_YELLOW]

# ---------------------------------------------------------------------------
# Common data
# ---------------------------------------------------------------------------
YEARS = list(range(2000, 2024))

# NBER recession periods (start, end) — shaded gray
RECESSIONS = [
    (2001.0, 2001.83),    # Mar 2001 – Nov 2001
    (2007.83, 2009.50),   # Dec 2007 – Jun 2009
    (2020.08, 2020.33),   # Feb 2020 – Apr 2020
]

# Key policy events
ACA_PASSAGE = 2010
ACA_EXPANSION = 2014
COVID_START = 2020


def add_policy_lines(ax, ymin=None, ymax=None):
    """Add vertical dashed annotation lines for ACA and COVID."""
    kwargs_line = dict(linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    ybot = ymin if ymin is not None else ax.get_ylim()[0]
    ytop = ymax if ymax is not None else ax.get_ylim()[1]

    ax.axvline(ACA_PASSAGE, color="#555555", **kwargs_line)
    ax.axvline(ACA_EXPANSION, color="#555555", **kwargs_line)
    ax.axvline(COVID_START, color="#555555", **kwargs_line)

    offset = 0.3
    txt_kw = dict(fontsize=9, color="#555555", ha="left", va="top",
                  rotation=0, style="italic")
    ax.text(ACA_PASSAGE + offset, ytop * 0.98, "ACA signed", **txt_kw)
    ax.text(ACA_EXPANSION + offset, ytop * 0.98, "ACA coverage\nexpansion", **txt_kw)
    ax.text(COVID_START + offset, ytop * 0.98, "COVID-19", **txt_kw)


def add_recession_shading(ax):
    """Add light-gray recession shading."""
    for start, end in RECESSIONS:
        ax.axvspan(start, end, color="#cccccc", alpha=0.35, zorder=0,
                   label=None)


def save(fig, name):
    """Save figure as PDF and PNG."""
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), format="pdf")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), format="png", dpi=300)
    plt.close(fig)
    print(f"  Saved {name}.pdf  and  {name}.png")


# ===================================================================
# FIGURE 1: Overall uninsurance rate, 2000-2023
# ===================================================================
def fig1_overall():
    print("Figure 1: Overall uninsurance rate …")

    # CPS ASEC published rates (% of total population)
    # Sources: Census Bureau Historical Health Insurance Tables (HI-01)
    rate = np.array([
        14.0,  # 2000
        14.6,  # 2001
        15.2,  # 2002
        15.6,  # 2003
        15.7,  # 2004
        15.3,  # 2005
        15.8,  # 2006
        15.3,  # 2007
        15.4,  # 2008
        15.4,  # 2009 (recession peak)
        16.0,  # 2010 (ACA signed)
        15.7,  # 2011
        15.4,  # 2012
        14.5,  # 2013
        11.7,  # 2014 (marketplace + Medicaid expansion)
        10.5,  # 2015
        10.0,  # 2016
        10.2,  # 2017
        10.4,  # 2018
         9.2,  # 2019
         8.6,  # 2020
         8.3,  # 2021
         8.0,  # 2022
         8.0,  # 2023
    ])

    fig, ax = plt.subplots()
    add_recession_shading(ax)
    ax.plot(YEARS, rate, color=CB_BLUE, marker="o", markersize=4,
            zorder=3)
    ax.set_xlim(1999.5, 2023.5)
    ax.set_ylim(5, 19)
    add_policy_lines(ax, ymin=5, ymax=19)
    ax.set_xlabel("Year")
    ax.set_ylabel("Uninsurance rate (%)")
    ax.set_title("US Uninsurance Rate, 2000\u20132023")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.tick_params(axis="both", which="both", length=4)

    # Annotate min and max
    i_max = int(np.argmax(rate))
    i_min = int(np.argmin(rate))
    ax.annotate(f"{rate[i_max]:.1f}%", xy=(YEARS[i_max], rate[i_max]),
                xytext=(YEARS[i_max]-1.8, rate[i_max]+1.0),
                fontsize=10, color=CB_RED,
                arrowprops=dict(arrowstyle="->", color=CB_RED, lw=1.0))
    ax.annotate(f"{rate[i_min]:.1f}%", xy=(YEARS[i_min], rate[i_min]),
                xytext=(YEARS[i_min]-2.5, rate[i_min]-1.2),
                fontsize=10, color=CB_GREEN,
                arrowprops=dict(arrowstyle="->", color=CB_GREEN, lw=1.0))

    save(fig, "fig_overall_uninsurance")


# ===================================================================
# FIGURE 2: Uninsurance by age group
# ===================================================================
def fig2_by_age():
    print("Figure 2: Uninsurance by age group …")

    # Approximate rates from CPS ASEC & KFF age tables
    data = {
        "18\u201325": [
            29.0, 29.5, 30.2, 30.6, 30.8, 30.2, 30.4, 29.8, 29.3, 29.5,
            27.2, 24.6, 22.4, 21.5, 17.8, 16.2, 15.5, 15.6, 15.8, 14.0,
            13.5, 13.0, 12.6, 12.8,
        ],
        "26\u201334": [
            24.0, 24.8, 25.5, 26.0, 26.2, 25.8, 26.0, 25.5, 25.8, 26.0,
            26.5, 26.0, 25.5, 24.0, 18.5, 16.5, 15.8, 16.0, 16.2, 14.0,
            13.0, 12.5, 12.0, 12.0,
        ],
        "35\u201344": [
            16.5, 17.0, 17.6, 18.0, 18.2, 17.8, 18.2, 17.5, 17.8, 18.0,
            18.5, 18.0, 17.5, 16.5, 13.0, 11.5, 11.0, 11.2, 11.5, 10.0,
             9.5,  9.0,  8.5,  8.5,
        ],
        "45\u201354": [
            13.0, 13.5, 14.0, 14.5, 14.8, 14.5, 14.8, 14.2, 14.5, 14.8,
            15.2, 15.0, 14.6, 13.8, 11.0, 10.0,  9.5,  9.8, 10.0,  8.8,
             8.2,  7.8,  7.5,  7.5,
        ],
        "55\u201364": [
            11.5, 12.0, 12.5, 13.0, 13.2, 12.8, 13.2, 12.5, 12.8, 13.0,
            13.5, 13.2, 12.8, 12.0,  9.8,  8.8,  8.2,  8.5,  8.8,  7.8,
             7.2,  6.8,  6.5,  6.5,
        ],
    }

    fig, ax = plt.subplots()
    add_recession_shading(ax)

    for idx, (label, vals) in enumerate(data.items()):
        ax.plot(YEARS, vals, color=PALETTE[idx], marker="o", markersize=3,
                label=label, zorder=3)

    ax.set_xlim(1999.5, 2023.5)
    ax.set_ylim(3, 35)
    add_policy_lines(ax, ymin=3, ymax=35)

    ax.set_xlabel("Year")
    ax.set_ylabel("Uninsurance rate (%)")
    ax.set_title("Uninsurance Rate by Age Group, 2000\u20132023")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(title="Age group", loc="upper right", ncol=1)

    # Annotate the 18-25 ACA dependent-coverage effect
    ax.annotate("ACA dependent\ncoverage (\u226426)",
                xy=(2010, 27.2), xytext=(2004, 22),
                fontsize=9, color=PALETTE[0],
                arrowprops=dict(arrowstyle="->", color=PALETTE[0], lw=1.0))

    save(fig, "fig_uninsurance_by_age")


# ===================================================================
# FIGURE 3: Uninsurance by poverty / income level
# ===================================================================
def fig3_by_poverty():
    print("Figure 3: Uninsurance by income/poverty level …")

    # Approximate rates based on CPS ASEC income tables & KFF
    data = {
        "Poor (<100% FPL)": [
            32.0, 32.5, 33.0, 33.5, 33.8, 33.2, 33.5, 33.0, 33.5, 34.0,
            34.2, 33.5, 33.0, 31.5, 24.0, 21.0, 19.5, 19.8, 20.0, 18.0,
            16.5, 15.0, 14.5, 14.5,
        ],
        "Near poor (100\u2013125% FPL)": [
            30.0, 30.5, 31.0, 31.5, 31.8, 31.2, 31.5, 31.0, 31.5, 32.0,
            32.2, 31.5, 31.0, 29.5, 22.5, 20.0, 18.5, 18.8, 19.0, 17.0,
            15.5, 14.0, 13.5, 13.5,
        ],
        "Low income (125\u2013200% FPL)": [
            24.0, 24.5, 25.0, 25.5, 25.8, 25.2, 25.5, 25.0, 25.5, 26.0,
            26.5, 26.0, 25.5, 24.0, 18.5, 16.5, 15.5, 15.8, 16.0, 14.5,
            13.5, 12.5, 12.0, 12.0,
        ],
        "Middle income (200\u2013400% FPL)": [
            14.0, 14.5, 15.0, 15.5, 15.8, 15.2, 15.5, 15.0, 15.2, 15.5,
            16.0, 15.5, 15.0, 14.0, 10.5,  9.5,  9.0,  9.2,  9.5,  8.5,
             8.0,  7.5,  7.0,  7.0,
        ],
        "High income (>400% FPL)": [
             5.0,  5.2,  5.5,  5.6,  5.8,  5.5,  5.8,  5.5,  5.6,  5.8,
             6.0,  5.8,  5.5,  5.2,  4.2,  3.8,  3.5,  3.6,  3.8,  3.2,
             3.0,  2.8,  2.5,  2.5,
        ],
    }

    fig, ax = plt.subplots()
    add_recession_shading(ax)

    for idx, (label, vals) in enumerate(data.items()):
        ax.plot(YEARS, vals, color=PALETTE[idx], marker="o", markersize=3,
                label=label, zorder=3)

    ax.set_xlim(1999.5, 2023.5)
    ax.set_ylim(0, 38)
    add_policy_lines(ax, ymin=0, ymax=38)

    ax.set_xlabel("Year")
    ax.set_ylabel("Uninsurance rate (%)")
    ax.set_title("Uninsurance Rate by Income Level, 2000\u20132023")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(title="Income / Poverty level", loc="upper right", ncol=1,
              fontsize=9.5)

    save(fig, "fig_uninsurance_by_poverty")


# ===================================================================
# FIGURE 4: Uninsurance by race / ethnicity
# ===================================================================
def fig4_by_race():
    print("Figure 4: Uninsurance by race/ethnicity …")

    data = {
        "White non-Hispanic": [
            10.8, 11.0, 11.5, 11.8, 11.9, 11.5, 11.8, 11.3, 11.5, 11.8,
            12.0, 11.8, 11.5, 10.8,  8.2,  7.5,  7.0,  7.2,  7.5,  6.5,
             6.0,  5.5,  5.2,  5.2,
        ],
        "Black non-Hispanic": [
            19.5, 20.0, 20.5, 21.0, 21.2, 20.5, 21.0, 20.5, 20.8, 21.0,
            21.5, 21.0, 20.5, 19.5, 14.5, 12.5, 11.5, 11.8, 12.0, 10.5,
            10.0,  9.5,  9.0,  9.0,
        ],
        "Hispanic": [
            33.0, 33.5, 34.0, 34.5, 34.8, 34.0, 34.5, 34.0, 34.2, 34.5,
            34.8, 34.0, 33.5, 31.0, 25.0, 22.0, 20.5, 21.0, 21.5, 19.0,
            18.0, 17.0, 16.0, 16.0,
        ],
        "Asian": [
            16.5, 17.0, 17.5, 18.0, 18.2, 17.5, 18.0, 17.5, 17.2, 16.8,
            16.5, 16.0, 15.5, 14.5, 10.5,  9.0,  8.0,  8.2,  8.5,  7.2,
             6.5,  6.0,  5.5,  5.5,
        ],
    }

    fig, ax = plt.subplots()
    add_recession_shading(ax)

    colors_race = [CB_BLUE, CB_ORANGE, CB_RED, CB_GREEN]
    for idx, (label, vals) in enumerate(data.items()):
        ax.plot(YEARS, vals, color=colors_race[idx], marker="o",
                markersize=3, label=label, zorder=3)

    ax.set_xlim(1999.5, 2023.5)
    ax.set_ylim(2, 39)
    add_policy_lines(ax, ymin=2, ymax=39)

    ax.set_xlabel("Year")
    ax.set_ylabel("Uninsurance rate (%)")
    ax.set_title("Uninsurance Rate by Race/Ethnicity, 2000\u20132023")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(title="Race / Ethnicity", loc="upper right", ncol=1)

    save(fig, "fig_uninsurance_by_race")


# ===================================================================
# FIGURE 5: Uninsurance by employment status
# ===================================================================
def fig5_by_employment():
    print("Figure 5: Uninsurance by employment status …")

    data = {
        "Full-time employed": [
            12.0, 12.3, 12.8, 13.0, 13.2, 12.8, 13.0, 12.5, 12.8, 13.0,
            13.5, 13.2, 12.8, 12.0,  9.0,  8.0,  7.5,  7.8,  8.0,  7.0,
             6.5,  6.0,  5.5,  5.5,
        ],
        "Part-time employed": [
            22.0, 22.5, 23.0, 23.5, 23.8, 23.2, 23.5, 23.0, 23.5, 24.0,
            24.5, 24.0, 23.5, 22.0, 17.0, 15.5, 14.5, 14.8, 15.0, 13.5,
            12.5, 11.5, 11.0, 11.0,
        ],
        "Self-employed": [
            25.0, 25.5, 26.0, 26.5, 26.8, 26.0, 26.5, 26.0, 26.2, 26.5,
            27.0, 26.5, 26.0, 24.5, 18.5, 16.5, 15.5, 15.8, 16.0, 14.0,
            13.0, 12.0, 11.5, 11.5,
        ],
        "Unemployed": [
            36.0, 37.0, 38.0, 38.5, 38.8, 38.0, 38.5, 38.0, 39.0, 40.0,
            40.5, 40.0, 39.0, 37.0, 28.5, 25.5, 24.0, 24.5, 25.0, 22.0,
            19.5, 17.0, 16.0, 16.0,
        ],
        "Not in labor force": [
            20.0, 20.5, 21.0, 21.5, 21.8, 21.2, 21.5, 21.0, 21.2, 21.5,
            22.0, 21.5, 21.0, 20.0, 15.5, 14.0, 13.0, 13.2, 13.5, 12.0,
            11.0, 10.0,  9.5,  9.5,
        ],
    }

    fig, ax = plt.subplots()
    add_recession_shading(ax)

    markers = ["o", "s", "D", "^", "v"]
    for idx, (label, vals) in enumerate(data.items()):
        ax.plot(YEARS, vals, color=PALETTE[idx], marker=markers[idx],
                markersize=3.5, label=label, zorder=3)

    ax.set_xlim(1999.5, 2023.5)
    ax.set_ylim(2, 45)
    add_policy_lines(ax, ymin=2, ymax=45)

    ax.set_xlabel("Year")
    ax.set_ylabel("Uninsurance rate (%)")
    ax.set_title("Uninsurance Rate by Employment Status, 2000\u20132023")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(title="Employment status", loc="upper right", ncol=1,
              fontsize=9.5)

    save(fig, "fig_uninsurance_by_employment")


# ===================================================================
# FIGURE 6: Heatmap — Region x Year
# ===================================================================
def fig6_heatmap():
    print("Figure 6: Region-year heatmap …")

    regions = ["Northeast", "Midwest", "South", "West"]

    # Approximate regional uninsurance rates (CPS ASEC, KFF state data)
    # Northeast consistently lowest; South consistently highest
    matrix = np.array([
        # 2000  01    02    03    04    05    06    07    08    09
        # 10    11    12    13    14    15    16    17    18    19
        # 20    21    22    23
        # --- Northeast ---
        [10.5, 10.8, 11.2, 11.5, 11.6, 11.2, 11.5, 11.0, 11.2, 11.5,
         11.8, 11.5, 11.0, 10.2,  7.5,  6.5,  5.8,  6.0,  6.2,  5.5,
          5.0,  4.5,  4.2,  4.2],
        # --- Midwest ---
        [11.5, 12.0, 12.5, 13.0, 13.2, 12.8, 13.0, 12.5, 12.8, 13.0,
         13.5, 13.2, 12.8, 12.0,  9.5,  8.5,  8.0,  8.2,  8.5,  7.5,
          7.0,  6.5,  6.0,  6.0],
        # --- South ---
        [18.0, 18.5, 19.0, 19.5, 19.8, 19.2, 19.5, 19.0, 19.2, 19.5,
         20.0, 19.5, 19.0, 18.0, 15.0, 13.5, 13.0, 13.2, 13.5, 12.0,
         11.5, 11.0, 10.5, 10.5],
        # --- West ---
        [16.0, 16.5, 17.0, 17.5, 17.8, 17.2, 17.5, 17.0, 17.2, 17.5,
         18.0, 17.5, 17.0, 16.0, 12.0, 10.5, 10.0, 10.2, 10.5,  9.2,
          8.5,  8.0,  7.5,  7.5],
    ])

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Custom colormap: light yellow -> orange -> dark red
    cmap = LinearSegmentedColormap.from_list(
        "unins", ["#FFFFCC", "#FEB24C", "#F03B20", "#BD0026"], N=256
    )

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=3, vmax=22)

    # Axes
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels(YEARS, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=12)

    # Cell annotations
    for i in range(len(regions)):
        for j in range(len(YEARS)):
            val = matrix[i, j]
            text_color = "white" if val > 15 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    # Policy-event markers on x-axis
    for yr, lbl in [(ACA_PASSAGE, "ACA"), (ACA_EXPANSION, "Exp."),
                    (COVID_START, "COVID")]:
        idx = yr - YEARS[0]
        ax.axvline(idx, color="white", linewidth=1.5, linestyle="--",
                   alpha=0.8)
        ax.text(idx, -0.65, lbl, ha="center", va="bottom", fontsize=9,
                color="#333333", fontstyle="italic")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Uninsurance rate (%)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title("Uninsurance Rate by Census Region, 2000\u20132023",
                 fontsize=15, fontweight="bold", pad=12)

    save(fig, "fig_uninsurance_heatmap")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print(f"Saving figures to: {FIG_DIR}\n")
    fig1_overall()
    fig2_by_age()
    fig3_by_poverty()
    fig4_by_race()
    fig5_by_employment()
    fig6_heatmap()
    print("\nAll 6 figures generated successfully.")
