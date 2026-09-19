# -*- coding: utf-8 -*-
"""JSTARS/IEEE 期刊图统一设计系统。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito 色盲友好调色板
C = {
    "blue":   "#0072B2",
    "sky":    "#56B4E9",
    "green":  "#009E73",
    "yellow": "#E69F00",
    "orange": "#D55E00",
    "purple": "#CC79A7",
    "grey":   "#7F7F7F",
    "dark":   "#2B2B2B",
    "light":  "#D9D9D9",
}
SEQ = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#7F7F7F"]

def apply():
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#4D4D4D",
        "axes.labelcolor": "#2B2B2B",
        "text.color": "#2B2B2B",
        "xtick.color": "#4D4D4D",
        "ytick.color": "#4D4D4D",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.color": "#B0B0B0",
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "errorbar.capsize": 2.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

def save(fig, outdir, name):
    import os
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    fig.savefig(os.path.join(outdir, name + ".png"))
    plt.close(fig)
    print("saved", name, flush=True)
