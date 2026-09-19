# -*- coding: utf-8 -*-
"""逐类 AP 热力图 + 雷达图 + 混淆矩阵热力图。"""
import csv, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figure_data"
OUT = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figures_jstars"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]
ORDER = ["from_scratch", "pretrained_50ep", "freeze_10", "freeze_20", "recommended"]
LABEL = {"from_scratch": "From scratch", "pretrained_50ep": "Pretrained 50 ep", "freeze_10": "Freeze 10",
         "freeze_20": "Freeze 20", "recommended": "Recommended"}

rows = {}
for f in glob.glob(os.path.join(DATA, "perclass_*.csv")):
    name = os.path.basename(f)[len("perclass_"):-len(".csv")]
    if name not in ORDER: continue
    with open(f, encoding="utf-8-sig") as fh:
        r = list(csv.DictReader(fh))[0]
    rows[name] = [float(r[c]) for c in CLASSES]
models = [m for m in ORDER if m in rows]
if models:
    M = np.array([rows[m] for m in models])
    fig, ax = plt.subplots(figsize=(8.2, 2.6), dpi=300)
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.75, M.max()))
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([LABEL[m] for m in models], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "%.3f" % M[i, j], ha="center", va="center", fontsize=6.2,
                    color="white" if M[i, j] > 0.45 * M.max() else "#222222")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("AP@0.5", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("Per-class AP@0.5 across training strategies", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig10_perclass_heatmap.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig10_perclass_heatmap.pdf"), bbox_inches="tight"); plt.close(fig)
    print("saved fig10_perclass_heatmap")

    # radar
    ang = np.linspace(0, 2*np.pi, len(CLASSES), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(5.4, 5.0), dpi=300); ax = plt.subplot(111, polar=True)
    colors = {"from_scratch": "#C44E52", "pretrained_50ep": "#55A868", "freeze_10": "#DD8452",
              "freeze_20": "#8172B2", "recommended": "#4C72B0"}
    for m in models:
        v = rows[m] + rows[m][:1]
        ax.plot(ang, v, lw=1.5, label=LABEL[m], color=colors.get(m))
        ax.fill(ang, v, alpha=0.08, color=colors.get(m))
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(CLASSES, fontsize=7)
    ax.tick_params(axis="y", labelsize=7); ax.set_ylim(0, 0.8)
    ax.set_title("Per-class AP@0.5", fontsize=10, pad=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig11_perclass_radar.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig11_perclass_radar.pdf"), bbox_inches="tight"); plt.close(fig)
    print("saved fig11_perclass_radar")

cmf = os.path.join(DATA, "confusion_recommended.csv")
if os.path.exists(cmf):
    with open(cmf, encoding="utf-8-sig") as fh:
        rd = list(csv.reader(fh))
    names = rd[0][1:]
    A = np.array([[float(x) for x in row[1:]] for row in rd[1:]])
    An = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-9)
    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=300)
    im = ax.imshow(An, cmap="magma_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("True", fontsize=9)
    for i in range(An.shape[0]):
        for j in range(An.shape[1]):
            if An[i, j] > 0.005:
                ax.text(j, i, "%.2f" % An[i, j], ha="center", va="center", fontsize=5.6,
                        color="white" if An[i, j] > 0.55 else "#333333")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Normalised frequency", fontsize=8)
    ax.set_title("Confusion matrix, recommended configuration", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig12_confusion_matrix.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig12_confusion_matrix.pdf"), bbox_inches="tight"); plt.close(fig)
    print("saved fig12_confusion_matrix")
else:
    print("confusion csv not ready yet")
