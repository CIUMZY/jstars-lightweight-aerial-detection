# -*- coding: utf-8 -*-
"""Fig10/11/12（热力图、雷达、混淆矩阵）——稳健版。"""
import csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_jstars import apply, save, C, SEQ
import matplotlib.pyplot as plt
apply()
OUT = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figures_jstars"
FDATA = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figure_data"
CLASSES = ["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]
ORDER = ["from_scratch","pretrained_50ep","freeze_10","freeze_20","recommended"]
LAB = {"from_scratch":"From scratch","pretrained_50ep":"Pretrained 50 ep","freeze_10":"Freeze 10",
       "freeze_20":"Freeze 20","recommended":"Recommended"}
rows2 = {}
for name in ORDER:
    f = os.path.join(FDATA, "perclass_%s.csv" % name)
    if not os.path.exists(f): continue
    with open(f, encoding="utf-8-sig") as fh:
        rd = list(csv.DictReader(fh))
    if rd:
        rows2[name] = [float(rd[0][c]) for c in CLASSES]
models2 = [m for m in ORDER if m in rows2]
print("loaded:", models2, flush=True)

M = np.array([rows2[m] for m in models2])
fig, ax = plt.subplots(figsize=(7.0, 2.3))
im = ax.imshow(M, aspect="auto", cmap="cividis", vmin=0, vmax=0.72)
ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=32, ha="right", fontsize=7)
ax.set_yticks(range(len(models2))); ax.set_yticklabels([LAB[m] for m in models2], fontsize=7.5)
ax.set_xticks(np.arange(-0.5, len(CLASSES), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(models2), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8); ax.tick_params(which="minor", length=0)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=6, color="white" if M[i, j] < 0.38 else "#1A1A1A")
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.012); cb.set_label("AP@0.5", fontsize=7.5); cb.ax.tick_params(labelsize=7)
ax.set_title("Per-class accuracy across training strategies", fontsize=8.5, pad=6)
save(fig, OUT, "fig10_perclass_heatmap")

ang = np.linspace(0, 2*np.pi, len(CLASSES), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(4.2, 3.9)); ax = plt.subplot(111, polar=True)
for i, m in enumerate(models2):
    v = rows2[m] + rows2[m][:1]
    ax.plot(ang, v, lw=1.3, label=LAB[m], color=SEQ[i], marker="o", markersize=2.2, markerfacecolor="white", markeredgewidth=0.6)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(CLASSES, fontsize=6.5)
ax.tick_params(axis="y", labelsize=6.5); ax.set_ylim(0, 0.75); ax.set_rlabel_position(150)
ax.grid(alpha=0.3, lw=0.5); ax.spines["polar"].set_color("#CCCCCC"); ax.spines["polar"].set_linewidth(0.7)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=6.5)
save(fig, OUT, "fig11_perclass_radar")

cmf = os.path.join(FDATA, "confusion_recommended.csv")
with open(cmf, encoding="utf-8-sig") as fh:
    rd = list(csv.reader(fh))
names = rd[0][1:]; A = np.array([[float(x) for x in row[1:]] for row in rd[1:]])
An = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-9)
fig, ax = plt.subplots(figsize=(4.6, 4.0))
im = ax.imshow(An, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=6.5)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=6.5)
ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.7); ax.tick_params(which="minor", length=0)
for i in range(An.shape[0]):
    for j in range(An.shape[1]):
        if An[i, j] >= 0.01:
            ax.text(j, i, "%.2f" % An[i, j], ha="center", va="center", fontsize=5.4, color="white" if An[i, j] > 0.5 else "#333333")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Row-normalised frequency", fontsize=7.5); cb.ax.tick_params(labelsize=7)
ax.set_title("Confusion matrix of the recommended configuration", fontsize=8.5, pad=6)
save(fig, OUT, "fig12_confusion_matrix")
print("DONE 10/11/12")
