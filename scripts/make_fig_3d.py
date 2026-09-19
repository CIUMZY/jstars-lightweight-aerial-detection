# -*- coding: utf-8 -*-
"""3D 柱状图：AI-TOD 三档标注预算 x 两种训练条件 -> mAP@0.5。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figures_jstars"
os.makedirs(OUT, exist_ok=True)

budgets = ["647", "1617", "3235"]
scratch = [0.0347, 0.1423, 0.1994]
pretrain = [0.1748, 0.2424, 0.2886]

fig = plt.figure(figsize=(6.2, 4.4), dpi=300)
ax = fig.add_subplot(111, projection="3d")
xpos = np.arange(len(budgets))
ypos = np.array([0, 1])
dx = dy = 0.5

for i, b in enumerate(xpos):
    ax.bar3d(b, 0, 0, 0.5, 0.5, scratch[i], color="#C44E52", alpha=0.95, edgecolor="white", linewidth=0.4)
    ax.bar3d(b, 1.0, 0, 0.5, 0.5, pretrain[i], color="#4C72B0", alpha=0.95, edgecolor="white", linewidth=0.4)
    ax.text(b + 0.25, 0.15, scratch[i] + 0.01, "%.3f" % scratch[i], ha="center", fontsize=7)
    ax.text(b + 0.25, 1.15, pretrain[i] + 0.01, "%.3f" % pretrain[i], ha="center", fontsize=7)
    ax.text(b + 0.25, 0.5, -0.045, "+%.3f" % (pretrain[i] - scratch[i]), ha="center", fontsize=7.5, color="#333333")

ax.set_xticks(xpos + 0.25); ax.set_xticklabels(budgets, fontsize=8)
ax.set_yticks([0.25, 1.25]); ax.set_yticklabels(["from scratch", "pretrained"], fontsize=8)
ax.set_xlabel("AI-TOD label budget (images)", fontsize=8.5, labelpad=6)
ax.set_ylabel("Training condition", fontsize=8.5, labelpad=6)
ax.set_zlabel("mAP@0.5", fontsize=8.5, labelpad=4)
ax.set_zlim(0, 0.36)
ax.view_init(elev=22, azim=-58)
ax.set_title("Pretraining gain shrinks as labels grow (AI-TOD)", fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#C44E52", label="from scratch"), Patch(facecolor="#4C72B0", label="pretrained")],
          fontsize=7.5, frameon=False, loc="upper left")
fig.subplots_adjust(left=0.01, right=0.92, top=0.94, bottom=0.06)
fig.savefig(os.path.join(OUT, "fig9_aitod_3d_bars.png"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig9_aitod_3d_bars.pdf"), bbox_inches="tight")
print("saved fig9_aitod_3d_bars")
