# -*- coding: utf-8 -*-
"""Fig. 3：预训练增益 vs 从头训练精度（5 个操作点，跨 3 个数据集）。"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pts = [
    ("NEU-DET (1458, easy)",  0.6942, 0.0484, "#4C72B0", "o"),
    ("AI-TOD (3235)",         0.1994, 0.0892, "#8172B2", "P"),
    ("AI-TOD (1617)",         0.1423, 0.1002, "#DD8452", "s"),
    ("VisDrone (1617)",       0.1011, 0.1095, "#55A868", "^"),
    ("AI-TOD (647)",          0.0347, 0.1401, "#C44E52", "D"),
]
out = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figures_jstars"
x = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts])
r = np.corrcoef(x, y)[0, 1]
slope, intercept = np.polyfit(x, y, 1)
xs = np.linspace(x.min()*0.75, x.max()*1.06, 60)

fig, ax = plt.subplots(figsize=(5.6, 3.7), dpi=300)
ax.plot(xs, slope*xs + intercept, "--", color="grey", lw=1.2, zorder=1, label="linear fit (r = %.3f)" % r)
for name, xv, yv, c, m in pts:
    ax.scatter(xv, yv, s=78, color=c, marker=m, zorder=3, edgecolor="white", linewidth=0.7, label=name)
    ax.annotate(name, (xv, yv), textcoords="offset points", xytext=(7, 4), fontsize=7, color=c)
ax.set_xlabel("mAP@0.5 trained from scratch", fontsize=9.5)
ax.set_ylabel("Pretraining gain (mAP@0.5)", fontsize=9.5)
ax.set_title("Pretraining pays off most where learning from scratch is hardest", fontsize=10)
ax.grid(alpha=0.25, lw=0.5)
ax.tick_params(labelsize=8.5)
ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(out, "fig3_pretrain_gain_vs_scratch.png"), bbox_inches="tight")
fig.savefig(os.path.join(out, "fig3_pretrain_gain_vs_scratch.pdf"), bbox_inches="tight")
print("r = %.4f, n = %d, slope = %.4f" % (r, len(pts), slope))
