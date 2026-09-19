# -*- coding: utf-8 -*-
"""生成 JSTARS 扩展版其余主图（数据来自已完成的实测结果）。"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figures_jstars"
ROOT = r"D:\Research\03_Codex\projects\yolo-paper"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 300})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)

# ---------- Fig 1: protocol ----------
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.axis("off")
boxes = [
    (0.02, 0.55, 0.20, 0.33, "Datasets\nVisDrone | AI-TOD | NEU-DET"),
    (0.27, 0.55, 0.20, 0.33, "Label budgets\n647 / 1617 / 3235 imgs\n(+ 10-100% VisDrone)"),
    (0.52, 0.55, 0.20, 0.33, "7 training choices x seeds\npretrain, mixup, CutMix,\nKD, freeze, epochs, TTA"),
    (0.77, 0.55, 0.21, 0.33, "Official-val evaluation\npaired t-tests, effect sizes"),
    (0.27, 0.10, 0.20, 0.30, "Mechanism\nCKA | size-stratified error"),
    (0.52, 0.10, 0.20, 0.30, "Deployment\nFP32 | FP16 | INT8"),
    (0.77, 0.10, 0.21, 0.30, "Artifacts\nprotocol, per-seed results,\nsplit lists"),
]
for x, y, w, h, txt in boxes:
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor="#eef3fa", edgecolor="#4C72B0", lw=1.2))
    ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=7.6)
for x0, x1, y in [(0.22,0.27,0.715),(0.47,0.52,0.715),(0.72,0.77,0.715)]:
    ax.annotate("", xy=(x1,y), xytext=(x0,y), arrowprops=dict(arrowstyle="->", color="#4C72B0"))
ax.annotate("", xy=(0.37,0.40), xytext=(0.37,0.55), arrowprops=dict(arrowstyle="->", color="#4C72B0"))
ax.annotate("", xy=(0.62,0.40), xytext=(0.62,0.55), arrowprops=dict(arrowstyle="->", color="#4C72B0"))
ax.text(0.5, 0.97, "Study design: matched-protocol evaluation of training strategies under data scarcity", ha="center", fontsize=8.5)
save(fig, "fig1_protocol")

# ---------- Fig 2: VisDrone scale curve ----------
frac = [10, 25, 50, 100]
map50 = [0.1853, 0.2276, 0.2508, 0.2780]
err = [0.0027, 0.0036, 0.0018, 0.0012]
fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.errorbar(frac, map50, yerr=err, marker="o", color="#4C72B0", capsize=3, lw=1.6)
ax.set_xlabel("VisDrone training data (%)"); ax.set_ylabel("mAP@0.5")
ax.set_xticks(frac); ax.set_title("Accuracy grows with labelled data", fontsize=9.5)
save(fig, "fig2_scale_curve")

# ---------- Fig 4: freezing ----------
fig, ax = plt.subplots(figsize=(4.4, 3.1))
x = [0, 10, 20]; y = [0.2106, 0.1696, 0.1162]; e = [0.0027, 0.0012, 0.0016]
ax.errorbar(x, y, yerr=e, marker="s", color="#C44E52", capsize=3, lw=1.6)
ax.set_xlabel("Number of frozen backbone modules"); ax.set_ylabel("mAP@0.5")
ax.set_xticks(x); ax.set_title("Freezing the backbone is harmful", fontsize=9.5)
save(fig, "fig4_freeze_curve")

# ---------- Fig 5: CKA ----------
layers, data = [], {}
with open(os.path.join(ROOT, "results_cka", "cka_vs_pretrained.csv"), encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        L = int(r["layer"]); data.setdefault(r["model"], {})[L] = float(r["cka_vs_pretrained"])
        if L not in layers: layers.append(L)
layers.sort()
fig, ax = plt.subplots(figsize=(4.8, 3.2))
colors = {"freeze10": "#DD8452", "freeze20": "#8172B2", "full_ft": "#C44E52"}
for m, vals in data.items():
    ax.plot(layers, [vals[L] for L in layers], marker="o", lw=1.6, label=m, color=colors.get(m, None))
ax.set_xlabel("Backbone stage index"); ax.set_ylabel("CKA vs. pretrained representation")
ax.set_xticks(layers); ax.set_ylim(0.5, 1.02); ax.legend(frameon=False, fontsize=8)
ax.set_title("Frozen models retain pretrained features", fontsize=9.5)
save(fig, "fig5_cka")

# ---------- Fig 6: size-stratified ----------
rows = []
with open(os.path.join(ROOT, "results_size", "size_stratified.csv"), encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["size_bin"] != "all"]
models = ["pretrained50", "full_ft", "freeze10", "freeze20"]
bins = ["small_lt32", "medium_32_96", "large_ge96"]
labels = ["small (<32$^2$)", "medium", "large ($\\geq$96$^2$)"]
fig, ax = plt.subplots(figsize=(5.4, 3.2))
w = 0.2
for i, m in enumerate(models):
    vals = [float(next(r["mAP50"] for r in rows if r["model"] == m and r["size_bin"] == b)) for b in bins]
    ax.bar(np.arange(len(bins)) + (i - 1.5) * w, vals, w, label=m)
ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels(labels)
ax.set_ylabel("mAP@0.5"); ax.legend(frameon=False, fontsize=7.5, ncol=2)
ax.set_title("The freezing penalty falls on small objects", fontsize=9.5)
save(fig, "fig6_size_stratified")

# ---------- Fig 7: deployment ----------
pts = [
    ("YOLOv8n FP32 (GPU)", 15.73, 0.2263, "#4C72B0", "o"),
    ("YOLOv8n FP16 (GPU)", 17.50, 0.2263, "#55A868", "s"),
    ("ONNX FP32 (CPU)", 39.7, 0.2270, "#8172B2", "^"),
    ("ONNX INT8 (CPU)", 71.9, 0.2245, "#C44E52", "D"),
]
fig, ax = plt.subplots(figsize=(5.0, 3.2))
for name, lat, acc, c, m in pts:
    ax.scatter(lat, acc, s=70, color=c, marker=m, edgecolor="white", lw=0.6, label=name)
    ax.annotate(name, (lat, acc), textcoords="offset points", xytext=(6, -2), fontsize=7, color=c)
ax.set_xlabel("Latency per image (ms)"); ax.set_ylabel("mAP@0.5")
ax.set_xscale("log"); ax.set_xlim(12, 100)
ax.set_title("Lower precision does not automatically mean lower latency", fontsize=9.5)
save(fig, "fig7_deployment")

# ---------- Fig 8: AI-TOD budgets ----------
bud = [647, 1617, 3235]
scr = [0.0347, 0.1423, 0.1994]; pre = [0.1748, 0.2424, 0.2886]
x = np.arange(len(bud)); w = 0.35
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.bar(x - w/2, scr, w, label="from scratch", color="#C44E52")
ax.bar(x + w/2, pre, w, label="COCO pretrained", color="#4C72B0")
for i in x:
    ax.annotate("+%.3f" % (pre[i] - scr[i]), (i, max(scr[i], pre[i]) + 0.012), ha="center", fontsize=8, color="#333333")
ax.set_xticks(x); ax.set_xticklabels(["%d imgs" % b for b in bud]); ax.set_ylabel("mAP@0.5")
ax.legend(frameon=False, fontsize=8); ax.set_ylim(0, 0.36)
ax.set_title("Smaller label budgets gain more from pretraining", fontsize=9.5)
save(fig, "fig8_aitod_budget")
print("ALL FIGURES DONE")
