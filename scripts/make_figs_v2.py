# -*- coding: utf-8 -*-
"""重建主图 v2（JSTARS 审美：Arial、Okabe-Ito、无边框、误差带、600 dpi）。"""
import csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_jstars import apply, save, C, SEQ
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch

apply()
ROOT = r"D:\Research\03_Codex\projects\yolo-paper"
OUT = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figures_jstars"
FDATA = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figure_data"
os.makedirs(OUT, exist_ok=True)

# ---------- Fig 2: scale curve with shaded band ----------
frac = np.array([10, 25, 50, 100]); y = np.array([0.1853, 0.2276, 0.2508, 0.2780]); e = np.array([0.0027, 0.0036, 0.0018, 0.0012])
fig, ax = plt.subplots(figsize=(3.45, 2.5))
ax.plot(frac, y, "-o", color=C["blue"], markerfacecolor="white", markeredgewidth=1.2, zorder=3)
ax.fill_between(frac, y - e, y + e, color=C["blue"], alpha=0.15, lw=0, zorder=2)
ax.scatter([25], [0.2276], s=55, facecolors="none", edgecolors=C["orange"], linewidths=1.3, zorder=4)
ax.annotate("25% operating point\n0.2276", xy=(25, 0.2276), xytext=(38, 0.203), fontsize=6.8, color=C["orange"],
            arrowprops=dict(arrowstyle="-", color=C["orange"], lw=0.7))
ax.set_xlabel("VisDrone training data (%)"); ax.set_ylabel("mAP@0.5"); ax.set_xticks(frac)
ax.set_ylim(0.17, 0.295); ax.grid(axis="y"); ax.set_axisbelow(True)
save(fig, OUT, "fig2_scale_curve")

# ---------- Fig 3: difficulty law ----------
pts = [("NEU-DET\n(1458, easy)", 0.6942, 0.0484, C["blue"], "o"),
       ("AI-TOD\n(3235)", 0.1994, 0.0892, C["purple"], "P"),
       ("AI-TOD\n(1617)", 0.1440, 0.1078, C["yellow"], "s"),
       ("VisDrone\n(1617)", 0.1011, 0.1095, C["green"], "^"),
       ("AI-TOD\n(647)", 0.0347, 0.1401, C["orange"], "D")]
x = np.array([p[1] for p in pts]); yy = np.array([p[2] for p in pts])
r = np.corrcoef(x, yy)[0, 1]; slope, intercept = np.polyfit(x, yy, 1)
xs = np.linspace(0.0, 0.75, 100); pred = slope * xs + intercept
resid = yy - (slope * x + intercept); s = np.sqrt(np.sum(resid**2) / (len(x) - 2))
band = s * np.sqrt(1/len(x) + (xs - x.mean())**2 / np.sum((x - x.mean())**2)) * 2.78
fig, ax = plt.subplots(figsize=(3.5, 2.7))
ax.fill_between(xs, pred - band, pred + band, color=C["grey"], alpha=0.15, lw=0)
ax.plot(xs, pred, "--", color=C["grey"], lw=1.1, zorder=1)
for name, xv, yv, col, mk in pts:
    ax.scatter(xv, yv, s=42, color=col, marker=mk, edgecolor="white", linewidth=0.7, zorder=3)
ax.annotate("NEU-DET (1458, easy)", (0.6942, 0.0484), xytext=(0.40, 0.031), fontsize=6.4,
            color=C["blue"], arrowprops=dict(arrowstyle="-", color=C["blue"], lw=0.6))
ax.annotate("AI-TOD 3235", (0.1994, 0.0892), xytext=(0.235, 0.072), fontsize=6.4, color=C["purple"])
ax.annotate("AI-TOD 1617", (0.1440, 0.1078), xytext=(0.155, 0.122), fontsize=6.4, color=C["yellow"])
ax.annotate("VisDrone 1617", (0.1011, 0.1095), xytext=(0.055, 0.126), fontsize=6.4, color=C["green"])
ax.annotate("AI-TOD 647", (0.0347, 0.1401), xytext=(0.10, 0.148), fontsize=6.4, color=C["orange"])
ax.text(0.52, 0.115, "r = %.2f" % r, fontsize=7.5, color=C["dark"])
ax.set_xlabel("mAP@0.5 trained from scratch"); ax.set_ylabel("Pretraining gain (mAP@0.5)")
ax.set_xlim(-0.02, 0.78); ax.set_ylim(0.02, 0.165); ax.grid(axis="y"); ax.set_axisbelow(True)
save(fig, OUT, "fig3_pretrain_gain_vs_scratch")

# ---------- Fig 4: freezing ----------
xf = np.array([0, 10, 20]); yf = np.array([0.2106, 0.1696, 0.1162]); ef = np.array([0.0027, 0.0012, 0.0016])
fig, ax = plt.subplots(figsize=(3.3, 2.4))
ax.bar(xf, yf, width=5.2, color=[C["green"], C["yellow"], C["orange"]], alpha=0.92, edgecolor="white", linewidth=0.5)
for xi, yi, ei in zip(xf, yf, ef):
    ax.errorbar(xi, yi, yerr=ei, fmt="none", ecolor="#3A3A3A", elinewidth=0.8, capsize=2)
    ax.text(xi, yi + 0.008, "%.4f" % yi, ha="center", fontsize=7)
ax.annotate("", xy=(10, 0.185), xytext=(0, 0.219), arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.0))
ax.text(5.0, 0.207, "$-0.041$", fontsize=7, color=C["orange"], ha="center")
ax.annotate("", xy=(20, 0.128), xytext=(10, 0.180), arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.0))
ax.text(15.0, 0.160, "$-0.053$", fontsize=7, color=C["orange"], ha="center")
ax.set_xlabel("Frozen backbone modules"); ax.set_ylabel("mAP@0.5"); ax.set_xticks(xf); ax.set_ylim(0, 0.25)
ax.grid(axis="y"); ax.set_axisbelow(True)
save(fig, OUT, "fig4_freeze_curve")

# ---------- Fig 5: CKA ----------
layers, data = [], {}
with open(os.path.join(ROOT, "results_cka", "cka_vs_pretrained.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        L = int(row["layer"]); data.setdefault(row["model"], {})[L] = float(row["cka_vs_pretrained"])
        if L not in layers: layers.append(L)
layers.sort()
fig, ax = plt.subplots(figsize=(3.5, 2.5))
styles = {"freeze10": (C["yellow"], "s", "Freeze 10"), "freeze20": (C["purple"], "P", "Freeze 20"), "full_ft": (C["orange"], "D", "Full fine-tuning")}
for m, (col, mk, lab) in styles.items():
    if m in data:
        ax.plot(layers, [data[m][L] for L in layers], "-", marker=mk, color=col, markerfacecolor="white",
                markeredgewidth=1.1, label=lab)
ax.annotate("deepest stage:\n0.605", xy=(9, 0.605), xytext=(5.6, 0.63), fontsize=6.6, color=C["orange"],
            arrowprops=dict(arrowstyle="-", color=C["orange"], lw=0.6))
ax.set_xlabel("Backbone stage index"); ax.set_ylabel("CKA vs. pretrained representation")
ax.set_xticks(layers); ax.set_ylim(0.5, 1.03); ax.legend(loc="lower left"); ax.grid(axis="y"); ax.set_axisbelow(True)
save(fig, OUT, "fig5_cka")

# ---------- Fig 6: size-stratified ----------
rows = []
with open(os.path.join(ROOT, "results_size", "size_stratified.csv"), encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["size_bin"] != "all"]
models = ["pretrained50", "full_ft", "freeze10", "freeze20"]
mlabel = {"pretrained50": "Pretrained 50 ep", "full_ft": "Full fine-tuning", "freeze10": "Freeze 10", "freeze20": "Freeze 20"}
bins = ["small_lt32", "medium_32_96", "large_ge96"]; blabel = ["small\n($<32^2$ px)", "medium", "large\n($\\geq 96^2$ px)"]
fig, ax = plt.subplots(figsize=(3.6, 2.6)); w = 0.2
for i, m in enumerate(models):
    vals = [float(next(r["mAP50"] for r in rows if r["model"] == m and r["size_bin"] == b)) for b in bins]
    ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=mlabel[m], color=SEQ[i], edgecolor="white", linewidth=0.4)
ax.set_xticks(range(3)); ax.set_xticklabels(blabel); ax.set_ylabel("mAP@0.5")
ax.legend(ncol=2, loc="upper left"); ax.grid(axis="y"); ax.set_axisbelow(True)
save(fig, OUT, "fig6_size_stratified")

# ---------- Fig 7: deployment bubble ----------
items = [("YOLOv8n FP32 (GPU)", 15.73, 0.2263, 3.0, C["blue"], "o"),
         ("YOLOv8n FP16 (GPU)", 17.50, 0.2263, 11.1, C["sky"], "s"),
         ("YOLOv8s FP32 (GPU)", 17.03, 0.2856, 3.2, C["green"], "^"),
         ("YOLOv5n FP32 (GPU)", 17.87, 0.2095, 2.5, C["yellow"], "v"),
         ("ONNX FP32 (CPU)", 39.7, 0.2270, 11.7, C["purple"], "P"),
         ("ONNX INT8 (CPU)", 71.9, 0.2245, 3.3, C["orange"], "D")]
fig, ax = plt.subplots(figsize=(3.6, 2.7))
for name, lat, acc, size, col, mk in items:
    ax.scatter(lat, acc, s=28 + size * 9, color=col, marker=mk, alpha=0.9, edgecolor="white", linewidth=0.7, zorder=3)
ax.annotate("YOLOv8n FP32\n63.6 FPS", (15.73, 0.2263), xytext=(18.5, 0.245), fontsize=6.2, color=C["blue"],
            arrowprops=dict(arrowstyle="-", color=C["blue"], lw=0.6))
ax.annotate("FP16: slower,\nsame accuracy", (17.50, 0.2263), xytext=(22.0, 0.207), fontsize=6.2, color=C["sky"],
            arrowprops=dict(arrowstyle="-", color=C["sky"], lw=0.6))
ax.annotate("INT8: 3.5x smaller,\nnot faster on CPU", (71.9, 0.2245), xytext=(36.0, 0.203), fontsize=6.2, color=C["orange"],
            arrowprops=dict(arrowstyle="-", color=C["orange"], lw=0.6))
ax.annotate("YOLOv8s: best accuracy,\nsimilar latency", (17.03, 0.2856), xytext=(21.0, 0.272), fontsize=6.2, color=C["green"],
            arrowprops=dict(arrowstyle="-", color=C["green"], lw=0.6))
ax.set_xscale("log"); ax.set_xlim(13, 100); ax.set_ylim(0.195, 0.30)
ax.set_xlabel("Latency per image (ms, log scale)"); ax.set_ylabel("mAP@0.5")
ax.grid(axis="y"); ax.set_axisbelow(True)
save(fig, OUT, "fig7_deployment")

# ---------- Fig 8: AI-TOD dumbbell ----------
bud = ["647", "1617", "3235"]; scr = np.array([0.0347, 0.1440, 0.1994]); pre = np.array([0.1748, 0.2518, 0.2886])
fig, ax = plt.subplots(figsize=(3.6, 2.4)); yy = np.arange(len(bud))
for i in range(len(bud)):
    ax.plot([scr[i], pre[i]], [yy[i], yy[i]], color=C["light"], lw=2.4, zorder=1, solid_capstyle="round")
ax.scatter(scr, yy, s=38, color=C["orange"], zorder=3, label="from scratch", edgecolor="white", linewidth=0.6)
ax.scatter(pre, yy, s=38, color=C["blue"], zorder=3, label="pretrained", edgecolor="white", linewidth=0.6)
for i in range(len(bud)):
    ax.text(pre[i] + 0.012, yy[i], "+%.4f" % (pre[i] - scr[i]), va="center", fontsize=6.8, color=C["dark"])
ax.set_yticks(yy); ax.set_yticklabels(["%s images" % b for b in bud]); ax.set_xlabel("mAP@0.5")
ax.set_xlim(0, 0.36); ax.invert_yaxis(); ax.legend(loc="lower right"); ax.grid(axis="x"); ax.set_axisbelow(True)
save(fig, OUT, "fig8_aitod_budget")

# ---------- Fig 9: clean 3D bars ----------
fig = plt.figure(figsize=(3.9, 3.1))
ax = fig.add_subplot(111, projection="3d")
xs = np.arange(3)
for i, xv in enumerate(xs):
    ax.bar3d(xv - 0.34, 0, 0, 0.3, 0.62, scr[i], color=C["orange"], alpha=0.95, edgecolor="white", linewidth=0.3)
    ax.bar3d(xv + 0.04, 0, 0, 0.3, 0.62, pre[i], color=C["blue"], alpha=0.95, edgecolor="white", linewidth=0.3)
    ax.text(xv, 0.0, max(scr[i], pre[i]) + 0.03, "+%.3f" % (pre[i] - scr[i]), ha="center", fontsize=6.4, color=C["dark"])
ax.set_xticks(xs); ax.set_xticklabels(["647", "1617", "3235"], fontsize=7)
ax.set_yticks([]); ax.set_zlim(0, 0.36)
ax.set_xlabel("AI-TOD label budget", fontsize=7.5, labelpad=4)
ax.set_zlabel("mAP@0.5", fontsize=7.5, labelpad=2)
ax.view_init(elev=20, azim=-62)
ax.set_title("Pretraining gain shrinks as labels grow", fontsize=8.5, pad=2)
ax.legend(handles=[Patch(facecolor=C["orange"], label="from scratch"), Patch(facecolor=C["blue"], label="pretrained")],
          fontsize=6.5, loc="upper left", bbox_to_anchor=(-0.05, 0.95))
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_alpha(0.04); pane.pane.set_edgecolor("#DDDDDD")
ax.grid(False)
save(fig, OUT, "fig9_aitod_3d_bars")

# ---------- Fig 10: per-class heatmap ----------
CLASSES = ["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]
ORDER = ["from_scratch", "pretrained_50ep", "freeze_10", "freeze_20", "recommended"]
LAB = {"from_scratch": "From scratch", "pretrained_50ep": "Pretrained 50 ep", "freeze_10": "Freeze 10",
       "freeze_20": "Freeze 20", "recommended": "Recommended"}
rows2 = {}
for f in glob.glob(os.path.join(FDATA, "perclass_*.csv")):
    nm = os.path.basename(f)[len("perclass_"):-len(".csv")]
    if nm in ORDER:
        with open(f, encoding="utf-8-sig") as fh:
            rows2[nm] = [float(list(csv.DictReader(fh))[0][c]) for c in CLASSES]
models2 = [m for m in ORDER if m in rows2]
if models2:
    M = np.array([rows2[m] for m in models2])
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    im = ax.imshow(M, aspect="auto", cmap="cividis", vmin=0, vmax=0.72)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=32, ha="right", fontsize=7)
    ax.set_yticks(range(len(models2))); ax.set_yticklabels([LAB[m] for m in models2], fontsize=7.5)
    ax.set_xticks(np.arange(-0.5, len(CLASSES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models2), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8); ax.tick_params(which="minor", length=0)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] < 0.38 else "#1A1A1A")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.012); cb.set_label("AP@0.5", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    ax.set_title("Per-class accuracy across training strategies", fontsize=8.5, pad=6)
    save(fig, OUT, "fig10_perclass_heatmap")

    # radar (clean)
    ang = np.linspace(0, 2 * np.pi, len(CLASSES), endpoint=False).tolist(); ang += ang[:1]
    fig = plt.figure(figsize=(4.2, 3.9)); ax = plt.subplot(111, polar=True)
    for i, m in enumerate(models2):
        v = rows2[m] + rows2[m][:1]
        ax.plot(ang, v, lw=1.3, label=LAB[m], color=SEQ[i], marker="o", markersize=2.2, markerfacecolor="white", markeredgewidth=0.6)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(CLASSES, fontsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5); ax.set_ylim(0, 0.75)
    ax.set_rlabel_position(150); ax.grid(alpha=0.3, lw=0.5)
    ax.spines["polar"].set_color("#CCCCCC"); ax.spines["polar"].set_linewidth(0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, fontsize=6.5)
    save(fig, OUT, "fig11_perclass_radar")

# ---------- Fig 12: confusion matrix ----------
cmf = os.path.join(FDATA, "confusion_recommended.csv")
if os.path.exists(cmf):
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
                ax.text(j, i, "%.2f" % An[i, j], ha="center", va="center", fontsize=5.4,
                        color="white" if An[i, j] > 0.5 else "#333333")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Row-normalised frequency", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    ax.set_title("Confusion matrix of the recommended configuration", fontsize=8.5, pad=6)
    save(fig, OUT, "fig12_confusion_matrix")
print("ALL V2 FIGURES DONE")
