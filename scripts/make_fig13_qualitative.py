# -*- coding: utf-8 -*-
"""定性检测面板：GT vs 从头训练 vs 推荐配置（两行图像 + 放大插图）。"""
import os, sys, glob
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_jstars import apply, save, C
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ultralytics import YOLO

apply()
ROOT = r"D:\Research\03_Codex\projects\yolo-paper"
OUT = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figures_jstars"
VAL = os.path.join(ROOT, "datasets", "VisDrone", "VisDrone2019-DET-val")
CLASSES = ["pedestrian","people","bicycle","car","van","truck","tricycle","awning-tricycle","bus","motor"]

# 选两张小目标密集的图（按标注数量排序）
cands = []
for f in sorted(os.listdir(os.path.join(VAL, "labels")))[:250]:
    lines = open(os.path.join(VAL, "labels", f), encoding="utf-8").read().strip().split("\n")
    cands.append((len(lines), f))
cands.sort(reverse=True)
picks = [f for _, f in cands[:2]]
print("picked:", picks, flush=True)

m_scratch = YOLO(os.path.join(ROOT, "runs_p0", "scratch50_p0", "weights", "best.pt"))
m_best = YOLO(os.path.join(ROOT, "runs_p0", "proposed100_p0", "weights", "best.pt"))

def draw_gt(img, lab_path):
    h, w = img.shape[:2]
    for line in open(lab_path, encoding="utf-8"):
        s = line.split()
        if len(s) < 5: continue
        c = int(float(s[0])); cx, cy, bw, bh = map(float, s[1:5])
        x1 = int((cx - bw/2)*w); y1 = int((cy - bh/2)*h); x2 = int((cx + bw/2)*w); y2 = int((cy + bh/2)*h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 1)
    return img

def draw_pred(img, model):
    r = model.predict(img, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
    if r.boxes is not None:
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 90, 210), 1)
    return img

fig, axes = plt.subplots(len(picks), 3, figsize=(7.0, 4.6))
for row, stem in enumerate(picks):
    ip = os.path.join(VAL, "images", stem.replace(".txt", ".jpg"))
    lp = os.path.join(VAL, "labels", stem)
    base = cv2.imread(ip)
    if base is None: continue
    ims = [draw_gt(base.copy(), lp), draw_pred(base.copy(), m_scratch), draw_pred(base.copy(), m_best)]
    ngt = len(open(lp).read().strip().split("\n"))
    titles = ["Ground truth (%d objects)" % ngt, "From scratch", "Recommended"]
    for col, (im, t) in enumerate(zip(ims, titles)):
        ax = axes[row, col]
        small = cv2.resize(im, (960, int(im.shape[0] * 960 / im.shape[1])), interpolation=cv2.INTER_AREA)
        ax.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#BBBBBB"); sp.set_linewidth(0.6)
        if row == 0: ax.set_title(t, fontsize=7.5, pad=3)
        if col == 0: ax.set_ylabel("(a)" if row == 0 else "(b)", fontsize=9)
fig.text(0.5, 0.015, "green: ground truth; orange: predictions at conf 0.25", ha="center", fontsize=6.5, color="#555555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
save(fig, OUT, "fig13_qualitative_detection")
