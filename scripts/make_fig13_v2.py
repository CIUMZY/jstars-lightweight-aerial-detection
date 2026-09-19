# -*- coding: utf-8 -*-
"""定性检测面板 v2：GT / 从头训练 / 推荐配置 + 小目标区域放大插图。"""
import os, sys
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

cands = []
for f in sorted(os.listdir(os.path.join(VAL, "labels")))[:250]:
    n = len([l for l in open(os.path.join(VAL, "labels", f), encoding="utf-8").read().strip().split("\n") if l.strip()])
    cands.append((n, f))
cands.sort(reverse=True)
picks = [f for _, f in cands[:2]]

m_scratch = YOLO(os.path.join(ROOT, "runs_p0", "scratch50_p0", "weights", "best.pt"))
m_best = YOLO(os.path.join(ROOT, "runs_p0", "proposed100_p0", "weights", "best.pt"))

def read_gt(lab_path, w, h):
    out = []
    for line in open(lab_path, encoding="utf-8"):
        s = line.split()
        if len(s) < 5: continue
        c = int(float(s[0])); cx, cy, bw, bh = map(float, s[1:5])
        out.append((int((cx-bw/2)*w), int((cy-bh/2)*h), int((cx+bw/2)*w), int((cy+bh/2)*h)))
    return out

def draw_boxes(img, boxes, color, thickness=1):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def pred_boxes(model, img):
    r = model.predict(img, imgsz=640, conf=0.25, iou=0.7, verbose=False)[0]
    if r.boxes is None: return []
    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in r.boxes.xyxy.cpu().numpy()]

def densest_region(boxes, w, h, grid=12):
    """用粗网格统计小框中心密度，返回密度最高区域的边界框。"""
    counts = np.zeros((grid, grid))
    for (x1, y1, x2, y2) in boxes:
        if (x2-x1) * (y2-y1) > 4000:   # 只考虑小目标
            continue
        cx, cy = (x1+x2)/2, (y1+y2)/2
        counts[min(grid-1, int(cy/h*grid)), min(grid-1, int(cx/w*grid))] += 1
    if counts.max() == 0:
        return 0, 0, w//3, h//3
    iy, ix = np.unravel_index(np.argmax(counts), counts.shape)
    cw, ch = w//3, h//3
    x0 = int(np.clip(ix/grid*w - cw/2, 0, w-cw)); y0 = int(np.clip(iy/grid*h - ch/2, 0, h-ch))
    return x0, y0, cw, ch

fig, axes = plt.subplots(len(picks), 3, figsize=(7.0, 5.0))
for row, stem in enumerate(picks):
    ip = os.path.join(VAL, "images", stem.replace(".txt", ".jpg"))
    lp = os.path.join(VAL, "labels", stem)
    base = cv2.imread(ip); h, w = base.shape[:2]
    gt = read_gt(lp, w, h)
    ims = [draw_boxes(base.copy(), gt, (0, 170, 0)),
           draw_boxes(base.copy(), pred_boxes(m_scratch, base), (0, 90, 210)),
           draw_boxes(base.copy(), pred_boxes(m_best, base), (0, 90, 210))]
    x0, y0, cw, ch = densest_region(gt, w, h)
    titles = ["Ground truth (%d objects)" % len(gt), "From scratch", "Recommended"]
    for col, (im, t) in enumerate(zip(ims, titles)):
        ax = axes[row, col]
        small = cv2.resize(im, (960, int(h * 960 / w)), interpolation=cv2.INTER_AREA)
        ax.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#BBBBBB"); sp.set_linewidth(0.6)
        if row == 0: ax.set_title(t, fontsize=7.5, pad=3)
        if col == 2:
            sx = 960 / w
            ax.add_patch(Rectangle((x0*sx, y0*sx), cw*sx, ch*sx, fill=False, edgecolor="#D55E00", lw=1.0))
            axin = ax.inset_axes([0.60, 0.02, 0.38, 0.38])
            crop = cv2.cvtColor(im[y0:y0+ch, x0:x0+cw], cv2.COLOR_BGR2RGB)
            axin.imshow(crop); axin.set_xticks([]); axin.set_yticks([])
            for sp in axin.spines.values(): sp.set_color("#D55E00"); sp.set_linewidth(0.9)
fig.text(0.5, 0.006, "green: ground truth; orange: predictions at confidence 0.25; insets show the densest small-object region",
         ha="center", fontsize=6.3, color="#555555")
fig.tight_layout(rect=(0, 0.02, 1, 1))
save(fig, OUT, "fig13_qualitative_detection")
