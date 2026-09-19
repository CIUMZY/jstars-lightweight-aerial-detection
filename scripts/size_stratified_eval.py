# -*- coding: utf-8 -*-
"""尺寸分层评测 v3（用 pycocotools 标准实现，COCO 面积区间）。
用法：python size_stratified_eval.py --data <val yaml> --weights name=path [...] --outdir results_size
"""
import argparse, csv, json, os, tempfile
import cv2, numpy as np, yaml
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_coco_gt(data_yaml, out_json):
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    root = cfg.get("path", ""); val = cfg["val"]
    val_dir = val if os.path.isabs(val) else os.path.join(root, val)
    lab_dir = os.path.join(os.path.dirname(val_dir.rstrip("/\\")), "labels")
    names = cfg.get("names", {})
    cats = [{"id": i, "name": (names[i] if isinstance(names, dict) else names[i])} for i in range(len(names))]
    images, anns = [], []
    aid = 1
    for idx, f in enumerate(sorted(os.listdir(val_dir))):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        im = cv2.imread(os.path.join(val_dir, f))
        if im is None:
            continue
        h, w = im.shape[:2]
        iid = idx + 1
        images.append({"id": iid, "file_name": f, "width": w, "height": h})
        lp = os.path.join(lab_dir, os.path.splitext(f)[0] + ".txt")
        if not os.path.exists(lp):
            continue
        for line in open(lp, encoding="utf-8", errors="ignore"):
            s = line.split()
            if len(s) < 5:
                continue
            c = int(float(s[0])); cx, cy, bw, bh = map(float, s[1:5])
            x = (cx - bw / 2) * w; y = (cy - bh / 2) * h
            W = bw * w; H = bh * h
            if W <= 1 or H <= 1:
                continue
            anns.append({"id": aid, "image_id": iid, "category_id": c,
                         "bbox": [x, y, W, H], "area": float(W * H), "iscrowd": 0})
            aid += 1
    json.dump({"images": images, "annotations": anns, "categories": cats}, open(out_json, "w"))
    print("COCO GT: %d images, %d boxes -> %s" % (len(images), len(anns), out_json), flush=True)
    return list({im["file_name"]: im["id"] for im in images}.items())


def predict_results(weights, val_dir, name_to_id, imgsz=640, conf=0.001, iou=0.7):
    m = YOLO(weights)
    res = []
    files = sorted(os.listdir(val_dir))
    for k, f in enumerate(files, 1):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        iid = name_to_id.get(f, name_to_id.get(os.path.basename(f)))
        if iid is None:
            continue
        r = m.predict(os.path.join(val_dir, f), imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xy = r.boxes.xyxy.cpu().numpy(); sc = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy()
        for b, s, c in zip(xy, sc, cl):
            res.append({"image_id": int(iid), "category_id": int(c),
                        "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                        "score": float(s)})
        if k % 100 == 0:
            print("  %d images, %d dets" % (k, len(res)), flush=True)
    return res


def evaluate(gt_json, det_list, tag):
    coco = COCO(gt_json)
    if not det_list:
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(det_list, fh); det_path = fh.name
    coco_dt = coco.loadRes(det_path)
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.params.maxDets = [1, 10, 300]
    ev.evaluate(); ev.accumulate()
    # 手工取得各面积区间的 AP 与 AP50
    prec = ev.eval["precision"]  # [T,R,K,A,M]
    out = {}
    area_names = ["all", "small_lt32", "medium_32_96", "large_ge96"]
    for ai, an in enumerate(area_names):
        p = prec[:, :, :, ai, 2]
        p = p[p > -1]
        out[an] = {"mAP50_95": float(np.mean(p)) if p.size else float("nan")}
        p50 = prec[0, :, :, ai, 2]; p50 = p50[p50 > -1]
        out[an]["mAP50"] = float(np.mean(p50)) if p50.size else float("nan")
    print("[%s] " % tag + "  ".join("%s: mAP50=%.4f/mAP50-95=%.4f" % (k, v["mAP50"], v["mAP50_95"]) for k, v in out.items()), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", action="append", required=True, help="name=path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--outdir", default="results_size")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    cfg = yaml.safe_load(open(a.data, encoding="utf-8"))
    root = cfg.get("path", ""); val = cfg["val"]
    val_dir = val if os.path.isabs(val) else os.path.join(root, val)
    gt_json = os.path.join(a.outdir, "gt_coco.json")
    pairs = build_coco_gt(a.data, gt_json)
    name_to_id = {n: i for n, i in pairs}
    rows = []
    for spec in a.weights:
        name, path = spec.split("=", 1)
        print("inference:", name, flush=True)
        dets = predict_results(path, val_dir, name_to_id, imgsz=a.imgsz)
        res = evaluate(gt_json, dets, name)
        for bin_name, m in res.items():
            rows.append(dict(model=name, size_bin=bin_name, mAP50=m["mAP50"], mAP50_95=m["mAP50_95"]))
    with open(os.path.join(a.outdir, "size_stratified.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["model", "size_bin", "mAP50", "mAP50_95"]); w.writeheader(); w.writerows(rows)
    lines = ["# 尺寸分层评测（COCO 面积区间，pycocotools）", "", "| model | size bin | mAP@0.5 | mAP@0.5:0.95 |", "|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %.4f | %.4f |" % (r["model"], r["size_bin"], r["mAP50"], r["mAP50_95"]))
    open(os.path.join(a.outdir, "size_stratified.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("done ->", a.outdir, flush=True)


if __name__ == "__main__":
    main()
