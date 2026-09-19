# -*- coding: utf-8 -*-
"""CKA 表征分析：比较 COCO 预训练、冻结微调、全量微调模型的骨干表征相似度。

用法（示例，先不要跑）：
  python cka_analysis.py --data datasets/VisDrone/visdrone25.yaml --n-images 128 ^
      --model pretrained=yolov8n.pt ^
      --model freeze10=runs_p0/freeze10_p0/weights/best.pt ^
      --model freeze20=runs_p0/freeze20_p0/weights/best.pt ^
      --model full_ft=runs_p0/proposed100_p0/weights/best.pt ^
      --outdir results_cka

输出：
  cka_matrix.csv        每个 layer 的两两 CKA
  cka_vs_pretrained.csv 每个微调模型相对预训练模型的 CKA（论文主图数据）
  cka_summary.md        可读小结
"""
import argparse, csv, os
import cv2, numpy as np, torch, yaml
from ultralytics import YOLO

DEFAULT_LAYERS = [2, 4, 6, 9]  # YOLOv8n backbone 的 C2/C3/C4/C5 输出层索引


def load_val_images(data_yaml, n):
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    root = cfg.get("path", "")
    val = cfg["val"]
    val_dir = val if os.path.isabs(val) else os.path.join(root, val)
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    files = []
    for name in sorted(os.listdir(val_dir)):
        if name.lower().endswith(exts):
            files.append(os.path.join(val_dir, name))
    return files[:n]


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def extract_layer_features(model, image_paths, layer_indices, device="cuda"):
    net = model.model.to(device).eval()
    feats = {i: [] for i in layer_indices}
    hooks = []

    def make_hook(idx):
        def hook(_module, _inp, out):
            t = out[0] if isinstance(out, (list, tuple)) else out
            t = t.detach().float()
            feats[idx].append(torch.nn.functional.adaptive_avg_pool2d(t, 1).flatten().cpu().numpy())
        return hook

    for i in layer_indices:
        hooks.append(net.model[i].register_forward_hook(make_hook(i)))
    with torch.no_grad():
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue
            x = letterbox(img, 640)[:, :, ::-1].transpose(2, 0, 1).copy()
            x = torch.from_numpy(x).float().unsqueeze(0).to(device) / 255.0
            net(x)
    for h in hooks:
        h.remove()
    return {i: np.stack(v) for i, v in feats.items() if v}


def cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xty = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    xtx = np.linalg.norm(X.T @ X, ord="fro")
    yty = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(xty / (xtx * yty)) if xtx > 0 and yty > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=path，可重复")
    ap.add_argument("--layers", default=",".join(map(str, DEFAULT_LAYERS)))
    ap.add_argument("--n-images", type=int, default=128)
    ap.add_argument("--outdir", default="results_cka")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    layers = [int(x) for x in a.layers.split(",")]
    models = [m.split("=", 1) for m in a.model]
    images = load_val_images(a.data, a.n_images)
    print("images:", len(images), "layers:", layers)

    feats = {}
    for name, path in models:
        print("extracting:", name, path)
        feats[name] = extract_layer_features(YOLO(path), images, layers)

    names = [n for n, _ in models]
    with open(os.path.join(a.outdir, "cka_matrix.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["layer", "model_a", "model_b", "cka"])
        for L in layers:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a_, b_ = names[i], names[j]
                    if L in feats[a_] and L in feats[b_]:
                        w.writerow([L, a_, b_, "%.4f" % cka(feats[a_][L], feats[b_][L])])

    ref = names[0]
    with open(os.path.join(a.outdir, "cka_vs_pretrained.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["layer", "model", "cka_vs_" + ref])
        for L in layers:
            for n in names[1:]:
                if L in feats[ref] and L in feats[n]:
                    w.writerow([L, n, "%.4f" % cka(feats[ref][L], feats[n][L])])

    lines = ["# CKA 表征分析结果", "", "参考模型: %s" % ref, "", "| layer | model | CKA vs %s |" % ref, "|---|---|---|"]
    for L in layers:
        for n in names[1:]:
            if L in feats[ref] and L in feats[n]:
                lines.append("| %d | %s | %.4f |" % (L, n, cka(feats[ref][L], feats[n][L])))
    open(os.path.join(a.outdir, "cka_summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("done ->", a.outdir)


if __name__ == "__main__":
    main()
