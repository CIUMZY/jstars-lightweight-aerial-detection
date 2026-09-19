# -*- coding: utf-8 -*-
"""部署与效率评测（端到端口径：含预处理 + NMS）。
FP32 / FP16 用 Ultralytics 预测器整条管线计时；可选导出 ONNX / TensorRT(INT8)。
用法见 README.md。
"""
import argparse, csv, glob, os, statistics as st, time
import torch, yaml
from ultralytics import YOLO


def pick_images(data_yaml, n=60):
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    root = cfg.get("path", ""); val = cfg["val"]
    val_dir = val if os.path.isabs(val) else os.path.join(root, val)
    files = [p for p in sorted(glob.glob(os.path.join(val_dir, "*")))
             if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    return files[:n] if n else files


def bench_predictor(weights, images, imgsz, half, warmup=5, repeats=2, device=0):
    """端到端：predict() 含 letterbox 预处理 + 推理 + NMS；按图计时。"""
    model = YOLO(weights)
    def once(p):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(p, imgsz=imgsz, half=half, conf=0.25, iou=0.7, device=device, verbose=False)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000
    torch.cuda.reset_peak_memory_stats()
    for p in images[:warmup]:
        once(p)
    lat = []
    for _ in range(repeats):
        for p in images:
            lat.append(once(p))
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return dict(precision="FP16" if half else "FP32", n_images=len(images) * repeats,
                latency_ms=st.mean(lat), latency_std=st.pstdev(lat), fps=1000.0 / st.mean(lat),
                peak_mem_mb=peak)


def accuracy(weights, data_yaml, imgsz, half=False):
    r = YOLO(weights).val(data=data_yaml, imgsz=imgsz, batch=16, half=half, device=0, verbose=False)
    return float(r.box.map50), float(r.box.map)


def try_export(weights, data_yaml, imgsz, fmt, half=False, int8=False):
    try:
        path = YOLO(weights).export(format=fmt, imgsz=imgsz, half=half, int8=int8, data=data_yaml)
        return str(path)
    except Exception as e:
        print(f"[export {fmt} half={half} int8={int8}] failed: {type(e).__name__}: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", action="append", required=True, help="name=path")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n-images", type=int, default=60)
    ap.add_argument("--outdir", default="results_deploy")
    ap.add_argument("--try-export", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    images = pick_images(a.data, a.n_images)
    print("benchmark images:", len(images), flush=True)
    rows = []
    for spec in a.weights:
        name, path = spec.split("=", 1)
        for half in (False, True):
            b = bench_predictor(path, images, a.imgsz, half)
            m50, m5095 = accuracy(path, a.data, a.imgsz, half=half)
            rows.append(dict(model=name, backend="PyTorch-e2e", **b, mAP50=m50, mAP50_95=m5095))
            print("%-6s %-4s %.2f ms (%.1f FPS, peak %.0f MB) mAP50=%.4f" %
                  (name, b["precision"], b["latency_ms"], b["fps"], b["peak_mem_mb"], m50), flush=True)
        if a.try_export:
            for fmt, h, i8 in (("onnx", False, False), ("onnx", True, False),
                               ("engine", False, False), ("engine", True, False), ("engine", True, True)):
                out = try_export(path, a.data, a.imgsz, fmt, half=h, int8=i8)
                if out:
                    rows.append(dict(model=name, backend=f"{fmt}{'-int8' if i8 else ('-fp16' if h else '-fp32')}",
                                     precision="INT8" if i8 else ("FP16" if h else "FP32"),
                                     n_images=0, latency_ms=float("nan"), latency_std=float("nan"),
                                     fps=float("nan"), peak_mem_mb=float("nan"),
                                     mAP50=float("nan"), mAP50_95=float("nan"), artifact=out))
    with open(os.path.join(a.outdir, "deploy_bench.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    lines = ["# 部署与效率评测（端到端口径）", "",
             "| model | backend | precision | latency(ms) | FPS | peak mem(MB) | mAP@0.5 | mAP@0.5:0.95 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %.2f | %.1f | %.0f | %.4f | %.4f |" % (
            r["model"], r["backend"], r["precision"], r["latency_ms"], r["fps"],
            r["peak_mem_mb"], r["mAP50"], r["mAP50_95"]))
    open(os.path.join(a.outdir, "deploy_bench.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("done ->", a.outdir, flush=True)


if __name__ == "__main__":
    main()
