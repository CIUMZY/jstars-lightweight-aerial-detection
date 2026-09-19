# -*- coding: utf-8 -*-
"""ONNX Runtime INT8（动态量化）评测：量化 best.onnx -> 精度(mAP) + CPU 延迟对比。"""
import os, time, glob
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType
from ultralytics import YOLO

ROOT = r"D:\Research\03_Codex\projects\yolo-paper"
W = os.path.join(ROOT, "runs_p0", "proposed100_p0", "weights")
SRC = os.path.join(W, "best.onnx")
DST = os.path.join(W, "best_int8.onnx")
DATA = os.path.join(ROOT, "datasets", "VisDrone", "visdrone25.yaml")
OUT = os.path.join(ROOT, "results_deploy")
VAL_IMG = os.path.join(ROOT, "datasets", "VisDrone", "VisDrone2019-DET-val", "images")
os.makedirs(OUT, exist_ok=True)

def log(m):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), m)
    open(os.path.join(OUT, "int8_onnx.log"), "a", encoding="utf-8").write(line + "\n")
    print(line, flush=True)

log("quantize_dynamic start")
quantize_dynamic(SRC, DST, weight_type=QuantType.QUInt8)
log("quantized -> %s (%.1f MB)" % (DST, os.path.getsize(DST) / 1e6))

imgs = sorted(glob.glob(os.path.join(VAL_IMG, "*")))[:200]
def bench(model_path, tag):
    m = YOLO(model_path, task="detect")
    for p in imgs[:3]:
        m.predict(p, imgsz=640, device="cpu", verbose=False)
    lat = []
    for p in imgs[:50]:
        t0 = time.perf_counter()
        m.predict(p, imgsz=640, device="cpu", verbose=False)
        lat.append((time.perf_counter() - t0) * 1000)
    log("%s CPU latency: %.1f ± %.1f ms (%.1f FPS)" % (tag, np.mean(lat), np.std(lat), 1000.0 / np.mean(lat)))
    r = YOLO(model_path, task="detect").val(data=DATA, imgsz=640, batch=1, device="cpu", verbose=False)
    log("%s mAP50=%.4f mAP50-95=%.4f" % (tag, r.box.map50, r.box.map))
    return float(np.mean(lat)), float(r.box.map50), float(r.box.map)

rows = []
rows.append(("FP32 ONNX (CPU)",) + bench(SRC, "FP32-ONNX"))
rows.append(("INT8 ONNX (CPU)",) + bench(DST, "INT8-ONNX"))
with open(os.path.join(OUT, "int8_onnx.md"), "w", encoding="utf-8") as f:
    f.write("# ONNX Runtime 精度/延迟对比（CPU 端）\n\n| 模型 | CPU 延迟(ms) | mAP@0.5 | mAP@0.5:0.95 |\n|---|---|---|---|\n")
    for name, lat, m50, m95 in rows:
        f.write("| %s | %.1f | %.4f | %.4f |\n" % (name, lat, m50, m95))
log("done")
