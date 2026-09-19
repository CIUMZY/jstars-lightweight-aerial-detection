# -*- coding: utf-8 -*-
"""渐进解冻训练：先冻结骨干，训练中由深到浅逐步解冻（Ultralytics callback 实现）。
用法示例：
  python staged_unfreeze.py --name staged_linear_s1_p0 --schedule linear --seed 1 ^
      --data datasets/VisDrone/visdrone25_iv.yaml --official datasets/VisDrone/visdrone25.yaml
"""
import argparse, os, sys, time
from multiprocessing import freeze_support

ROOT = r'D:\Research\03_Codex\projects\yolo-paper'
OUT = os.path.join(ROOT, 'runs_p0')
SUM1 = os.path.join(OUT, 'summary_seed_extra.txt')
SUM2 = os.path.join(OUT, 'summary_remaining_extra.txt')
N_BACKBONE = 10  # YOLOv8n 骨干层索引 0..9

def make_callback(schedule, epochs, start_frac=0.0, end_frac=0.6):
    state = {"k": 0}
    def set_frozen(trainer, k):
        """k = 已解冻的骨干层数（从最深层开始解冻）"""
        net = trainer.model.model  # Sequential
        start_idx = N_BACKBONE - k
        for i, layer in enumerate(net):
            if i >= N_BACKBONE:
                continue
            frozen = i < start_idx
            for prm in layer.parameters():
                prm.requires_grad = not frozen
    def cb(trainer):
        ep = int(getattr(trainer, "epoch", 0))
        frac = ep / max(1, epochs)
        if schedule == "linear":
            k = int(round(N_BACKBONE * min(1.0, max(0.0, (frac - start_frac) / max(1e-6, end_frac - start_frac)))))
        elif schedule == "late":
            k = 0 if frac < start_frac else N_BACKBONE
        else:
            k = N_BACKBONE
        if k != state["k"]:
            state["k"] = k
            set_frozen(trainer, k)
            print("[staged_unfreeze] epoch=%d frac=%.2f unfrozen_backbone_layers=%d" % (ep, frac, k), flush=True)
    return cb, set_frozen

def main(a):
    from ultralytics import YOLO
    run_dir = os.path.join(OUT, a.name)
    best = os.path.join(run_dir, "weights", "best.pt")
    if not os.path.exists(best):
        print("TRAIN start %s (schedule=%s seed=%s)" % (a.name, a.schedule, a.seed), flush=True)
        m = YOLO(a.model)
        cb, set_frozen = make_callback(a.schedule, a.epochs, a.start_frac, a.end_frac)
        m.add_callback("on_train_epoch_start", cb)
        kw = dict(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, workers=0, device=0,
                  close_mosaic=10, optimizer="AdamW", lr0=0.000714, momentum=0.9, warmup_bias_lr=0.0,
                  seed=a.seed, exist_ok=True, project=OUT, name=a.name, cls_remap=True,
                  mixup=a.mixup, cutmix=0.0, deterministic=True, pretrained=True)
        m.train(**kw)
        print("TRAIN done %s" % a.name, flush=True)
    if os.path.exists(best):
        print("VAL official %s" % a.name, flush=True)
        r = YOLO(best).val(data=a.official, imgsz=640, batch=16, device=0, verbose=False)
        line = "%s\tmAP50=%.4f\tmAP50-95=%.4f" % (a.name, r.box.map50, r.box.map)
        print(line, flush=True)
        for f in (SUM1, SUM2):
            open(f, "a", encoding="utf-8").write(line + "\n")
        return 0
    return 1

if __name__ == "__main__":
    freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--model", default=os.path.join(ROOT, "yolov8n.pt"))
    ap.add_argument("--data", required=True); ap.add_argument("--official", required=True)
    ap.add_argument("--epochs", type=int, default=100); ap.add_argument("--mixup", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--schedule", default="linear", choices=["linear", "late"])
    ap.add_argument("--start-frac", type=float, default=0.0); ap.add_argument("--end-frac", type=float, default=0.6)
    sys.exit(main(ap.parse_args()))
