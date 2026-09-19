# -*- coding: utf-8 -*-
"""单个训练任务：训练 + 官方val评测（workers=0）。完整性按 results.csv 行数 >= epochs 判定。"""
import os, sys, shutil, argparse
from multiprocessing import freeze_support

ROOT = r'D:\Research\03_Codex\projects\yolo-paper'
OUT  = os.path.join(ROOT, 'runs_p0')
SUM1 = os.path.join(OUT, 'summary_seed_extra.txt')
SUM2 = os.path.join(OUT, 'summary_remaining_extra.txt')

def epochs_done(run_dir):
    rc = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(rc): return 0
    try: return sum(1 for _ in open(rc, encoding='utf-8', errors='ignore')) - 1
    except Exception: return 0

def main(a):
    from ultralytics import YOLO
    run_dir = os.path.join(OUT, a.name)
    best = os.path.join(run_dir, 'weights', 'best.pt')
    done = epochs_done(run_dir)
    if done < a.epochs:
        if os.path.isdir(run_dir):
            print('incomplete (%d/%d epochs) -> clean restart' % (done, a.epochs), flush=True)
            shutil.rmtree(run_dir, ignore_errors=True)
        print('TRAIN start %s' % a.name, flush=True)
        m = YOLO(a.model)
        kw = dict(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, workers=0, device=0,
                  close_mosaic=10, optimizer='AdamW', lr0=0.000714, momentum=0.9, warmup_bias_lr=0.0,
                  seed=a.seed, exist_ok=True, project=OUT, name=a.name, cls_remap=True,
                  mixup=a.mixup, cutmix=0.0, deterministic=True)
        if a.model.endswith('.yaml'):
            kw['pretrained'] = False
        m.train(**kw)
        print('TRAIN done %s' % a.name, flush=True)
    else:
        print('TRAIN skip (complete %d epochs) %s' % (done, a.name), flush=True)
    if os.path.exists(best):
        print('VAL official %s' % a.name, flush=True)
        mm = YOLO(best)
        r = mm.val(data=a.official, imgsz=640, batch=16, device=0)
        line = '%s\tmAP50=%.4f\tmAP50-95=%.4f' % (a.name, r.box.map50, r.box.map)
        print(line, flush=True)
        for f in (SUM1, SUM2):
            with open(f, 'a', encoding='utf-8') as fh: fh.write(line + '\n')
        return 0
    print('ERROR no best.pt for %s' % a.name, flush=True)
    return 1

if __name__ == '__main__':
    freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True); ap.add_argument('--epochs', type=int, required=True)
    ap.add_argument('--mixup', type=float, default=0.0); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--official', required=True); ap.add_argument('--seed', type=int, default=42)
    sys.exit(main(ap.parse_args()))
