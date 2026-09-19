# -*- coding: utf-8 -*-
"""通用单 run 训练器（支持 mixup / cutmix / freeze），跑完做官方 val。"""
import argparse, os, shutil, sys
from multiprocessing import freeze_support
ROOT = r'D:\Research\03_Codex\projects\yolo-paper'
OUT = os.path.join(ROOT, 'runs_p0')
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
    done = epochs_done(run_dir)
    if done < a.epochs:
        if os.path.isdir(run_dir):
            print('clean restart (%d/%d epochs)' % (done, a.epochs), flush=True)
            shutil.rmtree(run_dir, ignore_errors=True)
        print('TRAIN start %s ep=%d mixup=%s cutmix=%s freeze=%s seed=%s' % (a.name, a.epochs, a.mixup, a.cutmix, a.freeze, a.seed), flush=True)
        m = YOLO(a.model)
        kw = dict(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, workers=0, device=0,
                  close_mosaic=10, optimizer='AdamW', lr0=0.000714, momentum=0.9, warmup_bias_lr=0.0,
                  seed=a.seed, exist_ok=True, project=OUT, name=a.name, cls_remap=True,
                  mixup=a.mixup, cutmix=a.cutmix, deterministic=True)
        if a.model.endswith('.yaml'):
            kw['pretrained'] = False
        if a.freeze is not None:
            kw['freeze'] = a.freeze
        m.train(**kw)
        print('TRAIN done %s' % a.name, flush=True)
    best = os.path.join(run_dir, 'weights', 'best.pt')
    if os.path.exists(best):
        r = YOLO(best).val(data=a.official, imgsz=640, batch=16, device=0, verbose=False)
        line = '%s\tmAP50=%.4f\tmAP50-95=%.4f' % (a.name, r.box.map50, r.box.map)
        print(line, flush=True)
        for f in (SUM1, SUM2):
            open(f, 'a', encoding='utf-8').write(line + '\n')
        return 0
    return 1

if __name__ == '__main__':
    freeze_support()
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True); ap.add_argument('--official', required=True)
    ap.add_argument('--epochs', type=int, required=True); ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mixup', type=float, default=0.0); ap.add_argument('--cutmix', type=float, default=0.0)
    ap.add_argument('--freeze', type=int, default=None); ap.add_argument('--batch', type=int, default=16)
    sys.exit(main(ap.parse_args()))
