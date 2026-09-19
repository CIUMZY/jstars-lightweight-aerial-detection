# -*- coding: utf-8 -*-
"""从中断的 last.pt 续跑一个 run，并在结束后做官方 val 评测。"""
import argparse, os, sys, time
from multiprocessing import freeze_support
ROOT = r'D:\Research\03_Codex\projects\yolo-paper'
OUT = os.path.join(ROOT, 'runs_p0')
SUM1 = os.path.join(OUT, 'summary_seed_extra.txt')
SUM2 = os.path.join(OUT, 'summary_remaining_extra.txt')

def main(a):
    from ultralytics import YOLO
    run_dir = os.path.join(OUT, a.name)
    last = os.path.join(run_dir, 'weights', 'last.pt')
    best = os.path.join(run_dir, 'weights', 'best.pt')
    print('resume from %s' % last, flush=True)
    m = YOLO(last)
    m.train(resume=True)
    print('resume training finished', flush=True)
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
    ap.add_argument('--name', required=True)
    ap.add_argument('--official', required=True)
    sys.exit(main(ap.parse_args()))
