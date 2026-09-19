# -*- coding: utf-8 -*-
"""5-seed 官方val 汇总（v2）：读取全部 summary 文件，输出汇总表到 vault。"""
import re, os, csv, math, statistics as st
from collections import defaultdict
P   = r"D:\Research\03_Codex\projects\yolo-paper\runs_p0"
OUT = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测"
data = defaultdict(dict)
r50  = re.compile(r"^(\S+)\s+mAP50=([\d.]+)\s+mAP50-95=([\d.]+)\s*$")
rtta = re.compile(r"^(s\d+)\s+(flip|scale|full)\s+mAP50=([\d.]+)\s+mAP50-95=([\d.]+)\s*$")
for fn in ["summary.txt","summary_seed_extra.txt","summary_baseline_extra.txt","summary_remaining_extra.txt"]:
    p = os.path.join(P, fn)
    if not os.path.exists(p): continue
    for line in open(p, encoding="utf-8", errors="ignore"):
        m = r50.match(line.strip())
        if not m: continue
        n = re.sub(r"_p0$","",m.group(1)); ms = re.search(r"_s(\d+)$", n)
        seed = int(ms.group(1)) if ms else 42
        data[re.sub(r"_s\d+$","",n)][seed] = (float(m.group(2)), float(m.group(3)))
p = os.path.join(P,"summary_tta5.txt")
if os.path.exists(p):
    for line in open(p, encoding="utf-8", errors="ignore"):
        m = rtta.match(line.strip())
        if m: data["tta_"+m.group(2)][int(m.group(1)[1:])] = (float(m.group(3)), float(m.group(4)))
LAB = {"scratch50":"From scratch (50 ep)","exp1_50":"COCO pretraining (50 ep)","mixup01":"+ mixup 0.1 (50 ep)",
"mixup05":"+ mixup 0.5 (50 ep)","mixup_cutmix":"+ mixup 0.3 + CutMix 0.3 (50 ep)","distill":"+ knowledge distillation (50 ep)",
"freeze10":"+ freeze 10 modules (50 ep)","freeze20":"+ freeze 20 modules (50 ep)","proposed100":"Recommended (100 ep + mixup 0.3)",
"scratch100_nomixup":"From scratch (100 ep)","scratch100_mixup03":"From scratch (100 ep + mixup 0.3)",
"decouple50_mixup03":"50 ep + mixup 0.3 (decoupling)","decouple100_nomixup":"100 ep, no mixup (decoupling)",
"scale10":"Training data 10%","scale50":"Training data 50%","scale100":"Training data 100%",
"neudet_scratch":"NEU-DET, from scratch","neudet_proposed":"NEU-DET, pretrained",
"v5n_rec":"YOLOv5n (recommended cfg)","v8s_rec":"YOLOv8s (recommended cfg)",
"aitod_b647_scratch":"AI-TOD 647 imgs, from scratch","aitod_b647_pretrain":"AI-TOD 647 imgs, pretrained","aitod_b1617_scratch":"AI-TOD 1617 imgs, from scratch","aitod_b1617_pretrain":"AI-TOD 1617 imgs, pretrained","aitod_b3235_scratch":"AI-TOD 3235 imgs, from scratch","aitod_b3235_pretrain":"AI-TOD 3235 imgs, pretrained","tta_flip":"TTA flip-only","tta_scale":"TTA multi-scale (scale only)","tta_full":"TTA multi-scale + flip"}
rows=[]
for cfg,lab in LAB.items():
    if cfg not in data: continue
    seeds=sorted(data[cfg]); v50=[data[cfg][s][0] for s in seeds]; v95=[data[cfg][s][1] for s in seeds]
    m=st.mean(v50); sd=st.stdev(v50) if len(v50)>1 else 0.0
    m95=st.mean(v95); sd95=st.stdev(v95) if len(v95)>1 else 0.0
    rows.append(dict(config=cfg,label=lab,seeds=",".join(map(str,seeds)),n=len(v50),mean50=m,sd50=sd,min50=min(v50),max50=max(v50),mean95=m95,sd95=sd95))
with open(os.path.join(OUT,"结果_5seed汇总_2026-09-11.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
allseeds=sorted({s for c in data for s in data[c]})
with open(os.path.join(OUT,"结果_5seed_逐seed矩阵_2026-09-11.csv"),"w",newline="",encoding="utf-8-sig") as f:
    w=csv.writer(f); w.writerow(["config"]+[f"seed{s}" for s in allseeds])
    for cfg in LAB:
        if cfg in data: w.writerow([cfg]+[f"{data[cfg][s][0]:.4f}" if s in data[cfg] else "" for s in allseeds])
L=["# 5-seed 汇总（官方 val 口径）— 2026-09-11\n",
   "> 官方验证集；SD 为样本标准差。数据源：summary*.txt\n"]
full=[r for r in rows if r["n"]>=5]; part=[r for r in rows if r["n"]<5]
for title, group in (("## A. 5 个 seed（完整）", full), ("## B. 不足 5 个 seed", part)):
    L.append("\n"+title+"\n"); L.append("| 配置 | n | mAP@0.5 (mean ± SD) | range | mAP@0.5:0.95 |")
    L.append("|---|---|---|---|---|")
    for r in group:
        L.append(f"| {r['label']} | {r['n']} | {r['mean50']:.4f} ± {r['sd50']:.4f} | {r['min50']:.4f}–{r['max50']:.4f} | {r['mean95']:.4f} |")
open(os.path.join(OUT,"结果_5seed汇总_2026-09-11.md"),"w",encoding="utf-8").write("\n".join(L))
print("aggregate done:", len(rows), "configs")
