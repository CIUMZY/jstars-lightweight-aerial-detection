# -*- coding: utf-8 -*-
"""主结论配对统计 v2：含 TTA；df=1 时不使用 Hedges 校正；输出 LaTeX 表 + CSV。"""
import csv, json, math, os, re, statistics as st
OUT = r"D:\Research\03_Codex\projects\yolo-paper\runs_p0"
BASE = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测"

def betacf(a,b,x):
    MAXIT,EPS,FPMIN=300,3e-16,1e-300
    qab,qap,qam=a+b,a+1.0,a-1.0
    c,d=1.0,1.0-qab*x/qap
    d=FPMIN if abs(d)<FPMIN else d
    d=1.0/d; h=d
    for m in range(1,MAXIT+1):
        m2=2*m
        aa=m*(b-m)*x/((qam+m2)*(a+m2)); d=1.0+aa*d; d=FPMIN if abs(d)<FPMIN else d
        c=1.0+aa/c; c=FPMIN if abs(c)<FPMIN else c; d=1.0/d; h*=d*c
        aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2)); d=1.0+aa*d; d=FPMIN if abs(d)<FPMIN else d
        c=1.0+aa/c; c=FPMIN if abs(c)<FPMIN else c; d=1.0/d; de=d*c; h*=de
        if abs(de-1.0)<EPS: break
    return h
def betai(a,b,x):
    if x<=0: return 0.0
    if x>=1: return 1.0
    bt=math.exp(math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log(1-x))
    return bt*betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1.0-bt*betacf(b,a,1-x)/b
def t_p(t,df): return betai(df/2,0.5,df/(df+t*t))
def t_crit95(df):
    lo,hi=0.0,200.0
    for _ in range(200):
        mid=(lo+hi)/2
        if t_p(mid,df)>0.05: lo=mid
        else: hi=mid
    return (lo+hi)/2

def read():
    data={}
    pat=re.compile(r"^(\S+)\s+mAP50=([\d.]+)\s+mAP50-95=([\d.]+)\s*$")
    for fn in ["summary.txt","summary_seed_extra.txt","summary_remaining_extra.txt"]:
        f=os.path.join(OUT,fn)
        if not os.path.exists(f): continue
        for line in open(f,encoding="utf-8",errors="ignore"):
            m=pat.match(line.strip())
            if m:
                n=re.sub(r"_p0$","",m.group(1)); ms=re.search(r"_s(\d+)$",n)
                seed=int(ms.group(1)) if ms else 42
                data.setdefault(re.sub(r"_s\d+$","",n),{})[seed]=float(m.group(2))
    # TTA 文件格式： s42 \t flip \t mAP50=... \t mAP50-95=...
    f=os.path.join(OUT,"summary_tta5.txt")
    if os.path.exists(f):
        pat2=re.compile(r"^s(\d+)\s+(flip|scale|full)\s+mAP50=([\d.]+)")
        for line in open(f,encoding="utf-8",errors="ignore"):
            m=pat2.match(line.strip())
            if m: data.setdefault("tta_"+m.group(2),{})[int(m.group(1))]=float(m.group(3))
    return data

def paired(d,a,b,label):
    d1,d2=d.get(a,{}),d.get(b,{})
    common=sorted(set(d1)&set(d2))
    if len(common)<2: return None
    diff=[d2[s]-d1[s] for s in common]; n=len(diff)
    m=st.mean(diff); sd=st.stdev(diff); se=sd/math.sqrt(n)
    t=m/se if se else float("inf"); p=t_p(t,n-1); tc=t_crit95(n-1)
    g=(m/sd*(1-3/(4*(n-1)-1))) if (sd and n>2) else (m/sd if sd else float("inf"))
    return dict(label=label,n=n,delta=m,lo=m-tc*se,hi=m+tc*se,t=t,p=p,g=g)

def main():
    d=read()
    rows=[]
    for a,b,lab in [("scratch50","exp1_50","Pretraining vs from scratch (both 50 epochs)"),
                    ("scratch100_nomixup","decouple100_nomixup","Pretraining vs from scratch (both 100 epochs)"),
                    ("exp1_50","decouple100_nomixup","Training duration: 100 vs 50 epochs (no mixup)"),
                    ("exp1_50","freeze10","Freeze 10 modules vs unfrozen"),
                    ("exp1_50","freeze20","Freeze 20 modules vs unfrozen"),
                    ("neudet_scratch","neudet_proposed","NEU-DET: pretraining vs from scratch"),
                    ("aitod_b1617_scratch","aitod_b1617_pretrain","AI-TOD (1617): pretraining vs from scratch")]:
        r=paired(d,a,b,lab)
        if r: rows.append(r)
    for mode,lab in [("flip","TTA flip-only vs no TTA"),("scale","TTA multi-scale vs no TTA"),("full","TTA multi-scale+flip vs no TTA")]:
        r=paired(d,"proposed100","tta_"+mode,lab)
        if r: rows.append(r)
    print("%-52s %2s %9s  %-22s %8s %10s %7s" % ("comparison","n","delta","95% CI","t","p","g"))
    for r in rows:
        print("%-52s %2d %+9.4f  [%+.4f, %+.4f] %8.2f %10.3g %7.2f" %
              (r["label"], r["n"], r["delta"], r["lo"], r["hi"], r["t"], r["p"], r["g"]))
    json.dump(rows, open(os.path.join(BASE,"stats_paired.json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(os.path.join(BASE,"stats_paired.csv"),"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=["label","n","delta","lo","hi","t","p","g"]); w.writeheader(); w.writerows(rows)
    # LaTeX 表
    lines=[r"\begin{table}[!t]", r"\centering",
           r"\caption{Paired comparisons: mean difference, 95\% confidence interval, paired $t$ test and Hedges' $g$ (same seeds paired).}",
           r"\label{tab:stats}", r"\scriptsize", r"\begin{tabular}{lcccc}", r"\toprule",
           r"Comparison & $n$ & $\Delta$ (95\% CI) & $t$ & $g$ \\", r"\midrule"]
    for r in rows:
        lines.append(r"%s & %d & $%+.4f$ [$%+.4f$, $%+.4f$] & %.2f & %.2f \\" %
                     (r["label"].replace("%","\\%"), r["n"], r["delta"], r["lo"], r["hi"], r["t"], r["g"]))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(BASE,"stats_table.tex"),"w",encoding="utf-8").write("\n".join(lines))
    print("\nLaTeX 表已写：stats_table.tex")
if __name__=="__main__": main()
