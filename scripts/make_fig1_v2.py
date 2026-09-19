# -*- coding: utf-8 -*-
"""Fig 1：研究协议流程图（干净版）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style_jstars import apply, save, C
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

apply()
OUT = r"D:\Research\02_Vault\30_Projects\轻量小样本目标检测\figures_jstars"
fig, ax = plt.subplots(figsize=(7.0, 2.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)

def box(x, y, w, h, text, fc, ec, fs=7.0, tc="#1A1A1A"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                facecolor=fc, edgecolor=ec, linewidth=0.8))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.35)

box(0.15, 1.75, 2.1, 0.95, "Datasets\nVisDrone · AI-TOD · NEU-DET", "#EAF2FB", C["blue"])
box(2.75, 1.75, 2.1, 0.95, "Label budgets\n647 · 1 617 · 3 235 imgs\n(VisDrone 10–100%)", "#FDF1E3", C["yellow"])
box(5.35, 1.75, 2.1, 0.95, "Seven training choices\npretrain · mixup · CutMix · KD\nfreeze · epochs · TTA", "#E8F5F1", C["green"])
box(7.95, 1.75, 1.9, 0.95, "Five-seed runs\npaired statistical tests", "#F5EFF7", C["purple"])
box(1.45, 0.35, 2.1, 0.85, "Mechanism\nCKA · size-stratified error", "#EAF2FB", C["blue"])
box(4.05, 0.35, 2.1, 0.85, "Deployment\nFP32 · FP16 · INT8", "#FDF1E3", C["yellow"])
box(6.65, 0.35, 3.2, 0.85, "Released artifacts\nprotocol · per-seed results · split lists", "#E8F5F1", C["green"])

for x0, x1 in [(2.28, 2.72), (4.88, 5.32), (7.48, 7.92)]:
    ax.add_patch(FancyArrowPatch((x0, 2.22), (x1, 2.22), arrowstyle="-|>", mutation_scale=8, color="#8C8C8C", lw=0.9))
for x0, x1 in [(1.2, 2.5), (3.9, 5.1), (6.6, 7.6)]:
    ax.add_patch(FancyArrowPatch((x0, 1.72), (x1, 1.22), arrowstyle="-|>", mutation_scale=8, color="#B5B5B5", lw=0.8))
ax.text(0.15, 2.88, "(a) Data and evaluation protocol", fontsize=7.5, color="#555555")
ax.text(0.15, 1.32, "(b) Diagnostics and deployment", fontsize=7.5, color="#555555")
save(fig, OUT, "fig1_protocol")
