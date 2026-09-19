# Training Lightweight Aerial Object Detectors under Data Scarcity

Reproducibility package for the manuscript:

> **Training Lightweight Aerial Object Detectors under Data Scarcity: A Bounded Empirical Study of Pretraining, Regularisation, Freezing and Consumer-Hardware Deployment**

## What is included

| Path | Contents |
|---|---|
| `data/splits/` | Image-ID lists for every training subset and holdout used in the paper (VisDrone 10/25/50/100 %, AI-TOD 647/1617/3235, NEU-DET) |
| `data/yaml/` | Ultralytics dataset YAML files that bind the splits |
| `configs/run_configs.csv` | Per-run configuration of all 112 training runs (model, data, epochs, seed, mixup, cutmix, freeze, batch, optimiser, learning rate, weight decay) |
| `results/` | Official-validation results per run, 5-seed aggregate tables, per-seed matrix, paired statistics (95 % CI, Hedges' g), CKA, size-stratified accuracy, deployment and INT8 measurements |
| `scripts/` | Training runners (YOLOv8n/v5n/v8s), staged-unfreezing trainer, KUN/CKA analysis, size-stratified evaluation, CKA, deployment benchmark, ONNX Runtime INT8 quantisation, statistics, figure generation |
| `scripts/make_figs_*.py` | Regenerate every manuscript figure from the result tables (binary figures are not stored here) |

## Key results (official validation set)

| Configuration | mAP@0.5 |
|---|---|
| VisDrone 25 %, from scratch, 50 epochs | 0.1011 ± 0.0018 (5 seeds) |
| VisDrone 25 %, COCO pretrained, 50 epochs | 0.2106 ± 0.0027 (5 seeds) |
| VisDrone 25 %, recommended (100 epochs + mixup 0.3) | 0.2276 ± 0.0036 (5 seeds) |
| VisDrone 25 %, from scratch, 200 epochs | 0.1740 ± 0.0021 (2 seeds) |
| AI-TOD 1617, from scratch / pretrained | 0.1440 / 0.2518 (3 seeds) |
| NEU-DET, from scratch / pretrained | 0.6942 ± 0.0058 / 0.7426 ± 0.0137 (5 seeds) |

## Environment

- Python 3.12, PyTorch 2.5.1+cu121, Ultralytics 8.4.132, ONNX Runtime 1.30, TensorRT 11.3 (engine build failed on the test machine)
- Single consumer laptop: NVIDIA RTX 3060 Laptop GPU (6 GB), AMD Ryzen 7 5800H, 16 GB RAM
- Training used a single GPU with `workers=0` and deterministic seeds

## How to reproduce

```bash
# 1. place the datasets so that the YAML paths resolve (VisDrone, AI-TOD, NEU-DET)
# 2. train one configuration
python scripts/run_one2.py --name repro_pretrained --model yolov8n.pt \
    --data data/yaml/visdrone25_iv.yaml --official data/yaml/visdrone25.yaml \
    --epochs 50 --seed 42 --mixup 0 --cutmix 0 --batch 16
# 3. re-run the statistics
python scripts/compute_stats2.py
```

The split ID lists are sufficient to reconstruct every training subset from the public datasets.

## Citation

If you use this package, please cite the manuscript above. The earlier conference version of this work is cited in the manuscript.
