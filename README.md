# FGVD Vehicle Classification — YOLOv8-nano + SGCN

Graph-based two/three-wheeler classification on Indian roads,  
implementing the pipeline from *"Graph-based Two-Three Wheeler Classification in Unconstrained Indian Roads"* (ITSC 2024)  
with **YOLOv8-nano** replacing the original YOLOv7-tiny detector.

---

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 1 — Detection                │
│  YOLOv8-nano (CSP-DarkNet backbone) │
│  → Bounding boxes + class logits   │
│  → Crops resized to 64 × 64        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Feature Extraction                 │
│  • RGB      (3 ch)                  │
│  • Gabor    (4 ch, λ=6, 4 angles)  │
│  • Sobel    (1 ch, gradient mag.)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Graph Construction                 │
│  2D grid graph — 8-connectivity    │
│  pixel → node, distance → weight   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Stage 2 — Classification           │
│  SGCN (3 layers, 64 filters, 3×3)  │
│  → H' = D̂^{-½} Â D̂^{-½} HW + b  │
│  → Global mean pool → Linear       │
└─────────────────────────────────────┘
    │
    ▼
Predicted class (car / motorcycle / scooter /
                 truck / autorickshaw / bus)
```

---

## Why YOLOv8-nano over YOLOv7-tiny?

| Property | YOLOv7-tiny | YOLOv8-nano |
|---|---|---|
| Parameters | ~6 M | ~3.2 M |
| mAP50 (COCO) | 38.7 | 37.3 |
| Inference speed | ~160 FPS | ~200 FPS |
| API / library | custom repo | `pip install ultralytics` |
| Maintainability | deprecated | actively maintained |

YOLOv8-nano is lighter, faster, better-supported, and integrates with one import.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Prepare data

Place FGVD images under `data/raw/images/{train,val,test}/`  
and YOLO-format labels under `data/raw/labels/{train,val,test}/`.

### 2. Train Stage 1 — YOLOv8-nano

```bash
python src/detection/train_yolo.py
# Resume from checkpoint:
python src/detection/train_yolo.py --resume runs/detect/fgvd_yolov8n/weights/last.pt
```

### 3. Extract crops

```bash
python src/detection/infer_yolo.py \
    --weights runs/detect/fgvd_yolov8n/weights/best.pt \
    --source  data/raw/images/train \
    --out     data/processed/crops/train

# repeat for val/ and test/
```

### 4. Train Stage 2 — SGCN

```bash
python src/training/train_gnn.py                     # SGCN (default)
python src/training/train_gnn.py --model gat         # GAT alternative
```

Feature combinations can be set in `configs/gnn_config.yaml`:
```yaml
features: [rgb, gabor, sobel]   # best for L-1 per paper Table II
# features: [rgb, gabor]        # best for L-2/L-3 per paper Tables III/IV
```

### 5. Evaluate full pipeline

```bash
python src/training/evaluate.py \
    --yolo_weights runs/detect/fgvd_yolov8n/weights/best.pt \
    --gnn_weights  runs/gnn/fgvd_sgcn/best.pt \
    --test_images  data/raw/images/test
```

### 6. Inference on new images

```bash
python run_pipeline.py \
    --yolo_weights runs/detect/fgvd_yolov8n/weights/best.pt \
    --gnn_weights  runs/gnn/fgvd_sgcn/best.pt \
    --source       path/to/image.jpg \
    --save_vis
```

### 7. Export YOLOv8-nano to ONNX / TensorRT

```bash
python src/detection/export_model.py \
    --weights runs/detect/fgvd_yolov8n/weights/best.pt \
    --format onnx --simplify

# TensorRT FP16:
python src/detection/export_model.py \
    --weights runs/detect/fgvd_yolov8n/weights/best.pt \
    --format engine --half
```

---

## Expected results (from paper, RGB+Gabor+Sobel features)

| Case | L-1 Accuracy |
|---|---|
| Two-wheelers vs All | 95.05% |
| Three-wheelers vs All | 93.24% |
| Two+Three vs All | 88.15% |
| All classes | 86.37% |

---

## Reference

Nayak et al., *"Graph-based Two-Three Wheeler Classification in Unconstrained Indian Roads"*,  
IEEE ITSC 2024. DOI: 10.1109/ITSC58415.2024.10919487
