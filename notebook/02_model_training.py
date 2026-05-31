# ═══════════════════════════════════════════════════════════════════
#  CAPSTONE MODULE 4 — Notebook 2: Model Training YOLO26
#  Vehicle Detection (car, van, bus)
#  YOLO26: Newest SOTA Ultralytics (Jan 2026) — NMS-free, End-to-End
#  Platform: https://platform.ultralytics.com/ultralytics/yolo26
#  Device: Apple M4 Pro (MPS Backend)
# ═══════════════════════════════════════════════════════════════════

# ─── CELL 1: Install Dependencies ───────────────────────────────────
# Pastikan ultralytics versi terbaru agar mendukung YOLO26
# !pip install -U ultralytics -q

# ─── CELL 2: Import & Verifikasi YOLO26 Support ─────────────────────
import os
import shutil
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from ultralytics import YOLO
import ultralytics

print(f"Ultralytics version : {ultralytics.__version__}")
print(f"PyTorch version     : {torch.__version__}")
print(f"CUDA available      : {torch.cuda.is_available()}")
print(f"MPS available       : {torch.backends.mps.is_available()}")

# Verifikasi YOLO26 model tersedia
try:
    _test = YOLO("yolo26n.pt")
    print("✅ YOLO26 model supported!")
    del _test
except Exception as e:
    print(f"⚠️  YOLO26 check: {e}")
    print("   Pastikan ultralytics versi terbaru: pip install -U ultralytics")

# ─── CELL 3: Pilih Device ───────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("\n✅ Device: Apple MPS (Metal Performance Shaders) — M4 Pro")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"\n✅ Device: CUDA — {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("\n⚠️  Device: CPU (training akan lebih lambat)")

# ─── CELL 4: Konfigurasi Path & Hyperparameter ──────────────────────
DATASET_ROOT = Path("./data/vehicle_detection")  # sesuaikan dengan lokasi dataset
DATA_YAML    = DATASET_ROOT / "data.yaml"
OUTPUT_DIR   = Path("./runs/train")

# ──────────────────────────────────────────────────────────────────
#  YOLO26 Model Variants:
#  yolo26n.pt  → Nano   (~3M params)   — paling ringan, cocok edge device
#  yolo26s.pt  → Small  (~12M params)  — balance speed/accuracy
#  yolo26m.pt  → Medium (~26M params)  ← REKOMENDASI untuk vehicle detection
#  yolo26l.pt  → Large  (~44M params)  — paling akurat, butuh RAM lebih
#
#  YOLO26 Keunggulan vs YOLOv8:
#  ✅ NMS-free (native end-to-end inference — lebih cepat di deployment)
#  ✅ MuSGD optimizer (lebih stabil dan konvergen lebih cepat)
#  ✅ Tanpa Distribution Focal Loss (lebih efisien)
#  ✅ Dirilis Januari 2026 — SOTA terbaru Ultralytics
# ──────────────────────────────────────────────────────────────────

CONFIG = {
    # Model YOLO26 — medium variant
    "model"     : "yolo26m.pt",

    # Dataset
    "data"      : str(DATA_YAML),

    # Training
    "epochs"    : 50,
    "imgsz"     : 640,
    "batch"     : 4,
    "cache"     : True,
    "device"    : DEVICE,

    # Optimizer — YOLO26 pakai MuSGD secara default, bisa override
    "optimizer" : "auto",   # biarkan auto agar pakai MuSGD bawaan YOLO26
    "lr0"       : 0.001,
    "lrf"       : 0.01,
    "momentum"  : 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,

    # Early stopping
    "patience"  : 15,

    # Output
    "project"   : str(OUTPUT_DIR),
    "name"      : "vehicle_yolo26_v1",
    "exist_ok"  : True,
    "save"      : True,
    "plots"     : True,
    "verbose"   : True,

    # ── Augmentasi: HANYA Non-Geometrik ────────────────────────────
    # Matikan semua augmentasi geometrik (flip, rotate, scale, dll)
    "degrees"   : 0.0,    # ❌ no rotation
    "translate" : 0.0,    # ❌ no translation
    "scale"     : 0.0,    # ❌ no scale jitter
    "shear"     : 0.0,    # ❌ no shear
    "perspective": 0.0,   # ❌ no perspective
    "flipud"    : 0.0,    # ❌ no vertical flip
    "fliplr"    : 0.0,    # ❌ no horizontal flip
    "mosaic"    : 0.0,    # ❌ no mosaic
    "mixup"     : 0.0,    # ❌ no mixup
    "copy_paste": 0.0,    # ❌ no copy-paste

    # ✅ Non-geometrik augmentation (perubahan warna & cahaya)
    "hsv_h"     : 0.015,  # hue jitter ±1.5%
    "hsv_s"     : 0.30,   # saturation ±30%
    "hsv_v"     : 0.30,   # brightness ±30%
    "erasing"   : 0.0,
}

print("\n📋 YOLO26 Training Config:")
print(f"  Model    : {CONFIG['model']} (Medium variant)")
print(f"  Device   : {CONFIG['device']}")
print(f"  Epochs   : {CONFIG['epochs']}")
print(f"  Batch    : {CONFIG['batch']}")
print(f"  Optimizer: {CONFIG['optimizer']} (MuSGD default)")
print(f"  Aug mode : Non-geometrik only (no flip/rotate/mosaic)")

# ─── CELL 5: Training YOLO26 ────────────────────────────────────────
print(f"\n🚀 Memulai training YOLO26 pada device: {DEVICE}")
print("=" * 60)

# Load pretrained YOLO26 — weights otomatis diunduh dari Ultralytics
model = YOLO(CONFIG["model"])

print(f"\n✅ Model YOLO26 loaded: {CONFIG['model']}")
print(f"   Total parameters: {sum(p.numel() for p in model.model.parameters()):,}")

# Jalankan training
train_args = {k: v for k, v in CONFIG.items() if k != "model"}
results = model.train(**train_args)

print("\n✅ Training YOLO26 selesai!")
results_dir = Path(OUTPUT_DIR) / CONFIG["name"]
print(f"   Weights: {results_dir}/weights/")

# ─── CELL 6: Tampilkan Hasil Training ───────────────────────────────
for plot_name in [
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "PR_curve.png",
    "F1_curve.png",
]:
    plot_path = results_dir / plot_name
    if plot_path.exists():
        fig, ax = plt.subplots(figsize=(12, 7))
        img = mpimg.imread(str(plot_path))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"YOLO26 — {plot_name.replace('.png','').replace('_',' ').title()}",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

# ─── CELL 7: Evaluasi pada Validation Set ───────────────────────────
best_pt = results_dir / "weights" / "best.pt"
print(f"\n📦 Loading best YOLO26 model: {best_pt}")
best_model = YOLO(str(best_pt))

val_metrics = best_model.val(
    data=str(DATA_YAML),
    imgsz=640,
    device=DEVICE,
    verbose=True,
)

print("\n" + "="*50)
print("📊 YOLO26 Validation Metrics")
print("="*50)
print(f"  mAP@50       : {val_metrics.box.map50:.4f}")
print(f"  mAP@50-95    : {val_metrics.box.map:.4f}")
print(f"  Precision    : {val_metrics.box.mp:.4f}")
print(f"  Recall       : {val_metrics.box.mr:.4f}")

CLASS_NAMES = ["bus", "car", "van"]
print(f"\n{'Kelas':<10} {'Precision':>10} {'Recall':>10} {'mAP50':>10} {'mAP50-95':>10}")
print("-" * 54)
for i, cls_name in enumerate(CLASS_NAMES):
    try:
        p  = val_metrics.box.p[i]
        r  = val_metrics.box.r[i]
        m  = val_metrics.box.ap50[i]
        m2 = val_metrics.box.ap[i]
        print(f"{cls_name:<10} {p:>10.4f} {r:>10.4f} {m:>10.4f} {m2:>10.4f}")
    except Exception:
        pass

# ─── CELL 8: Inferensi Sample Test Images ───────────────────────────
test_img_dir = DATASET_ROOT / "test" / "images"
sample_tests = list(test_img_dir.glob("*.jpg"))[:6]

if sample_tests:
    import cv2
    test_results = best_model.predict(
        source=sample_tests,
        conf=0.40,
        imgsz=640,
        device=DEVICE,
        save=True,
        project=str(results_dir / "test_predictions"),
        name="samples",
        verbose=False,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, (result, img_path) in enumerate(zip(test_results, sample_tests)):
        annotated     = result.plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        axes[i].imshow(annotated_rgb)
        axes[i].set_title(img_path.name, fontsize=8)
        axes[i].axis("off")

    plt.suptitle(
        "YOLO26 — Prediksi pada Test Set (Vehicle Detection)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("yolo26_test_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()

# ─── CELL 9: Export Model ke Streamlit App ──────────────────────────
app_model_dir = Path(__file__).resolve().parent.parent / "app" / "models"
app_model_dir.mkdir(parents=True, exist_ok=True)

dest = app_model_dir / "best.pt"
shutil.copy2(str(best_pt), str(dest))
print(f"\n✅ YOLO26 model disalin ke Streamlit app: {dest}")

print("\n🎉 Pipeline YOLO26 selesai!")
print("   Selanjutnya: streamlit run app/main.py")
print("   Atau deploy ke Streamlit Community Cloud")
