# ═══════════════════════════════════════════════════════════════════
#  CAPSTONE MODULE 4 — Notebook 1: Data Pipeline
#  Vehicle Detection Dataset (car, van, bus)
#  Jalankan di: Google Colab / Lokal (Apple M4 Pro)
# ═══════════════════════════════════════════════════════════════════

# ─── CELL 1: Install Dependencies ───────────────────────────────────
# !pip install ultralytics albumentations roboflow -q

# ─── CELL 2: Import Libraries ───────────────────────────────────────
import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import cv2
from PIL import Image

# Ultralytics & Albumentations
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

print("✅ Libraries loaded")

# ─── CELL 3: Konfigurasi Path ───────────────────────────────────────
# === SESUAIKAN PATH INI DENGAN LOKASI DATASET KAMU ===
# Jika dari Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')
# DATASET_ROOT = Path("/content/drive/MyDrive/vehicle_detection_dataset")

# Jika lokal (Apple M4):
DATASET_ROOT = Path("./data/vehicle_detection")  # ganti sesuai lokasi dataset

TRAIN_IMG = DATASET_ROOT / "train" / "images"
TRAIN_LBL = DATASET_ROOT / "train" / "labels"
VAL_IMG   = DATASET_ROOT / "valid" / "images"
VAL_LBL   = DATASET_ROOT / "valid" / "labels"
TEST_IMG  = DATASET_ROOT / "test"  / "images"
TEST_LBL  = DATASET_ROOT / "test"  / "labels"

# Kelas sesuai dataset
CLASS_NAMES = ["bus", "car", "van"]
CLASS_COLORS = {"bus": "#FF6B6B", "car": "#4ECDC4", "van": "#FFE66D"}

print(f"Dataset root: {DATASET_ROOT}")
print(f"Train images : {len(list(TRAIN_IMG.glob('*.jpg')))} gambar")
print(f"Val images   : {len(list(VAL_IMG.glob('*.jpg')))} gambar")
print(f"Test images  : {len(list(TEST_IMG.glob('*.jpg')))} gambar")

# ─── CELL 4: Eksplorasi Dataset (EDA) ───────────────────────────────
def parse_yolo_labels(label_dir: Path, class_names: list) -> pd.DataFrame:
    """
    Baca semua file .txt label YOLO dan return sebagai DataFrame.
    Format YOLO: class_id cx cy w h (semua normalized 0-1)
    """
    records = []
    for lbl_file in label_dir.glob("*.txt"):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, cx, cy, w, h = parts
                    records.append({
                        "file"    : lbl_file.stem,
                        "class_id": int(cls_id),
                        "class"   : class_names[int(cls_id)],
                        "cx"      : float(cx),
                        "cy"      : float(cy),
                        "width"   : float(w),
                        "height"  : float(h),
                        "area"    : float(w) * float(h),
                    })
    return pd.DataFrame(records)

df_train = parse_yolo_labels(TRAIN_LBL, CLASS_NAMES)
df_val   = parse_yolo_labels(VAL_LBL, CLASS_NAMES)
df_test  = parse_yolo_labels(TEST_LBL, CLASS_NAMES)

print("\n📊 EDA — Training Set")
print(df_train["class"].value_counts())
print(f"\nTotal anotasi (train): {len(df_train)}")
print(f"Total anotasi (val)  : {len(df_val)}")
print(f"Total anotasi (test) : {len(df_test)}")

# ─── CELL 5: Visualisasi Distribusi Kelas ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
fig.suptitle("Distribusi Kelas per Split", fontsize=14, fontweight="bold")

for ax, (df, title) in zip(
    axes,
    [(df_train, "Train"), (df_val, "Validation"), (df_test, "Test")]
):
    counts = df["class"].value_counts()
    colors = [CLASS_COLORS.get(c, "#888") for c in counts.index]
    bars   = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(f"{title} ({len(df)} anotasi)", fontweight="bold")
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah")
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("eda_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── CELL 6: Distribusi Ukuran BBox ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(df_train["width"], df_train["height"],
                c=[CLASS_COLORS.get(c, "#888") for c in df_train["class"]],
                alpha=0.4, s=15)
axes[0].set_xlabel("Width (normalized)")
axes[0].set_ylabel("Height (normalized)")
axes[0].set_title("Distribusi Ukuran Bounding Box")
axes[0].spines[["top", "right"]].set_visible(False)

# Legend manual
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=v, label=k) for k, v in CLASS_COLORS.items()]
axes[0].legend(handles=legend_elements)

axes[1].hist(df_train["area"], bins=40, color="#7B61FF", edgecolor="white", linewidth=0.3)
axes[1].set_xlabel("Area (w×h, normalized)")
axes[1].set_ylabel("Frekuensi")
axes[1].set_title("Distribusi Area BBox")
axes[1].spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("eda_bbox_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── CELL 7: Visualisasi Sample Gambar + BBox ───────────────────────
def visualize_sample(img_dir: Path, lbl_dir: Path, class_names: list,
                     class_colors: dict, n_samples: int = 6):
    """Tampilkan n_samples gambar dari dataset dengan bounding boxes."""
    img_files = list(img_dir.glob("*.jpg"))[:n_samples]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, img_path in zip(axes, img_files):
        img  = cv2.imread(str(img_path))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ax.imshow(img)

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts  = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = [float(p) for p in parts[1:5]]

                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    bw_px = int(bw * w)
                    bh_px = int(bh * h)

                    cls_name = class_names[cls_id]
                    color    = class_colors.get(cls_name, "#FFFFFF")

                    rect = patches.Rectangle(
                        (x1, y1), bw_px, bh_px,
                        linewidth=2, edgecolor=color, facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, cls_name.upper(),
                            color=color, fontsize=8, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

        ax.set_title(img_path.name, fontsize=8)
        ax.axis("off")

    plt.suptitle("Sample Gambar Training dengan Anotasi", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()

visualize_sample(TRAIN_IMG, TRAIN_LBL, CLASS_NAMES, CLASS_COLORS)

# ─── CELL 8: Data.yaml ─────────────────────────────────────────────
# Buat/verifikasi data.yaml untuk YOLOv8
import yaml

data_yaml = {
    "path"  : str(DATASET_ROOT.resolve()),
    "train" : "train/images",
    "val"   : "valid/images",
    "test"  : "test/images",
    "nc"    : len(CLASS_NAMES),
    "names" : CLASS_NAMES,
}

yaml_path = DATASET_ROOT / "data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print("✅ data.yaml dibuat:")
print(yaml.dump(data_yaml))

# ─── CELL 9: Augmentasi Pipeline (Non-Geometrik) ────────────────────
# Augmentasi yang digunakan: HANYA non-geometrik
# (brightness, contrast, saturation, noise, blur)
# TIDAK menggunakan: flip, rotate, crop, affine transform

augmentation_pipeline = A.Compose(
    [
        # Perubahan pencahayaan & warna
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.4,
        ),
        # Noise
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        # Blur ringan
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        # JPEG compression artifact simulation
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
    ),
)

print("✅ Augmentation pipeline siap (non-geometrik only)")
print("Pipeline steps:")
for t in augmentation_pipeline.transforms:
    print(f"  - {t.__class__.__name__} (p={t.p})")

# ─── CELL 10: Test Augmentasi Pada 1 Gambar ────────────────────────
def demo_augmentation(img_path: Path, lbl_path: Path, class_names: list,
                      pipeline: A.Compose, n_aug: int = 5):
    """Visualisasi hasil augmentasi pada satu gambar."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Baca label
    bboxes, labels = [], []
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            labels.append(int(parts[0]))
            bboxes.append([float(x) for x in parts[1:5]])

    fig, axes = plt.subplots(1, n_aug + 1, figsize=(18, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    for i in range(n_aug):
        aug = pipeline(image=img, bboxes=bboxes, class_labels=labels)
        axes[i + 1].imshow(aug["image"])
        axes[i + 1].set_title(f"Aug #{i+1}")
        axes[i + 1].axis("off")

    plt.suptitle("Demo Augmentasi Non-Geometrik", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("augmentation_demo.png", dpi=150, bbox_inches="tight")
    plt.show()

# Ambil gambar pertama dari train untuk demo
sample_imgs = list(TRAIN_IMG.glob("*.jpg"))
if sample_imgs:
    si = sample_imgs[0]
    sl = TRAIN_LBL / (si.stem + ".txt")
    if sl.exists():
        demo_augmentation(si, sl, CLASS_NAMES, augmentation_pipeline)

print("✅ Data Pipeline selesai! Lanjut ke notebook 02_model_training.py")
