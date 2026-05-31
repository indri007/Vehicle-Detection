# ─────────────────────────────────────────────────────────
#  utils/model.py — YOLO26 inference helper
#  YOLO26: NMS-free, end-to-end, SOTA Jan 2026
# ─────────────────────────────────────────────────────────
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Tambah parent ke path agar config bisa di-import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLASS_COLORS, CLASS_NAMES, IMG_SIZE, MODEL_PATH


@st.cache_resource(show_spinner="⚩️ Memuat model YOLO26...")
def load_model():
    """
    Load YOLO26 model dari file best.pt.
    YOLO26 adalah model SOTA Ultralytics (Jan 2026):
    - NMS-free (native end-to-end inference)
    - Lebih kecil, cepat, dan akurat dari YOLOv8
    - Menggunakan MuSGD optimizer
    Di-cache agar hanya di-load sekali per sesi Streamlit.
    """
    from ultralytics import YOLO

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di: {MODEL_PATH}\n"
            "Jalankan notebook training terlebih dahulu, lalu salin best.pt ke app/models/\n"
            "Untuk training YOLO26: python notebook/02_model_training.py"
        )
    model = YOLO(str(MODEL_PATH))
    return model


def run_inference(
    image: Image.Image,
    conf: float = 0.40,
    iou: float = 0.45,
) -> tuple[Image.Image, dict[str, int], list[dict]]:
    """
    Jalankan YOLOv8 inference pada gambar PIL.

    Returns:
        annotated_img : gambar dengan bounding box
        counts        : {"car": 3, "van": 1, "bus": 2}
        detections    : list detail tiap deteksi
    """
    model = load_model()

    # Inferensi
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        imgsz=IMG_SIZE,
        verbose=False,
    )

    result      = results[0]
    boxes       = result.boxes
    class_names = model.names  # {0: 'bus', 1: 'car', 2: 'van'}

    # Hitung per kelas
    counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
    detections: list[dict] = []

    for box in boxes:
        cls_id     = int(box.cls[0])
        cls_name   = class_names[cls_id]
        confidence = float(box.conf[0])
        xyxy       = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

        if cls_name in counts:
            counts[cls_name] += 1

        detections.append({
            "class"     : cls_name,
            "confidence": round(confidence, 3),
            "bbox"      : xyxy,
        })

    # Gambar anotasi
    annotated_img = _draw_boxes(image.copy(), detections)

    return annotated_img, counts, detections


def _draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Gambar bounding boxes dengan label & confidence di atas gambar PIL."""
    draw = ImageDraw.Draw(image)

    # Coba load font, fallback ke default jika tidak ada
    try:
        font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=13)
    except Exception:
        font_label = ImageFont.load_default()
        font_small = font_label

    for det in detections:
        cls_name   = det["class"]
        confidence = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]

        # Warna kelas
        color_hex = CLASS_COLORS.get(cls_name, "#FFFFFF")
        color_rgb = _hex_to_rgb(color_hex)

        # Bounding box (tebal 3px)
        draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

        # Label background
        label_text = f"{cls_name.upper()}  {confidence:.0%}"
        bbox_text  = draw.textbbox((x1, y1 - 22), label_text, font=font_label)
        draw.rectangle(
            [bbox_text[0] - 4, bbox_text[1] - 2, bbox_text[2] + 4, bbox_text[3] + 2],
            fill=color_rgb,
        )
        draw.text((x1, y1 - 22), label_text, fill=(10, 10, 10), font=font_label)

    return image


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
