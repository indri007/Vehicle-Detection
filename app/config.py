# ─────────────────────────────────────────────────────────
#  config.py — Konfigurasi terpusat aplikasi
#  Model: YOLO26 (Ultralytics SOTA, Jan 2026)
#  NMS-free, end-to-end inference
# ─────────────────────────────────────────────────────────
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "best.pt"

# ── Model ──────────────────────────────────────────────────
CLASS_NAMES   = ["bus", "car", "van"]
CLASS_COLORS  = {
    "bus": "#FF6B6B",   # merah
    "car": "#4ECDC4",   # teal
    "van": "#FFE66D",   # kuning
}
DEFAULT_CONF  = 0.40   # confidence threshold default
DEFAULT_IOU   = 0.45   # NMS IOU threshold (YOLO26 NMS-free, tp sbg fallback)
IMG_SIZE      = 640    # ukuran input model

# ── n8n Webhook ────────────────────────────────────────────
N8N_WEBHOOK_URL = os.getenv(
    "N8N_WEBHOOK_URL",
    "http://localhost:5678/webhook/vehicle-detection"   # ganti sesuai URL n8n kamu
)
N8N_TIMEOUT_SEC = 5

# ── Traffic Alert Threshold ────────────────────────────────
ALERT_THRESHOLD = 10   # kirim notif jika total kendaraan > nilai ini

# ── App Meta ───────────────────────────────────────────────
APP_NAME    = "VehicleScope AI"
APP_VERSION = "1.0.0"
APP_DESC    = "Deteksi & Hitung Kendaraan Otomatis dengan YOLO26"
