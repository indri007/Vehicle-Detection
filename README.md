# VehicleScope AI — Capstone Project Module 4

> 🚗 **Deteksi & Hitung Kendaraan Otomatis** menggunakan **YOLO26** + n8n + Streamlit
> 
> YOLO26 adalah model SOTA terbaru dari Ultralytics (Jan 2026) — NMS-free, end-to-end inference.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 📋 Deskripsi

Aplikasi ini mendeteksi dan menghitung kendaraan (🚗 Car, 🚐 Van, 🚌 Bus) dari gambar menggunakan model YOLOv8 yang telah dilatih pada dataset Vehicle Detection. Setiap hasil deteksi di-log secara otomatis ke Google Sheets via **n8n webhook**, dan alert dikirimkan ke Telegram jika traffic melebihi threshold.

## 🏗️ Arsitektur

```
User → Streamlit App → YOLOv8 Model → Hasil Deteksi
                                ↓
                        n8n Webhook → Google Sheets log
                                    → Telegram Alert
```

## 🚀 Cara Menjalankan

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/USERNAME/capstone-module4.git
cd capstone-module4
pip install -r requirements.txt
```

### 2. Training Model

```bash
# Siapkan dataset di ./data/vehicle_detection/
# Jalankan pipeline & training:
python notebook/01_data_pipeline.py
python notebook/02_model_training.py
# Model best.pt otomatis disalin ke app/models/
```

### 3. Konfigurasi n8n

```bash
# Salin .env.example ke .env dan isi N8N_WEBHOOK_URL
cp .env.example .env
```

Import workflow di n8n:
- Buka n8n → **Import from file** → pilih `n8n/workflow_vehicle_detection.json`
- Isi `YOUR_GOOGLE_SHEET_ID` dan `YOUR_TELEGRAM_CHAT_ID`
- Aktifkan workflow

### 4. Jalankan Streamlit

```bash
streamlit run app/main.py
```

## 📁 Struktur Proyek

```
capstone-module4/
├── notebook/
│   ├── 01_data_pipeline.py     # EDA + augmentasi
│   └── 02_model_training.py    # Training YOLOv8 + evaluasi
├── app/
│   ├── main.py                 # Landing page
│   ├── config.py               # Konfigurasi terpusat
│   ├── utils/
│   │   ├── model.py            # YOLOv8 inference helper
│   │   └── n8n_client.py       # n8n webhook client
│   ├── pages/
│   │   ├── 1_Detection.py      # Upload & deteksi
│   │   ├── 2_Dashboard.py      # Analytics history
│   │   └── 3_About.py          # Info model & arsitektur
│   └── models/
│       └── best.pt             # Model YOLOv8 (hasil training)
├── n8n/
│   └── workflow_vehicle_detection.json
├── .streamlit/config.toml
├── requirements.txt
└── .env.example
```

## 🔬 Model

| Parameter | Nilai |
|-----------|-------|
| Architecture | **YOLO26m** (Ultralytics SOTA Jan 2026) |
| Dataset | Vehicle Detection (car, van, bus) |
| Input Size | 640 × 640 |
| Training Device | Apple M4 Pro (MPS) |
| Epochs | 50 |
| Optimizer | MuSGD (auto) |
| Inference | NMS-free (end-to-end) |
| Augmentasi | Non-geometrik only |

## 📡 n8n Webhook Payload

```json
{
  "timestamp": "2026-05-24T10:30:00+07:00",
  "image_name": "traffic.jpg",
  "detections": {"car": 4, "van": 2, "bus": 1},
  "total_vehicles": 7,
  "avg_confidence": 0.87,
  "alert": false
}
```

---

*Capstone Project Module 4 — Dibuat dengan YOLO26 + n8n + Streamlit*
