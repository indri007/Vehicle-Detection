# ─────────────────────────────────────────────────────────
#  pages/3_About.py — Info model & project
# ─────────────────────────────────────────────────────────
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APP_DESC, APP_NAME, APP_VERSION, CLASS_COLORS, CLASS_NAMES, MODEL_PATH

st.set_page_config(
    page_title=f"About · {APP_NAME}",
    page_icon="ℹ️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .info-card {
        background: linear-gradient(145deg, #111827, #1a2234);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
    }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #FF6B6B, #FFE66D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
    ">ℹ️ About VehicleScope AI</h1>
    """,
    unsafe_allow_html=True,
)

# ── Project Info ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Tentang Proyek")
    st.markdown(
        f"""
        **{APP_NAME}** adalah aplikasi deteksi kendaraan berbasis AI yang dibangun sebagai
        Capstone Project Module 4.

        Aplikasi ini menggabungkan:
        - 🤖 **YOLO26** (Ultralytics SOTA Jan 2026) — model object detection end-to-end terbaru
        - ⚙️ **n8n** — workflow automation untuk logging & alerting
        - 🎨 **Streamlit** — frontend web interaktif

        > **{APP_DESC}**

        ---
        **YOLO26 Keunggulan:**
        - 🚀 **NMS-free** — native end-to-end inference, lebih cepat saat deployment
        - 🔬 **MuSGD optimizer** — konvergensi lebih stabil
        - 📉 Tanpa Distribution Focal Loss — arsitektur lebih efisien
        - ✅ Lebih kecil, cepat & akurat dari YOLOv8

        ---
        **Version**: `{APP_VERSION}`  
        **Model**: `{MODEL_PATH.name if MODEL_PATH.exists() else "best.pt (belum ada)"}`  
        **Status Model**: {'✅ Model tersedia' if MODEL_PATH.exists() else '⚠️ Model belum ada — jalankan notebook/02_model_training.py terlebih dahulu'}
        """
    )

with col2:
    st.markdown("### 🏷️ Kelas yang Dideteksi")
    for cls in CLASS_NAMES:
        color = CLASS_COLORS[cls]
        icons = {"bus": "🚌", "car": "🚗", "van": "🚐"}
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.03);
                border-left: 4px solid {color};
                border-radius: 8px;
                padding: 0.8rem 1rem;
                margin-bottom: 0.6rem;
            ">
                <b style="color:{color};">{icons[cls]} {cls.upper()}</b>
                <span style="color:#64748B; font-size:0.85rem; margin-left:8px;">
                    Warna bounding box: <code>{color}</code>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Arsitektur ───────────────────────────────────────────────
st.markdown("### 🏗️ Arsitektur Sistem")
st.markdown(
    """
    ```
    ┌─────────────┐     Upload      ┌─────────────────────┐
    │    User     │ ─────────────►  │  Streamlit Frontend │
    │  (Browser)  │                 │   (Vehicle Scope)   │
    └─────────────┘                 └──────────┬──────────┘
                                               │ Inference
                                               ▼
                                   ┌─────────────────────┐
                                   │   YOLOv8 Model      │
                                   │   (best.pt / MPS)   │
                                   │                     │
                                   │ ► Detect: car,van,bus│
                                   │ ► Count per class   │
                                   │ ► Draw bounding box │
                                   └──────────┬──────────┘
                                              │ Webhook POST
                                              ▼
                                   ┌─────────────────────┐
                                   │    n8n Workflow      │
                                   │  (Self-hosted)       │
                                   │                     │
                                   │ ► Log → Google Sheets│
                                   │ ► Alert → Telegram  │
                                   │   (jika traffic > N)│
                                   └─────────────────────┘
    ```
    """
)

# ── Training Info ────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔬 Info Training")
ti1, ti2 = st.columns(2)

with ti1:
    st.markdown(
        """
        | Parameter | Nilai |
        |-----------|-------|
        | Base Model | `yolo26m.pt` |
        | Dataset | Vehicle Detection |
        | Input Size | 640 × 640 px |
        | Epochs | 50 |
        | Batch Size | 16 |
        | Optimizer | MuSGD (auto) |
        | Learning Rate | 0.001 |
        | Device | Apple MPS (M4 Pro) |
        | Inference | NMS-free (end-to-end) |
        """
    )

with ti2:
    st.markdown(
        """
        | Augmentasi | Metode |
        |-----------|--------|
        | Brightness | ±0.3 |
        | Contrast | ±0.3 |
        | Saturation | ±0.3 |
        | Gaussian Noise | σ=0.01 |
        | Blur | kernel 3×3 |
        | **Non-geometrik** | ✅ Tidak ada flip/rotate |
        """
    )

# ── n8n Payload Schema ───────────────────────────────────────
st.markdown("---")
st.markdown("### 📡 n8n Webhook Payload Schema")
st.code(
    """{
  "timestamp"      : "2026-05-24T10:30:00+07:00",
  "image_name"     : "traffic_001.jpg",
  "detections"     : {
    "car" : 4,
    "van" : 2,
    "bus" : 1
  },
  "total_vehicles" : 7,
  "avg_confidence" : 0.87,
  "alert"          : false,
  "alert_message"  : ""
}""",
    language="json",
)

st.markdown("---")
st.caption(
    f"🚗 {APP_NAME} v{APP_VERSION} · Capstone Project Module 4 · "
    "Dibuat dengan ❤️ menggunakan YOLO26 + n8n + Streamlit"
)
