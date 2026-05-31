# ─────────────────────────────────────────────────────────
#  main.py — Landing page VehicleScope AI
# ─────────────────────────────────────────────────────────
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from config import APP_DESC, APP_NAME, APP_VERSION, CLASS_COLORS, CLASS_NAMES

# ── Page Setup ─────────────────────────────────────────────
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hero gradient */
    .hero-container {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 60%);
        animation: pulse 6s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50%       { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00D4FF, #7B61FF, #FF6B6B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #94A3B8;
        font-weight: 300;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(0,212,255,0.15);
        border: 1px solid rgba(0,212,255,0.4);
        color: #00D4FF;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }

    /* Feature cards */
    .feature-card {
        background: linear-gradient(145deg, #111827, #1a2234);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, border-color 0.3s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0,212,255,0.4);
    }
    .feature-icon { font-size: 2.4rem; margin-bottom: 0.8rem; }
    .feature-title { font-size: 1rem; font-weight: 700; color: #E2E8F0; margin-bottom: 0.4rem; }
    .feature-desc  { font-size: 0.85rem; color: #64748B; line-height: 1.5; }

    /* Class badge */
    .class-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 4px;
    }
    .class-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }

    /* Metrics row */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #111827, #1a2234);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117, #111827);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero Section ────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero-container">
        <div class="hero-badge">⚡ YOLO26 · NMS-free · Real-time Detection</div>
        <div class="hero-title">🚗 {APP_NAME}</div>
        <div class="hero-sub">{APP_DESC}</div>
        <br>
        <div class="hero-sub" style="font-size:0.8rem; color:#475569;">
            v{APP_VERSION} &nbsp;·&nbsp; Capstone Project Module 4
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Feature Cards ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
features = [
    ("🎯", "Deteksi Otomatis", "Upload gambar, YOLOv8 mendeteksi bus, car, dan van secara instan."),
    ("📊", "Vehicle Counting", "Hitung jumlah tiap jenis kendaraan secara akurat dari satu gambar."),
    ("📡", "Live Logging via n8n", "Setiap deteksi di-log otomatis ke Google Sheets via n8n webhook."),
    ("🔔", "Traffic Alert", "Notifikasi otomatis jika kepadatan kendaraan melebihi threshold."),
]
for col, (icon, title, desc) in zip([c1, c2, c3, c4], features):
    with col:
        st.markdown(
            f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# helper untuk rgba
def _hex_parts(h):
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"

# ── Kelas yang Dideteksi ────────────────────────────────────
st.subheader("🏷️ Kelas Kendaraan yang Dideteksi")
badge_html = ""
for cls in CLASS_NAMES:
    color = CLASS_COLORS[cls]
    badge_html += (
        f'<span class="class-badge" style="background:rgba({_hex_parts(color)},0.15);'
        f'border:1px solid {color}; color:{color};">'
        f'<span class="class-dot" style="background:{color};"></span>'
        f'{cls.upper()}</span>'
    )

st.markdown(
    f'<div style="margin:1rem 0;">{badge_html}</div>',
    unsafe_allow_html=True,
)

# ── Quick Stats ─────────────────────────────────────────────
st.markdown("---")
st.subheader("⚡ Quick Info")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Model", "YOLO26", "Ultralytics SOTA 2026")
q2.metric("Dataset", "Vehicle Detection", "3 kelas")
q3.metric("Input Size", "640 × 640", "px")
q4.metric("Backend", "Apple MPS", "M4 Pro")

# ── CTA ─────────────────────────────────────────────────────
st.markdown("---")
st.info(
    "👈 **Mulai dari sidebar** — pilih **Detection** untuk upload gambar dan deteksi kendaraan, "
    "atau **Dashboard** untuk melihat history log deteksi."
)
