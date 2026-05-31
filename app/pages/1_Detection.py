# ─────────────────────────────────────────────────────────
#  pages/1_Detection.py — Halaman utama deteksi kendaraan
# ─────────────────────────────────────────────────────────
import io
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    APP_NAME,
    CLASS_COLORS,
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_IOU,
)
from utils.model import run_inference
from utils.n8n_client import send_to_n8n

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title=f"Detection · {APP_NAME}",
    page_icon="🎯",
    layout="wide",
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .result-card {
        background: linear-gradient(145deg, #111827, #1a2234);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .count-badge {
        background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,97,255,0.1));
        border: 1px solid rgba(0,212,255,0.3);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .count-number {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, #00D4FF, #7B61FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .count-label {
        font-size: 0.85rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .detection-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 2px;
    }
    .upload-zone {
        border: 2px dashed rgba(0,212,255,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(0,212,255,0.03);
    }
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #111827, #1a2234);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ──────────────────────────────────────────────────
st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #00D4FF, #7B61FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        margin-bottom: 0.2rem;
    ">🎯 Vehicle Detection</h1>
    <p style="color:#64748B; margin-bottom:1.5rem;">Upload gambar lalu biarkan YOLOv8 mendeteksi & menghitung kendaraan.</p>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar — Parameters ────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter Model")
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=DEFAULT_CONF,
        step=0.05,
        help="Semakin tinggi = hanya deteksi yang lebih yakin yang ditampilkan.",
    )
    iou_thresh = st.slider(
        "NMS IOU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_IOU,
        step=0.05,
        help="Threshold untuk Non-Maximum Suppression.",
    )

    st.markdown("---")
    st.markdown("### 📡 n8n Logging")
    auto_log = st.toggle(
        "Auto-log ke n8n",
        value=True,
        help="Setiap deteksi otomatis dikirim ke n8n webhook.",
    )
    st.markdown("---")
    st.markdown(
        "**Kelas:**\n"
        + "".join(
            f"\n- <span style='color:{CLASS_COLORS[c]};'>●</span> **{c.upper()}**"
            for c in CLASS_NAMES
        ),
        unsafe_allow_html=True,
    )

# ── Upload Section ──────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁 Upload Gambar Kendaraan",
    type=["jpg", "jpeg", "png", "webp"],
    help="Format yang didukung: JPG, JPEG, PNG, WebP",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown(
        """
        <div class="upload-zone">
            <p style="font-size:2rem; margin:0;">📷</p>
            <p style="color:#64748B; margin:0.5rem 0 0;">
                Drag & drop gambar di sini, atau klik tombol di atas
            </p>
            <p style="color:#374151; font-size:0.75rem; margin:0.3rem 0 0;">
                JPG · PNG · WebP
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ── Inference ───────────────────────────────────────────────
image = Image.open(uploaded_file).convert("RGB")

col_img, col_result = st.columns([1.2, 1], gap="large")

with col_img:
    st.markdown("**📸 Gambar Input**")
    st.image(image, use_container_width=True, caption=uploaded_file.name)

with col_result:
    with st.spinner("🔍 Mendeteksi kendaraan..."):
        t0 = time.perf_counter()
        annotated_img, counts, detections = run_inference(image, conf_thresh, iou_thresh)
        elapsed = time.perf_counter() - t0

    st.markdown(f"**✅ Hasil Deteksi** &nbsp; <span style='color:#64748B;font-size:0.8rem;'>({elapsed*1000:.0f} ms)</span>", unsafe_allow_html=True)
    st.image(annotated_img, use_container_width=True, caption="Hasil anotasi YOLOv8")

    # Tombol download annotated image
    buf = io.BytesIO()
    annotated_img.save(buf, format="PNG")
    st.download_button(
        "⬇️ Download Hasil",
        data=buf.getvalue(),
        file_name=f"detected_{uploaded_file.name}",
        mime="image/png",
        use_container_width=True,
    )

# ── Count Summary ────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Rekap Kendaraan Terdeteksi")

total = sum(counts.values())
c_bus, c_car, c_van, c_total = st.columns(4)

for col, (cls, emoji) in zip(
    [c_car, c_van, c_bus],
    [("car", "🚗"), ("van", "🚐"), ("bus", "🚌")],
):
    color = CLASS_COLORS[cls]
    with col:
        st.markdown(
            f"""
            <div class="count-badge">
                <div style="font-size:1.8rem;">{emoji}</div>
                <div class="count-number" style="
                    background: linear-gradient(90deg, {color}, {color}aa);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">{counts[cls]}</div>
                <div class="count-label">{cls}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with c_total:
    st.markdown(
        f"""
        <div class="count-badge">
            <div style="font-size:1.8rem;">🚦</div>
            <div class="count-number">{total}</div>
            <div class="count-label">Total</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Summary Text ─────────────────────────────────────────────
if detections:
    summary_parts = [f"**{v} {k}**" for k, v in counts.items() if v > 0]
    summary_text  = ", ".join(summary_parts)
    st.success(f"🔎 Terdeteksi: {summary_text} — Total **{total} kendaraan**")
else:
    st.warning("⚠️ Tidak ada kendaraan yang terdeteksi. Coba turunkan Confidence Threshold.")

# ── Bar Chart ────────────────────────────────────────────────
if detections:
    df_counts = pd.DataFrame(
        [{"Kelas": k.upper(), "Jumlah": v, "Warna": CLASS_COLORS[k]} for k, v in counts.items()]
    )
    fig = px.bar(
        df_counts,
        x="Kelas",
        y="Jumlah",
        color="Kelas",
        color_discrete_map={k.upper(): v for k, v in CLASS_COLORS.items()},
        text="Jumlah",
        template="plotly_dark",
        title="Distribusi Kendaraan",
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        height=280,
        margin=dict(t=40, b=20, l=0, r=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Detail Table ─────────────────────────────────────────────
if detections:
    with st.expander("📋 Detail Semua Deteksi"):
        df_det = pd.DataFrame(
            [
                {
                    "#"          : i + 1,
                    "Kelas"      : d["class"].upper(),
                    "Confidence" : f"{d['confidence']:.1%}",
                    "BBox [x1,y1,x2,y2]": [round(v, 1) for v in d["bbox"]],
                }
                for i, d in enumerate(detections)
            ]
        )
        st.dataframe(df_det, use_container_width=True, hide_index=True)

# ── n8n Log ──────────────────────────────────────────────────
st.markdown("---")
col_log1, col_log2 = st.columns([3, 1])
with col_log1:
    st.markdown("### 📡 Log ke n8n")
    st.caption(
        "Kirim hasil deteksi ini ke n8n untuk di-log ke Google Sheets "
        "dan memicu alert jika traffic padat."
    )

with col_log2:
    manual_log = st.button(
        "📤 Kirim ke n8n",
        use_container_width=True,
        type="primary",
        disabled=(len(detections) == 0),
    )

if auto_log and detections:
    ok, msg = send_to_n8n(uploaded_file.name, counts, detections)
    if ok:
        st.toast(msg, icon="✅")
    else:
        st.toast(msg, icon="⚠️")

if manual_log:
    with st.spinner("Mengirim ke n8n..."):
        ok, msg = send_to_n8n(uploaded_file.name, counts, detections)
    if ok:
        st.success(msg)
    else:
        st.error(msg)
