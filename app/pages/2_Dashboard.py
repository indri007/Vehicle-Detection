# ─────────────────────────────────────────────────────────
#  pages/2_Dashboard.py — Analytics dashboard
# ─────────────────────────────────────────────────────────
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APP_NAME, CLASS_COLORS, CLASS_NAMES

st.set_page_config(
    page_title=f"Dashboard · {APP_NAME}",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
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

st.markdown(
    """
    <h1 style="
        background: linear-gradient(90deg, #7B61FF, #00D4FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
    ">📊 Analytics Dashboard</h1>
    <p style="color:#64748B; margin-bottom:1.5rem;">
        History log deteksi kendaraan dari n8n / Google Sheets.
    </p>
    """,
    unsafe_allow_html=True,
)

# ── Data Source Toggle ──────────────────────────────────────
data_source = st.radio(
    "Sumber Data",
    ["📊 Demo Data (Built-in)", "🔗 Google Sheets via n8n"],
    horizontal=True,
)

@st.cache_data(ttl=60)
def load_demo_data() -> pd.DataFrame:
    """Generate synthetic detection log data untuk demo."""
    np.random.seed(42)
    n = 80
    base = datetime.now() - timedelta(days=7)
    timestamps = [base + timedelta(minutes=int(i * 130 + np.random.randint(0, 60))) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp"     : timestamps,
            "image_name"    : [f"frame_{i:04d}.jpg" for i in range(n)],
            "car"           : np.random.randint(0, 8, n),
            "van"           : np.random.randint(0, 4, n),
            "bus"           : np.random.randint(0, 3, n),
            "avg_confidence": np.round(np.random.uniform(0.65, 0.95, n), 2),
        }
    ).assign(total_vehicles=lambda d: d["car"] + d["van"] + d["bus"])


@st.cache_data(ttl=60)
def load_sheets_data(sheet_url: str) -> pd.DataFrame:
    """
    Baca Google Sheets yang di-share sebagai CSV.
    Format: timestamp, image_name, car, van, bus, total_vehicles, avg_confidence
    """
    try:
        df = pd.read_csv(sheet_url)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
    except Exception as e:
        st.error(f"Gagal membaca Google Sheets: {e}")
        return pd.DataFrame()


# ── Load Data ───────────────────────────────────────────────
if data_source.startswith("📊"):
    df = load_demo_data()
    st.info("ℹ️ Menggunakan **demo data** sintetis. Hubungkan Google Sheets untuk data real-time.")
else:
    sheet_url = st.text_input(
        "🔗 Google Sheets CSV Export URL",
        placeholder="https://docs.google.com/spreadsheets/d/.../export?format=csv",
    )
    if not sheet_url:
        st.warning("Masukkan URL Google Sheets untuk melanjutkan.")
        st.stop()
    df = load_sheets_data(sheet_url)
    if df.empty:
        st.stop()

# ── Date Filter ──────────────────────────────────────────────
st.markdown("---")
col_f1, col_f2 = st.columns(2)
min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

with col_f1:
    start_date = st.date_input("Dari Tanggal", value=min_date, min_value=min_date, max_value=max_date)
with col_f2:
    end_date = st.date_input("Sampai Tanggal", value=max_date, min_value=min_date, max_value=max_date)

mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
df_filtered = df[mask].copy()

if df_filtered.empty:
    st.warning("Tidak ada data dalam rentang tanggal tersebut.")
    st.stop()

# ── KPI Metrics ──────────────────────────────────────────────
st.markdown("### 🔢 Summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Sesi", len(df_filtered))
m2.metric("Total Kendaraan", int(df_filtered["total_vehicles"].sum()))
m3.metric("Avg. Conf", f"{df_filtered['avg_confidence'].mean():.1%}")
m4.metric("Peak (sesi)", int(df_filtered["total_vehicles"].max()))
m5.metric("Avg/Sesi", f"{df_filtered['total_vehicles'].mean():.1f}")

st.markdown("---")

# ── Time Series ──────────────────────────────────────────────
col_ts, col_pie = st.columns([2, 1])

with col_ts:
    st.markdown("#### 📈 Total Kendaraan per Sesi")
    df_sorted = df_filtered.sort_values("timestamp")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=df_sorted["timestamp"],
        y=df_sorted["total_vehicles"],
        mode="lines+markers",
        line=dict(color="#00D4FF", width=2),
        marker=dict(size=5, color="#00D4FF"),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.07)",
        name="Total Kendaraan",
    ))
    fig_ts.add_hline(
        y=df_filtered["total_vehicles"].mean(),
        line_dash="dash",
        line_color="rgba(255,255,255,0.3)",
        annotation_text="Rata-rata",
    )
    fig_ts.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(t=20, b=20, l=0, r=0),
        font=dict(family="Inter"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

with col_pie:
    st.markdown("#### 🥧 Distribusi Kelas")
    class_totals = {cls: int(df_filtered[cls].sum()) for cls in CLASS_NAMES}
    fig_pie = go.Figure(go.Pie(
        labels=[c.upper() for c in class_totals.keys()],
        values=list(class_totals.values()),
        marker=dict(colors=list(CLASS_COLORS.values())),
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(size=13),
    ))
    fig_pie.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(t=20, b=20, l=20, r=20),
        font=dict(family="Inter"),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Stacked Bar per Hari ─────────────────────────────────────
st.markdown("#### 📅 Kendaraan per Hari")
df_daily = (
    df_filtered.copy()
    .assign(date=lambda d: d["timestamp"].dt.date)
    .groupby("date")[CLASS_NAMES]
    .sum()
    .reset_index()
)
fig_bar = px.bar(
    df_daily.melt(id_vars="date", var_name="Kelas", value_name="Jumlah"),
    x="date",
    y="Jumlah",
    color="Kelas",
    color_discrete_map=CLASS_COLORS,
    barmode="stack",
    template="plotly_dark",
    title=None,
)
fig_bar.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=280,
    margin=dict(t=10, b=20, l=0, r=0),
    font=dict(family="Inter"),
    legend=dict(orientation="h", y=-0.2),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Confidence Distribution ──────────────────────────────────
st.markdown("#### 🎯 Distribusi Confidence Score")
fig_hist = px.histogram(
    df_filtered,
    x="avg_confidence",
    nbins=20,
    template="plotly_dark",
    color_discrete_sequence=["#7B61FF"],
)
fig_hist.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    height=220,
    margin=dict(t=10, b=20, l=0, r=0),
    font=dict(family="Inter"),
    xaxis_title="Avg Confidence",
    yaxis_title="Frekuensi",
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Raw Data Table ───────────────────────────────────────────
with st.expander("📋 Raw Data"):
    st.dataframe(
        df_filtered.sort_values("timestamp", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=300,
    )
    csv = df_filtered.to_csv(index=False)
    st.download_button("⬇️ Export CSV", csv, "detection_log.csv", "text/csv")
