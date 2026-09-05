"""pages/4_Upload_Data_Baru.py — Proses data transaksi baru menjadi segmentasi otomatis.
Fitur inti yang membuat ini menjadi 'aplikasi', bukan sekadar laporan statis:
pengguna dapat mengunggah data transaksi apa pun (format mirip Online Retail II)
dan sistem akan otomatis membersihkan data, menghitung RFM, menentukan k optimal,
menjalankan clustering, dan menyediakan hasil segmentasi untuk diunduh.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from style_and_pipeline import (
    inject_style, render_hero, base_layout, COLOR_MAP, ORDER,
    clean_transactions, compute_rfm, run_clustering, find_optimal_k,
)

st.set_page_config(page_title="Unggah Data Baru — Client Ledger", page_icon="◆", layout="wide")
inject_style()

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(
    config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days']
)
authenticator.login()
if not st.session_state.get("authentication_status"):
    st.warning("Silakan login terlebih dahulu melalui halaman Home.")
    st.stop()

st.sidebar.markdown(f"**Masuk sebagai:** {st.session_state.get('name')}")
authenticator.logout("Keluar", "sidebar")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Format data yang diterima: CSV dengan kolom Invoice/InvoiceNo, "
    "InvoiceDate, Quantity, Price/UnitPrice, Customer ID/CustomerID."
)

render_hero(
    "Unggah Data Baru",
    "Proses data transaksi baru menjadi segmentasi pelanggan secara otomatis — "
    "tanpa perlu menjalankan notebook secara manual.",
)

uploaded = st.file_uploader("Unggah file transaksi (.csv)", type=["csv"])

if uploaded is None:
    st.info("Belum ada file yang diunggah. Silakan unggah data transaksi untuk memulai analisis.")
    st.markdown("""
    <div class="exec-summary">
        <b>Contoh format data yang didukung</b> (mengikuti struktur dataset Online Retail II):<br>
        <code>InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ---------- Tahap 1: Baca & bersihkan ----------
try:
    raw = pd.read_csv(uploaded, encoding="ISO-8859-1")
except Exception:
    uploaded.seek(0)
    raw = pd.read_csv(uploaded)

st.markdown('<div class="panel-title">Tahap 1 — Data Mentah</div>', unsafe_allow_html=True)
st.markdown(f'<div class="panel-hint">Ditemukan {len(raw):,} baris, {len(raw.columns)} kolom.</div>', unsafe_allow_html=True)
st.dataframe(raw.head(10), use_container_width=True, hide_index=True)

try:
    with st.spinner("Membersihkan data (menghapus retur, data kosong, dan duplikat)..."):
        cleaned = clean_transactions(raw)
except ValueError as e:
    st.error(f"Data tidak dapat diproses: {e}")
    st.stop()

st.success(f"Pembersihan selesai — {len(cleaned):,} baris valid dari {cleaned['CustomerID'].nunique():,} pelanggan unik.")

# ---------- Tahap 2: RFM ----------
st.markdown('<div class="panel-title">Tahap 2 — Perhitungan RFM</div>', unsafe_allow_html=True)
with st.spinner("Menghitung skor Recency, Frequency, Monetary per pelanggan..."):
    rfm = compute_rfm(cleaned)
st.dataframe(rfm.describe().round(1), use_container_width=True)

# ---------- Tahap 3: Tentukan k optimal ----------
st.markdown('<div class="panel-title">Tahap 3 — Menentukan Jumlah Segmen Optimal</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-hint">Dihitung otomatis menggunakan Silhouette Score. Kamu tetap bisa mengubah jumlah segmen secara manual di bawah.</div>', unsafe_allow_html=True)

with st.spinner("Menguji beberapa jumlah cluster (k=2 s.d. k=7)..."):
    k_eval = find_optimal_k(rfm)

col1, col2 = st.columns([1.3, 1])
with col1:
    st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
    fig_k = px.line(k_eval, x="k", y="silhouette", markers=True)
    fig_k.update_traces(line_color=COLOR_MAP["Champions"])
    fig_k.update_layout(**base_layout(yaxis_title="Silhouette Score", xaxis_title="Jumlah Cluster (k)"))
    st.plotly_chart(fig_k, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.dataframe(k_eval.round(3), use_container_width=True, hide_index=True)

best_k = int(k_eval.loc[k_eval["silhouette"].idxmax(), "k"])
k_choice = st.slider("Jumlah segmen (k)", min_value=2, max_value=7, value=min(best_k, 4) if best_k <= 7 else 4)
st.caption(f"Rekomendasi otomatis berdasarkan Silhouette Score tertinggi: k = {best_k}")

# ---------- Tahap 4: Clustering ----------
st.markdown('<div class="panel-title">Tahap 4 — Hasil Segmentasi</div>', unsafe_allow_html=True)
with st.spinner("Menjalankan K-Means dan memberi label segmen..."):
    result, sil, db = run_clustering(rfm, k=k_choice)

c1, c2, c3 = st.columns(3)
c1.metric("Silhouette Score", f"{sil:.3f}")
c2.metric("Davies-Bouldin Index", f"{db:.3f}")
c3.metric("Jumlah Pelanggan Diproses", f"{len(result):,}")

if k_choice == 4:
    labels_used = result["Segmen"].unique().tolist()
else:
    result["Segmen"] = "Segmen " + result["Cluster"].astype(str)

st.dataframe(result.head(20), use_container_width=True, hide_index=True)

fig_scatter = px.scatter(
    result.sample(min(1500, len(result)), random_state=42),
    x="Recency", y="Monetary", color="Segmen", size="Frequency",
    log_y=True, opacity=0.8, hover_data=["CustomerID"],
)
fig_scatter.update_layout(**base_layout())
st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
st.plotly_chart(fig_scatter, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.download_button(
    "↓  Unduh Hasil Segmentasi (CSV)",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="hasil_segmentasi_baru.csv",
    mime="text/csv",
)
st.caption(
    "Catatan: hasil ini diproses langsung dari data yang kamu unggah dan belum otomatis "
    "menggantikan data di halaman Ringkasan/Detail Segmen/Data Pelanggan. "
    "Unduh hasilnya, lalu ganti file 'rfm_segmentasi_pelanggan.csv' jika ingin menjadikannya data utama."
)
