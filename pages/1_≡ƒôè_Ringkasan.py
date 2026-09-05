"""pages/1_Ringkasan.py — Ikhtisar KPI, sebaran pelanggan, kontribusi revenue."""

import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from style_and_pipeline import inject_style, render_hero, base_layout, COLOR_MAP, ORDER, GLOSSARY, CARD

st.set_page_config(page_title="Ringkasan — Client Ledger", page_icon="◆", layout="wide")
inject_style()

# ---------- Auth guard ----------
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

# ---------- Data ----------
@st.cache_data
def load_data():
    return pd.read_csv("rfm_segmentasi_pelanggan.csv")

df = load_data()

st.sidebar.markdown("### Saring Data")
segments = st.sidebar.multiselect("Segmen Pelanggan", options=ORDER, default=ORDER)
filtered = df[df["Segmen"].isin(segments)]

render_hero("Ringkasan", "Ikhtisar performa dan sebaran pelanggan berdasarkan hasil segmentasi RFM + K-Means.")

with st.expander("ℹ️  Belum familiar dengan istilah di dashboard ini?"):
    for term, desc in GLOSSARY:
        st.markdown(f"""<div class="gloss-item"><b>{term}</b><span>{desc}</span></div>""", unsafe_allow_html=True)

summary_all = (
    df.groupby("Segmen").agg(Jumlah=("CustomerID", "count"), Recency=("Recency", "mean"),
                               Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                               TotalRevenue=("Monetary", "sum")).round(1).reindex(ORDER).dropna(how="all").reset_index()
)
summary_all["Persentase"] = (summary_all["Jumlah"] / summary_all["Jumlah"].sum() * 100).round(1)

summary = (
    filtered.groupby("Segmen").agg(Jumlah=("CustomerID", "count"), Recency=("Recency", "mean"),
                                     Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                                     TotalRevenue=("Monetary", "sum")).round(1).reindex(ORDER).dropna(how="all").reset_index()
)
if len(summary):
    summary["Persentase"] = (summary["Jumlah"] / summary["Jumlah"].sum() * 100).round(1)

st.markdown(f"""
<div class="kpi-row">
    <div class="kpi-cell"><div class="kpi-label">Total Pelanggan</div><div class="kpi-value">{len(filtered):,}</div><div class="kpi-sub">dari {len(df):,} total data</div></div>
    <div class="kpi-cell"><div class="kpi-label">Total Revenue</div><div class="kpi-value">£{filtered['Monetary'].sum():,.0f}</div><div class="kpi-sub">akumulasi segmen terpilih</div></div>
    <div class="kpi-cell"><div class="kpi-label">Rata-rata Frequency</div><div class="kpi-value">{filtered['Frequency'].mean():.1f}×</div><div class="kpi-sub">transaksi per pelanggan</div></div>
    <div class="kpi-cell"><div class="kpi-label">Rata-rata Recency</div><div class="kpi-value">{filtered['Recency'].mean():.0f} hari</div><div class="kpi-sub">sejak transaksi terakhir</div></div>
</div>
""", unsafe_allow_html=True)

if len(summary_all) == 4:
    top_seg = summary_all.loc[summary_all["TotalRevenue"].idxmax()]
    risk_seg = summary_all.loc[summary_all["Segmen"] == "Lost/Churned"].iloc[0]
    top_rev_share = (top_seg["TotalRevenue"] / summary_all["TotalRevenue"].sum() * 100)
    st.markdown(f"""
    <div class="exec-summary">
        <b>Ringkasan.</b> Dari {int(summary_all['Jumlah'].sum()):,} pelanggan, segmen
        <b>{top_seg['Segmen']}</b> hanya berjumlah {top_seg['Persentase']}% dari total pelanggan,
        namun menyumbang <b>{top_rev_share:.0f}% dari seluruh pendapatan</b>. Sementara itu,
        <b>{risk_seg['Persentase']}% pelanggan</b> sudah masuk kategori <b>Lost/Churned</b>
        (rata-rata {risk_seg['Recency']:.0f} hari tanpa transaksi).
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="panel-title">Sebaran Pelanggan — Recency × Monetary</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-hint">Titik kiri-bawah = pelanggan baru & bernilai tinggi. Titik kanan-bawah = berisiko hilang. Ukuran titik = frekuensi belanja.</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([1.4, 1])
with col_a:
    st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
    if len(filtered):
        fig = px.scatter(
            filtered.sample(min(1500, len(filtered)), random_state=42),
            x="Recency", y="Monetary", color="Segmen", size="Frequency",
            color_discrete_map=COLOR_MAP, log_y=True, opacity=0.8,
            hover_data=["CustomerID"], category_orders={"Segmen": ORDER},
        )
        fig.update_traces(marker=dict(line=dict(width=0.5, color="#FFFFFF")))
        fig.update_layout(**base_layout(legend=dict(orientation="h", y=-0.22, bgcolor="rgba(0,0,0,0)")))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Tidak ada data untuk filter yang dipilih.")
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
    if len(summary):
        fig_pie = px.pie(summary, names="Segmen", values="Jumlah", hole=0.66,
                          color="Segmen", color_discrete_map=COLOR_MAP, category_orders={"Segmen": ORDER})
        fig_pie.update_traces(textinfo="percent", textposition="outside", marker=dict(line=dict(color=CARD, width=2)))
        fig_pie.update_layout(**base_layout(showlegend=True))
        st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="panel-title">Kontribusi Revenue per Segmen</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
if len(summary):
    fig2 = px.bar(summary.sort_values("TotalRevenue"), x="TotalRevenue", y="Segmen", orientation="h",
                  color="Segmen", color_discrete_map=COLOR_MAP, text="TotalRevenue")
    fig2.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
    fig2.update_layout(**base_layout(showlegend=False, yaxis_title="", xaxis_title="Total Revenue (£)"))
    st.plotly_chart(fig2, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
