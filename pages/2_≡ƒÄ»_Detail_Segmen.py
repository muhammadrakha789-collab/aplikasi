"""pages/2_Detail_Segmen.py — Profil RFM tiap segmen dan rekomendasi strategi."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from style_and_pipeline import (
    inject_style, render_hero, base_layout, COLOR_MAP, INITIAL_MAP, ORDER, INSIGHTS, CARD
)

st.set_page_config(page_title="Detail Segmen — Client Ledger", page_icon="◆", layout="wide")
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

@st.cache_data
def load_data():
    return pd.read_csv("rfm_segmentasi_pelanggan.csv")

df = load_data()

st.sidebar.markdown("### Saring Data")
segments = st.sidebar.multiselect("Segmen Pelanggan", options=ORDER, default=ORDER)
filtered = df[df["Segmen"].isin(segments)]

render_hero("Detail Segmen", "Profil RFM relatif tiap segmen beserta rekomendasi strategi pemasaran yang sesuai.")

summary = (
    filtered.groupby("Segmen").agg(Jumlah=("CustomerID", "count"), Recency=("Recency", "mean"),
                                     Frequency=("Frequency", "mean"), Monetary=("Monetary", "mean"),
                                     TotalRevenue=("Monetary", "sum")).round(1).reindex(ORDER).dropna(how="all").reset_index()
)
if len(summary):
    summary["Persentase"] = (summary["Jumlah"] / summary["Jumlah"].sum() * 100).round(1)

st.markdown('<div class="panel-title">Profil RFM Relatif per Segmen</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-hint">Semakin luas area suatu segmen, semakin unggul ia di ketiga aspek RFM.</div>', unsafe_allow_html=True)

if len(summary):
    norm = summary.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        norm[col + "_n"] = (norm[col] - norm[col].min()) / (norm[col].max() - norm[col].min() + 1e-9)
    norm["Recency_score"] = 1 - norm["Recency_n"]

    st.markdown('<div class="chart-frame">', unsafe_allow_html=True)
    fig_radar = go.Figure()
    for _, row in norm.iterrows():
        c = COLOR_MAP.get(row["Segmen"], "#888")
        fig_radar.add_trace(go.Scatterpolar(
            r=[row["Recency_score"], row["Frequency_n"], row["Monetary_n"], row["Recency_score"]],
            theta=["Recency (semakin baru)", "Frequency", "Monetary", "Recency (semakin baru)"],
            fill="toself", name=row["Segmen"], line_color=c, fillcolor=c, opacity=0.22,
        ))
    fig_radar.update_layout(**base_layout(
        polar=dict(bgcolor=CARD, radialaxis=dict(visible=True, range=[0, 1], gridcolor="#EDEAE0"),
                   angularaxis=dict(gridcolor="#EDEAE0")),
        showlegend=True,
    ))
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="panel-title">Rekomendasi Strategi per Segmen</div>', unsafe_allow_html=True)
for seg in ORDER:
    if seg not in summary["Segmen"].values:
        continue
    row = summary[summary["Segmen"] == seg].iloc[0]
    color = COLOR_MAP[seg]
    st.markdown(f"""
        <div class="seg-card">
            <div class="seg-badge" style="--seg-color:{color}">{INITIAL_MAP[seg]}</div>
            <div>
                <b>{seg}</b>
                <span class="desc">{INSIGHTS[seg]}</span>
                <div class="seg-stat">
                    <span>Pelanggan <b>{int(row['Jumlah']):,}</b> ({row['Persentase']}%)</span>
                    <span>Recency <b>{row['Recency']:.0f} hari</b></span>
                    <span>Frequency <b>{row['Frequency']:.1f}×</b></span>
                    <span>Monetary <b>£{row['Monetary']:,.0f}</b></span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="panel-title">Tabel Ringkasan Segmen</div>', unsafe_allow_html=True)
if len(summary):
    st.dataframe(summary[["Segmen", "Jumlah", "Persentase", "Recency", "Frequency", "Monetary", "TotalRevenue"]],
                 use_container_width=True, hide_index=True)
