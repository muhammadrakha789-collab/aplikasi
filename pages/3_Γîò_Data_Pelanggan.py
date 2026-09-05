"""pages/3_Data_Pelanggan.py — Jelajahi dan unduh data pelanggan hasil segmentasi."""

import streamlit as st
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from style_and_pipeline import inject_style, render_hero, ORDER

st.set_page_config(page_title="Data Pelanggan — Client Ledger", page_icon="◆", layout="wide")
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
cust_search = st.sidebar.text_input("Cari Customer ID")

filtered = df[df["Segmen"].isin(segments)]
if cust_search:
    filtered = filtered[filtered["CustomerID"].astype(str).str.contains(cust_search)]

st.sidebar.markdown("---")
st.sidebar.download_button(
    "↓  Unduh Data Terfilter", data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="segmentasi_terfilter.csv", mime="text/csv", use_container_width=True,
)

render_hero("Data Pelanggan", "Jelajahi data pelanggan hasil segmentasi secara rinci, urutkan atau cari Customer ID tertentu.")

st.markdown(f'<div class="panel-title">Data Pelanggan — {len(filtered):,} baris</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-hint">Klik judul kolom untuk mengurutkan.</div>', unsafe_allow_html=True)
st.dataframe(filtered, use_container_width=True, hide_index=True, height=560)
