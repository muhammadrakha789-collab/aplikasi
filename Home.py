"""
Home.py — Gerbang login + halaman pembuka aplikasi Client Ledger.
Jalankan dengan: streamlit run Home.py
"""

import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from style_and_pipeline import (
    inject_style, render_hero, render_feature_row,
    inject_login_style, render_login_brand_panel, render_login_card_head,
)

st.set_page_config(
    page_title="Client Ledger — Segmentasi Pelanggan",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_style()

# ---------- Load konfigurasi login ----------
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
)

auth_status = st.session_state.get("authentication_status")

# ---------- Layar login (dipisah total dari tampilan dashboard) ----------
if not auth_status:
    inject_login_style()

    col_brand, col_form = st.columns([1, 1.2], gap="large")

    with col_brand:
        render_login_brand_panel()

    with col_form:
        render_login_card_head()

        try:
            authenticator.login()
        except Exception as e:
            st.error(f"Terjadi kesalahan saat login: {e}")

        if st.session_state.get("authentication_status") is False:
            st.error("Username atau password salah.")
        elif st.session_state.get("authentication_status") is None:
            st.info("Masukkan kredensial yang telah diberikan oleh administrator.")

# ---------- Halaman utama (setelah berhasil login) ----------
else:
    name = st.session_state.get("name", "Pengguna")
    st.sidebar.markdown(f"**Masuk sebagai:** {name}")
    authenticator.logout("Keluar", "sidebar")
    st.sidebar.markdown("---")

    render_hero(
        "Selamat Datang di Client Ledger",
        "Aplikasi Business Intelligence untuk menganalisis dan mengelompokkan pelanggan e-commerce "
        "berdasarkan pola transaksi (RFM), guna mendukung strategi retensi dan akuisisi pelanggan.",
    )

    st.markdown("##### Mulai dari menu di sidebar kiri:")
    render_feature_row()

    st.markdown("---")
    st.caption("Metode: RFM (Recency, Frequency, Monetary) + K-Means Clustering  ·  "
               "Dikembangkan sebagai bagian dari Comprehensive Project / Tugas Akhir")
