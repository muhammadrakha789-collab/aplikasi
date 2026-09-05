"""
Modul bersama: styling (tema Client Ledger) + fungsi pipeline RFM & Clustering.
Diimpor oleh Home.py dan semua file di folder pages/.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ---------- Token warna ----------
IVORY = "#F7F4EC"
INK = "#12131A"
INK_SOFT = "#1A1C27"
CARD = "#FFFFFF"
NAVY_DEEP = "#0F1938"
NAVY = "#1C2B57"
TEXT = "#1A1B22"
TEXT_MUTED = "#54566A"
SIDEBAR_TEXT = "#EDEBE0"
SIDEBAR_MUTED = "#9497AA"
GOLD = "#A9822B"          # aksen brass — dipakai tipis-tipis, bukan dominan
GOLD_SOFT = "#C9A96A"
HAIRLINE = "#DDD9CB"

COLOR_MAP = {
    "Champions": "#0F1938",
    "Promising/New Active": "#2B4590",
    "At Risk": "#8A6D2E",
    "Lost/Churned": "#8C2F3B",
}
INITIAL_MAP = {"Champions": "C", "Promising/New Active": "P", "At Risk": "R", "Lost/Churned": "L"}
ORDER = ["Champions", "Promising/New Active", "At Risk", "Lost/Churned"]

INSIGHTS = {
    "Champions": "Pelanggan paling bernilai — baru bertransaksi, frekuensi tinggi, nilai belanja terbesar. "
                 "Diprioritaskan untuk program loyalitas dan akses awal produk baru.",
    "Promising/New Active": "Baru aktif dengan potensi berkembang. "
                             "Didorong lewat rekomendasi personal dan insentif transaksi kedua.",
    "At Risk": "Sebelumnya aktif, kini mulai jarang bertransaksi. "
               "Perlu kampanye keterlibatan ulang sebelum berpindah menjadi Lost.",
    "Lost/Churned": "Sudah lama tidak bertransaksi. "
                     "Kandidat kampanye win-back atau evaluasi ulang biaya retensi.",
}

GLOSSARY = [
    ("Recency (Kebaruan)", "Sudah berapa hari sejak pelanggan terakhir kali belanja. Semakin kecil angkanya, semakin baru ia aktif."),
    ("Frequency (Frekuensi)", "Berapa kali pelanggan tersebut sudah bertransaksi. Semakin sering, semakin loyal."),
    ("Monetary (Nilai)", "Total uang yang sudah dihabiskan pelanggan tersebut sepanjang periode data."),
    ("K-Means Clustering", "Metode statistik yang mengelompokkan pelanggan dengan pola R, F, M yang mirip menjadi satu segmen — tanpa ditentukan manual, algoritma yang menemukan polanya."),
]

# Badge fitur di halaman Home — pengganti emoji, mengikuti bahasa visual seg-badge
FEATURE_ICONS = [
    {"glyph": "I", "title": "Ringkasan", "desc": "Ikhtisar KPI, sebaran pelanggan, dan kontribusi revenue per segmen."},
    {"glyph": "II", "title": "Detail Segmen", "desc": "Profil RFM tiap segmen dan rekomendasi strategi pemasaran."},
    {"glyph": "III", "title": "Data Pelanggan", "desc": "Jelajahi dan unduh data pelanggan hasil segmentasi."},
    {"glyph": "IV", "title": "Unggah Data Baru", "desc": "Proses data transaksi baru menjadi segmentasi otomatis."},
]

_FONT_LINK = (
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&'
    'family=Inter:wght@400;500;600;700&'
    'family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">'
)

# ---------- CSS dasar (dipakai di semua halaman) ----------
_BASE_CSS = f"""
<style>
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    [data-testid="stDecoration"] {{ display: none; }}

    .stApp {{ background: {IVORY}; color: {TEXT}; }}
    html, body, * {{ font-family: 'Inter', sans-serif; }}
    * {{ -webkit-font-smoothing: antialiased; }}

    section[data-testid="stSidebar"] {{ background: {INK}; }}
    section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] p {{ color: {SIDEBAR_TEXT} !important; }}
    section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: {SIDEBAR_MUTED} !important;
    }}
    section[data-testid="stSidebar"] h3 {{
        font-family: 'IBM Plex Mono', monospace !important; font-size: 0.72rem !important;
        text-transform: uppercase; letter-spacing: 0.16em; color: {SIDEBAR_MUTED} !important; font-weight: 500 !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span {{
        font-family: 'Inter', sans-serif !important; font-size: 0.88rem !important; color: {SIDEBAR_TEXT} !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li div[aria-current="page"] span {{
        color: {GOLD_SOFT} !important; font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {{
        border-bottom: 1px solid #2A2C3A; padding-bottom: 12px; margin-bottom: 6px;
    }}

    h1, h2, h3, h4, h5 {{ font-family: 'Fraunces', serif !important; font-weight: 600 !important; color: {TEXT} !important; letter-spacing: -0.01em; }}
    p, span, label {{ color: {TEXT}; }}
    .stCaption, [data-testid="stCaptionContainer"] p {{ color: {TEXT_MUTED} !important; }}

    .hero {{ padding: 30px 36px; margin-bottom: 22px; background: {INK}; border-bottom: 3px double #3A3C4C; }}
    .hero-top {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; }}
    .crest {{
        width: 44px; height: 44px; border: 1.5px solid #9099B8; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Fraunces', serif; font-size: 1.2rem; color: #C9CEE0; flex-shrink: 0;
    }}

    .kpi-row {{ display: flex; border: 1px solid {INK}; background: {CARD}; flex-wrap: wrap; }}
    .kpi-cell {{ flex: 1; min-width: 160px; padding: 18px 22px; border-right: 1px solid #E3E0D4; }}
    .kpi-cell:last-child {{ border-right: none; }}
    .kpi-label {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; text-transform: uppercase;
        letter-spacing: 0.1em; color: {TEXT_MUTED}; margin-bottom: 10px;
    }}
    .kpi-value {{
        font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: {NAVY_DEEP};
        font-variant-numeric: tabular-nums;
    }}
    .kpi-sub {{ font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 4px; }}

    .exec-summary {{
        background: {CARD}; border: 1px solid {INK}; border-left: 4px solid {NAVY_DEEP};
        padding: 20px 24px; margin: 20px 0; font-size: 0.94rem; line-height: 1.75; color: {TEXT};
    }}
    .exec-summary b {{ color: {NAVY_DEEP}; }}

    .panel-title {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; text-transform: uppercase;
        letter-spacing: 0.12em; color: {TEXT_MUTED}; margin: 26px 0 6px; display: flex; align-items: center; gap: 10px;
    }}
    .panel-title::after {{ content: ""; flex: 1; height: 1px; background: #D8D4C4; }}
    .panel-hint {{ font-size: 0.82rem; color: {TEXT_MUTED}; margin-bottom: 14px; font-style: italic; }}

    .chart-frame {{ background: {CARD}; border: 1px solid {INK}; padding: 18px 20px; }}

    .seg-card {{ display: flex; gap: 16px; padding: 18px 4px; border-bottom: 1px solid #DDD9CB; }}
    .seg-badge {{
        width: 40px; height: 40px; border-radius: 50%; flex-shrink: 0; background: {CARD};
        border: 2px solid var(--seg-color); display: flex; align-items: center; justify-content: center;
        font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.05rem; color: var(--seg-color);
    }}
    .seg-card b {{ font-size: 1.12rem; font-family: 'Fraunces', serif; color: {TEXT}; }}
    .seg-card .desc {{ color: {TEXT_MUTED}; font-size: 0.86rem; margin-top: 4px; display: block; line-height: 1.55; }}
    .seg-stat {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 6px; display: flex; gap: 14px; flex-wrap: wrap; }}
    .seg-stat b {{ color: {TEXT}; }}

    .feature-row {{ display: flex; gap: 0; border: 1px solid {INK}; background: {CARD}; flex-wrap: wrap; margin-top: 6px; }}
    .feature-cell {{ flex: 1; min-width: 200px; padding: 20px 22px; border-right: 1px solid #E3E0D4; }}
    .feature-cell:last-child {{ border-right: none; }}
    .feature-num {{
        display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px;
        border: 1px solid {GOLD}; color: {GOLD}; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        margin-bottom: 12px;
    }}
    .feature-cell b {{ display: block; font-family: 'Fraunces', serif; font-size: 1.02rem; margin-bottom: 6px; }}
    .feature-cell span {{ color: {TEXT_MUTED}; font-size: 0.82rem; line-height: 1.55; }}

    .gloss-item {{ padding: 10px 0; border-bottom: 1px solid #E3E0D4; }}
    .gloss-item b {{ font-family: 'Fraunces', serif; font-size: 1rem; color: {NAVY_DEEP}; }}
    .gloss-item span {{ display: block; color: {TEXT_MUTED}; font-size: 0.85rem; margin-top: 3px; line-height: 1.55; }}

    div[data-testid="stMetric"] {{ background: {CARD}; padding: 10px; border: 1px solid {INK}; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {INK}; }}
    .stTabs [data-baseweb="tab"] {{
        color: {TEXT_MUTED}; background: transparent; font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.08em; padding: 0 18px 12px;
    }}
    .stTabs [aria-selected="true"] {{ color: {NAVY_DEEP} !important; border-bottom: 2px solid {NAVY_DEEP} !important; }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {NAVY_DEEP} !important; }}

    span[data-baseweb="tag"], div[data-baseweb="tag"] {{
        background-color: #1C1E29 !important; border: 1px solid #9099B8 !important; border-radius: 0 !important;
    }}
    span[data-baseweb="tag"] *, div[data-baseweb="tag"] * {{
        color: {SIDEBAR_TEXT} !important; fill: {SIDEBAR_TEXT} !important; background-color: transparent !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: #1C1E29 !important; border: 1px solid #3A3C4C !important; border-radius: 0 !important;
    }}
    div[data-baseweb="select"] > div:hover {{ border-color: {GOLD_SOFT} !important; }}
    div[data-baseweb="popover"] ul {{ background-color: {INK} !important; border: 1px solid #3A3C4C !important; }}
    div[data-baseweb="popover"] li {{ color: {SIDEBAR_TEXT} !important; }}
    div[data-baseweb="popover"] li:hover {{ background-color: #1C1E29 !important; }}

    section[data-testid="stSidebar"] input {{
        background-color: #1C1E29 !important; color: {SIDEBAR_TEXT} !important; border: 1px solid #3A3C4C !important;
        border-radius: 0 !important;
    }}

    /* ---- Form field & tombol, berlaku di seluruh halaman (termasuk login) ---- */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {{
        border-radius: 0 !important; border: 1px solid {INK} !important; background: {CARD} !important;
        color: {TEXT} !important; font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important;
        padding: 10px 12px !important;
    }}
    .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {{
        border-color: {GOLD} !important; box-shadow: 0 0 0 1px {GOLD} !important;
    }}
    .stTextInput label, .stNumberInput label, .stTextArea label {{
        font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important;
        text-transform: uppercase; letter-spacing: 0.1em; color: {TEXT_MUTED} !important; font-weight: 500 !important;
    }}

    .stDownloadButton button, .stButton button {{
        background: transparent !important; color: {TEXT} !important; border: 1px solid {INK} !important;
        border-radius: 0 !important; font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.74rem !important; letter-spacing: 0.06em; text-transform: uppercase;
        transition: all 0.15s ease;
    }}
    .stDownloadButton button:hover, .stButton button:hover {{ background: {INK} !important; color: #fff !important; border-color: {INK} !important; }}
    .stFormSubmitButton button {{
        background: {NAVY_DEEP} !important; color: {IVORY} !important; border: 1px solid {NAVY_DEEP} !important;
        border-radius: 0 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.76rem !important;
        letter-spacing: 0.1em; text-transform: uppercase; padding: 10px 0 !important; width: 100%;
    }}
    .stFormSubmitButton button:hover {{ background: {GOLD} !important; border-color: {GOLD} !important; color: {INK} !important; }}

    section[data-testid="stSidebar"] .stButton button {{
        color: {SIDEBAR_TEXT} !important; border: 1px solid #9099B8 !important;
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{ background: #1C1E29 !important; }}

    div[data-testid="stForm"] {{ border: none !important; padding: 0 !important; background: transparent !important; }}

    div[data-testid="stAlert"] {{ border-radius: 0 !important; font-family: 'Inter', sans-serif !important; border: 1px solid {INK} !important; }}
    div[data-testid="stFileUploaderDropzone"] {{
        border-radius: 0 !important; background: {CARD} !important; border: 1.5px dashed #B7B2A0 !important;
    }}

    .streamlit-expanderHeader {{
        background: {CARD} !important; border: 1px solid {INK} !important; font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; color: {TEXT} !important;
    }}
    .streamlit-expanderContent {{ background: {CARD} !important; border: 1px solid {INK} !important; border-top: none !important; }}

    [data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace !important; color: {NAVY_DEEP} !important; }}

    hr {{ border-color: #DDD9CB !important; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {INK}; }}
</style>
"""


def inject_style():
    st.markdown(_FONT_LINK + _BASE_CSS, unsafe_allow_html=True)


def base_layout(**overrides):
    layout = dict(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter, sans-serif", size=12),
        xaxis=dict(gridcolor="#EDEAE0", zerolinecolor="#EDEAE0", color=TEXT_MUTED),
        yaxis=dict(gridcolor="#EDEAE0", zerolinecolor="#EDEAE0", color=TEXT_MUTED),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED)),
        margin=dict(t=10, l=10, r=10, b=10),
    )
    layout.update(overrides)
    return layout


def render_hero(title, subtitle, eyebrow="Client Ledger · Business Intelligence"):
    st.markdown(f"""
    <div class="hero">
        <div class="hero-top">
            <div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;text-transform:uppercase;
                            letter-spacing:0.22em;color:#9099B8;margin-bottom:10px;">
                    {eyebrow}
                </div>
                <h1 style="font-size:2rem;margin:0 0 10px;color:#F7F5EF !important;-webkit-text-fill-color:#F7F5EF !important;">
                    {title}
                </h1>
                <p style="color:#C4C6D2 !important;margin:0;font-size:0.9rem;max-width:600px;line-height:1.65;">
                    {subtitle}
                </p>
            </div>
            <div class="crest">◆</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_row(items=None):
    """Baris fitur bergaya kartu bernomor (pengganti emoji) untuk halaman Home."""
    items = items or FEATURE_ICONS
    cells = "".join(f"""
        <div class="feature-cell">
            <div class="feature-num">{it['glyph']}</div>
            <b>{it['title']}</b>
            <span>{it['desc']}</span>
        </div>
    """ for it in items)
    st.markdown(f'<div class="feature-row">{cells}</div>', unsafe_allow_html=True)


# ================= HALAMAN LOGIN — dipisah, elegan, split-screen =================

_LOGIN_CSS = f"""
<style>
    section[data-testid="stSidebar"] {{ display: none !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; }}

    .stApp {{ background: {IVORY}; }}

    .login-brand {{
        position: fixed; top: 0; left: 0; bottom: 0; width: 40%; min-width: 340px;
        background: linear-gradient(165deg, {INK} 0%, {NAVY_DEEP} 100%);
        display: flex; flex-direction: column; justify-content: space-between;
        padding: 56px 48px; z-index: 0; box-sizing: border-box;
        border-right: 1px solid #2A2C3A;
    }}
    .login-brand .crest-lg {{
        width: 56px; height: 56px; border: 1.5px solid {GOLD_SOFT}; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Fraunces', serif; font-size: 1.5rem; color: {GOLD_SOFT};
    }}
    .login-brand .brand-eyebrow {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; text-transform: uppercase;
        letter-spacing: 0.24em; color: #9099B8; margin: 26px 0 14px;
    }}
    .login-brand h1 {{
        font-family: 'Fraunces', serif !important; font-weight: 600 !important; font-size: 2.6rem;
        color: {IVORY} !important; -webkit-text-fill-color: {IVORY} !important; margin: 0 0 18px; line-height: 1.12;
    }}
    .login-brand p {{
        color: #B9BCCB !important; font-size: 0.96rem; line-height: 1.75; max-width: 380px; font-weight: 400;
    }}
    .login-brand .brand-quote {{
        font-family: 'Fraunces', serif; font-style: italic; font-size: 1.05rem; color: {GOLD_SOFT} !important;
        line-height: 1.6; border-left: 2px solid {GOLD}; padding-left: 18px; max-width: 360px;
    }}
    .login-brand .brand-foot {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; letter-spacing: 0.1em;
        color: #6D7089; text-transform: uppercase;
    }}

    div[data-testid="stAppViewBlockContainer"] {{
        margin-left: 40%; max-width: 480px; padding-top: 9vh !important;
    }}
    @media (max-width: 900px) {{
        .login-brand {{ display: none; }}
        div[data-testid="stAppViewBlockContainer"] {{ margin-left: 0; max-width: 92%; padding-top: 6vh !important; }}
    }}

    .login-card-head {{ margin-bottom: 30px; }}
    .login-card-head .eyebrow {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; text-transform: uppercase;
        letter-spacing: 0.2em; color: {TEXT_MUTED}; margin-bottom: 12px;
    }}
    .login-card-head h2 {{
        font-family: 'Fraunces', serif !important; font-size: 1.7rem !important; margin: 0 0 8px !important;
        color: {TEXT} !important;
    }}
    .login-card-head .gold-rule {{ width: 46px; height: 2px; background: {GOLD}; margin: 16px 0 18px; }}
    .login-card-head p {{ color: {TEXT_MUTED} !important; font-size: 0.88rem; line-height: 1.6; margin: 0; }}
</style>
"""


def inject_login_style():
    """CSS khusus layar login: sidebar disembunyikan, layout dipecah dua panel."""
    st.markdown(_LOGIN_CSS, unsafe_allow_html=True)


def render_login_brand_panel():
    """Panel kiri (dekoratif) pada layar login — hanya HTML statis, tanpa widget."""
    st.markdown(f"""
    <div class="login-brand">
        <div>
            <div class="crest-lg">◆</div>
            <div class="brand-eyebrow">Business Intelligence · Akses Terbatas</div>
            <h1>Client<br>Ledger</h1>
            <p>Platform analitik untuk membaca perilaku pelanggan e-commerce secara jernih —
            menyatukan riwayat transaksi menjadi segmen yang bisa langsung ditindaklanjuti.</p>
        </div>
        <div>
            <div class="brand-quote">"Data yang rapi adalah awal dari keputusan yang tepat."</div>
            <div class="brand-foot" style="margin-top:28px;">RFM Analysis · K-Means Clustering</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_login_card_head(title="Masuk ke Akun Anda",
                            subtitle="Gunakan kredensial yang telah diberikan oleh administrator untuk mengakses dashboard."):
    st.markdown(f"""
    <div class="login-card-head">
        <div class="eyebrow">Client Ledger</div>
        <h2>{title}</h2>
        <div class="gold-rule"></div>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


# ================= PIPELINE RFM + CLUSTERING (dipakai halaman Upload Data) =================

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Menyesuaikan nama kolom & membersihkan data transaksi mentah (format ala Online Retail II)."""
    rename_map = {
        'Invoice': 'InvoiceNo', 'InvoiceNo': 'InvoiceNo',
        'Price': 'UnitPrice', 'UnitPrice': 'UnitPrice',
        'Customer ID': 'CustomerID', 'CustomerID': 'CustomerID',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df.columns = df.columns.str.strip()

    required = {'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', 'CustomerID'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")

    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df.drop_duplicates()
    df['CustomerID'] = df['CustomerID'].astype(int)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm


def run_clustering(rfm: pd.DataFrame, k: int = 4):
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(lambda x: np.log1p(x))
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    rfm = rfm.copy()
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    sil = silhouette_score(rfm_scaled, rfm['Cluster'])
    db = davies_bouldin_score(rfm_scaled, rfm['Cluster'])

    # Label otomatis berdasarkan urutan rata-rata Monetary & Recency per cluster
    stats = rfm.groupby('Cluster').agg(Recency=('Recency', 'mean'), Monetary=('Monetary', 'mean')).reset_index()
    stats = stats.sort_values('Monetary', ascending=False).reset_index(drop=True)
    rank_to_label = {}
    labels_by_value_rank = ["Champions", "Promising/New Active", "At Risk", "Lost/Churned"]
    # Urutkan juga mempertimbangkan recency agar label lebih masuk akal
    stats_sorted = stats.sort_values(['Monetary', 'Recency'], ascending=[False, True]).reset_index(drop=True)
    for i, row in stats_sorted.iterrows():
        label = labels_by_value_rank[i] if i < len(labels_by_value_rank) else f"Segmen {i+1}"
        rank_to_label[row['Cluster']] = label

    rfm['Segmen'] = rfm['Cluster'].map(rank_to_label)
    return rfm, sil, db


def find_optimal_k(rfm: pd.DataFrame, k_range=range(2, 8)):
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(lambda x: np.log1p(x))
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        results.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette": silhouette_score(rfm_scaled, labels),
            "davies_bouldin": davies_bouldin_score(rfm_scaled, labels),
        })
    return pd.DataFrame(results)
