import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS (TEMA TERANG)
# ==========================================
st.set_page_config(
    page_title="Dashboard Biofarmaka Jabar",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 1. BACKGROUND UTAMA (Menjadi Putih/Terang) */
    .stApp { 
        background-color: #F8F9FA; 
        color: #31333F; 
    }

    /* 2. SIDEBAR (Tetap Hijau Tua agar kontras dan elegan) */
    [data-testid="stSidebar"] { 
        background-color: #004d40 !important; 
        border-right: 2px solid #00796b;
    }

    /* 3. PENGATURAN WARNA TEKS */
    /* Teks di halaman utama jadi Hitam/Gelap */
    h1, h2, h3, h4, h5, h6, p, li, div { 
        color: #31333F; 
    }
    
    /* KHUSUS Teks di Sidebar harus Putih */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }

    /* 4. STYLING INPUT/SELECTBOX */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #4CAF50 !important;
        color: #000000 !important;
    }
    
    /* 5. STYLING KARTU CLUSTER */
    .cluster-card {
        padding: 20px; border-radius: 10px; margin-bottom: 15px; color: white !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    /* Pastikan teks dalam kartu tetap putih */
    .cluster-card div, .cluster-card .card-header {
        color: #FFFFFF !important;
    }
    
    .card-0 { background: linear-gradient(135deg, #1565C0, #42A5F5); border-left: 5px solid #0D47A1; }
    .card-1 { background: linear-gradient(135deg, #AD1457, #EC407A); border-left: 5px solid #880E4F; }
    .card-2 { background: linear-gradient(135deg, #EF6C00, #FF9800); border-left: 5px solid #E65100; }
    .card-default { background: linear-gradient(135deg, #424242, #616161); border-left: 5px solid #212121; }

</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA & CLEANING
# ==========================================
def clean_indo_number(value):
    if pd.isna(value): return 0.0
    str_val = str(value).strip()
    for char in ['Rp', 'rp', ' ', 'kg', 'm2', '%', 'ton']:
        str_val = str_val.replace(char, '')
    try:
        if '.' in str_val and ',' in str_val: str_val = str_val.replace('.', '').replace(',', '.')
        elif '.' in str_val:
             if len(str_val.split('.')[-1]) == 3: str_val = str_val.replace('.', '')
        elif ',' in str_val: str_val = str_val.replace(',', '.')
        return float(str_val)
    except: return 0.0

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset_final.csv')
        if len(df.columns) < 2: 
            df = pd.read_csv('dataset_final.csv', sep=';')
    except Exception as e:
        st.error(f"Gagal memuat file: {e}")
        return pd.DataFrame(), []

    df_clean = df.copy()

    first_col = df_clean.columns[0]
    df_clean.rename(columns={first_col: 'Wilayah'}, inplace=True)

    df_clean['Wilayah'] = df_clean['Wilayah'].astype(str)
    filter_kata = ['catatan', 'angka sementara', 'angka tetap', 'sumber', 'keterangan']
    pattern = '|'.join(filter_kata)
    df_clean = df_clean[~df_clean['Wilayah'].str.contains(pattern, case=False, na=False)]
    df_clean = df_clean[df_clean['Wilayah'].str.strip() != ""]

    new_cols = {}
    for col in df_clean.columns:
        c_clean = col
        for term in filter_kata:
            c_clean = re.sub(term, '', c_clean, flags=re.IGNORECASE)
        new_cols[col] = " ".join(c_clean.split()).strip()
    
    df_clean.rename(columns=new_cols, inplace=True)

    cols_prod = [c for c in df_clean.columns if 'produksi' in c.lower() and 'total' not in c.lower()]

    for c in df_clean.columns:
        if c != 'Wilayah':
            df_clean[c] = df_clean[c].apply(clean_indo_number)
            
    if 'Total_Produksi' not in df_clean.columns and cols_prod:
        df_clean['Total_Produksi'] = df_clean[cols_prod].sum(axis=1)
    
    if 'Total_Luas' not in df_clean.columns:
         c_luas = [c for c in df_clean.columns if 'luas' in c.lower() or 'panen' in c.lower()]
         if c_luas: df_clean['Total_Luas'] = df_clean[c_luas].sum(axis=1)
         else: df_clean['Total_Luas'] = 0

    if 'Tahun' not in df_clean.columns: 
        df_clean['Tahun'] = 2024
        
    return df_clean, cols_prod

df, prod_cols = load_data()

# ==========================================
# 3. SIDEBAR & PROSES CLUSTERING
# ==========================================
with st.sidebar:
    st.title("📊 Menu Analisis")
    selected_page = st.radio("Pilih Analisis:", ["Dashboard", "EDA & Visualisasi", "Hasil Clustering", "Dataset"])
    
    st.markdown("---")
    st.subheader("⚙️ Parameter K-Means")
    
    if not df.empty:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'Tahun' in num_cols: num_cols.remove('Tahun')
        feats = st.multiselect("Pilih Fitur Clustering:", num_cols, default=['Total_Produksi', 'Total_Luas'])
        k_val = st.slider("Jumlah Cluster (K):", min_value=2, max_value=8, value=3)
    else:
        st.stop()

# --- PROSES CLUSTERING ---
if len(feats) >= 2:
    X = df[feats].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    df['Cluster'] = model.fit_predict(X_scaled).astype(str)
    
    cluster_means = df.groupby('Cluster')[feats[0]].mean().sort_values(ascending=False)
    cat_names = ["Potensi Tinggi", "Potensi Sedang", "Perlu Optimalisasi", "Potensi Rendah"]
    
    labels_map = {}
    for i, (cluster, _) in enumerate(cluster_means.items()):
        labels_map[cluster] = cat_names[i] if i < len(cat_names) else f"Kelompok {i+1}"
        
    df['Kategori'] = df['Cluster'].map(labels_map)
    sorted_clusters = sorted(df['Cluster'].unique())
    colors_list = ['card-0', 'card-1', 'card-2']
else:
    if selected_page in ["Dashboard", "Hasil Clustering"]:
        st.warning("⚠️ Silakan pilih minimal 2 fitur di sidebar untuk melakukan analisis.")
        st.stop()
    sorted_clusters = []
    labels_map = {}

# ==========================================
# 4. LOGIKA HALAMAN
# ==========================================

# --- HALAMAN 1: DASHBOARD ---
if selected_page == "Dashboard":
    st.title("🖥️ DASHBOARD BIOFARMAKA")
    st.markdown("---")
    
    if not sorted_clusters:
        st.info("Data belum dicluster. Pilih fitur di sidebar.")
    else:
        cols = st.columns(3)
        for i, c in enumerate(sorted_clusters[:3]):
            with cols[i]:
                stats = df[df['Cluster'] == c]
                bg_class = colors_list[i] if i < len(colors_list) else 'card-default'
                
                st.markdown(f"""
                <div class="cluster-card {bg_class}">
                    <div class="card-header">CLUSTER {c}</div>
                    <div style="font-size: 1.2em; font-weight:bold;">{labels_map.get(c, c)}</div>
                    <hr style="border-color: rgba(255,255,255,0.3); margin: 10px 0;">
                    <div class="card-stat-value">{len(stats)} Wilayah</div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"📍 Lihat Wilayah Cluster {c}"):
                    st.dataframe(stats[['Wilayah', 'Total_Produksi']], hide_index=True, use_container_width=True)

# --- HALAMAN 2: EDA ---
elif selected_page == "EDA & Visualisasi":
    st.title("📈 EXPLORATORY DATA ANALYSIS")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribusi Total Produksi")
        # Menggunakan template 'seaborn' agar cocok dengan background putih
        fig_hist = px.histogram(df, x="Total_Produksi", template="seaborn", color_discrete_sequence=['#00796b'])
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        st.subheader("Tren Tahunan (Rata-rata)")
        trend_df = df.groupby('Tahun')['Total_Produksi'].mean().reset_index()
        fig_trend = px.line(trend_df, x='Tahun', y='Total_Produksi', markers=True, template="seaborn", color_discrete_sequence=['#d32f2f'])
        st.plotly_chart(fig_trend, use_container_width=True)

# --- HALAMAN 3: CLUSTERING ---
elif selected_page == "Hasil Clustering":
    st.title("🎯 DETAIL ANALISIS CLUSTER")
    
    tab1, tab2 = st.tabs(["📊 Plot Clustering", "🔍 Detail Wilayah & Tanaman"])
    
    with tab1:
        st.subheader("Visualisasi Scatter Plot K-Means")
        if len(feats) >= 2:
            fig_scatter = px.scatter(
                df, x=feats[0], y=feats[1], 
                color='Cluster', symbol='Cluster',
                hover_data=['Wilayah', 'Kategori'], 
                template="seaborn", # Ubah ke seaborn (tema terang)
                title=f"Sebaran Cluster berdasarkan {feats[0]} vs {feats[1]}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Pilih minimal 2 fitur di sidebar untuk melihat plot.")

    with tab2:
        st.subheader("🔍 Cek Wilayah Spesifik")
        
        clusters_available = sorted(df['Cluster'].unique())
        sel_clust = st.selectbox("1. Pilih Cluster:", clusters_available, key="clust_final")
        
        d_clust = df[df['Cluster'].astype(str) == str(sel_clust)]
        
        regions_available = d_clust['Wilayah'].unique()
        sel_reg = st.selectbox("2. Pilih Wilayah:", regions_available, key="reg_final")
        
        if sel_reg:
            reg_row = d_clust[d_clust['Wilayah'] == sel_reg].iloc[0]
            st.info(f"Detail Produksi Tanaman di **{sel_reg}** (Cluster {sel_clust})")
            
            plant_data = reg_row[prod_cols].reset_index()
            plant_data.columns = ['Tanaman', 'Produksi (kg)']
            
            bad_words = ['produksi', 'Angka Sementara', 'Angka Tetap', 'Catatan', 'angka sementara', 'angka tetap']
            for word in bad_words:
                plant_data['Tanaman'] = plant_data['Tanaman'].str.replace(word, '', case=False).str.strip()
            
            plant_data = plant_data[plant_data['Produksi (kg)'] > 0].sort_values('Produksi (kg)', ascending=False)
            
            st.dataframe(plant_data.style.format({"Produksi (kg)": "{:,.0f} kg"}), use_container_width=True)

# --- HALAMAN 4: DATASET ---
elif selected_page == "Dataset":
    st.title("📂 DATASET UTAMA")
    st.markdown("---")
    
    st.write(f"**Dimensi Data:** {df.shape[0]} Baris x {df.shape[1]} Kolom")
    st.dataframe(df, use_container_width=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data CSV",
        data=csv,
        file_name='dataset_bersih.csv',
        mime='text/csv',
    )