# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Memuat Model yang Sudah Ada ---
# Model ini diasumsikan sudah dilatih dengan data yang sudah di-preprocess (One-Hot Encoded)
@st.cache_resource
def load_model(path):
    """Memuat model pickle dari path yang diberikan, dengan caching."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: File model '{path}' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

model = load_model('car_price_xgboost_model.pkl')
if model is None:
    st.stop()  # Hentikan eksekusi jika model gagal dimuat

# --- Data untuk Dropdown UI ---
# PENTING: Daftar ini harus sama persis dengan kategori yang digunakan saat melatih model
brands = ['Audi', 'BMW', 'Chevrolet', 'Ford', 'Honda', 'Hyundai', 'Kia', 
          'Mercedes', 'Toyota', 'Volkswagen']
models_dict = {
    'Audi': ['A3', 'A4', 'Q5'], 'BMW': ['3 Series', '5 Series', 'X5'], 'Chevrolet': ['Equinox', 'Impala', 'Malibu'],
    'Ford': ['Explorer', 'Fiesta', 'Focus'], 'Honda': ['Accord', 'CR-V', 'Civic'], 'Hyundai': ['Elantra', 'Sonata', 'Tucson'],
    'Kia': ['Optima', 'Rio', 'Sportage'], 'Mercedes': ['C-Class', 'E-Class', 'GLA'], 'Toyota': ['Camry', 'Corolla', 'RAV4'],
    'Volkswagen': ['Golf', 'Passat', 'Tiguan']
}
fuel_types = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
transmissions = ['Automatic', 'Manual', 'Semi-Automatic']
doors_options = [2, 3, 4, 5]
CURRENT_YEAR = datetime.datetime.now().year

# --- UI Header ---
st.title('🚗 Prediksi Harga Mobil')
st.markdown("""
**Prediksikan harga mobil bekas Anda dengan akurat menggunakan model machine learning XGBoost.**
Masukkan detail mobil di sidebar untuk mendapatkan estimasi harga.
""")

# --- Sidebar untuk Input Pengguna ---
with st.sidebar:
    st.header('📋 Detail Mobil')
    st.subheader('Informasi Umum')
    
    brand = st.selectbox('Merek', brands, key='brand')
    model_name = st.selectbox('Model', models_dict[brand], key='model_name')
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.slider('Tahun Pembuatan', 2000, CURRENT_YEAR, 2018)
    with col2:
        engine_size = st.slider('Ukuran Mesin (L)', 1.0, 5.0, 2.0, 0.1)
    
    col3, col4 = st.columns(2)
    with col3:
        fuel_type = st.selectbox('Jenis Bahan Bakar', fuel_types)
    with col4:
        transmission = st.selectbox('Transmisi', transmissions)
    
    mileage = st.number_input('Jarak Tempuh (km)', min_value=0, max_value=300000, value=50000, step=1000)
    
    col5, col6 = st.columns(2)
    with col5:
        doors = st.selectbox('Jumlah Pintu', doors_options, index=2)
    with col6:
        owner_count = st.slider('Jumlah Pemilik Sebelumnya', 1, 5, 1)

# --- Logika Prediksi ---
if st.button('🚀 Prediksi Harga', use_container_width=True, type="primary"):
    # 1. Hitung fitur tambahan dari input pengguna
    car_age = CURRENT_YEAR - year
    mileage_per_year = mileage / car_age if car_age > 0 else 0
    brand_model = f"{brand}_{model_name}"

    # 2. Buat DataFrame dari input pengguna (1 baris)
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Model': [model_name],
        'Year': [year],
        'Engine_Size': [engine_size],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Mileage': [mileage],
        'Doors': [doors],
        'Owner_Count': [owner_count],
        'Car_Age': [car_age],
        'Mileage_per_Year': [mileage_per_year],
        'Brand_Model': [brand_model]
    })

    # --- BAGIAN KRITIS: Pra-pemrosesan Input ---
    # Model Anda mengharapkan kolom numerik (bukan teks). Kita harus mengubah input
    # pengguna agar cocok dengan format data saat training.

    st.info("Memproses input agar sesuai dengan format model...", icon="⚙️")
    
    # Definisikan fitur kategorikal dan numerik (harus sama dengan saat training)
    categorical_features = ['Brand', 'Model', 'Fuel_Type', 'Transmission', 'Brand_Model']
    numerical_features = ['Year', 'Engine_Size', 'Mileage', 'Doors', 'Owner_Count', 'Car_Age', 'Mileage_per_Year']

    # Lakukan One-Hot Encoding pada input pengguna
    input_encoded = pd.get_dummies(input_data, columns=categorical_features)
    
    # Buat daftar SEMUA kemungkinan kolom setelah encoding, berdasarkan opsi di UI
    # Ini untuk memastikan input memiliki semua kolom yang diharapkan model.
    all_models = [f"{b}_{m}" for b in brands for m in models_dict[b]]
    
    expected_cols = numerical_features + \
                    [f'Brand_{b}' for b in brands] + \
                    [f'Model_{m}' for b in brands for m in models_dict[b]] + \
                    [f'Fuel_Type_{f}' for f in fuel_types] + \
                    [f'Transmission_{t}' for t in transmissions] + \
                    [f'Brand_Model_{bm}' for bm in all_models]
    
    # Hapus duplikat kolom jika ada
    expected_cols = sorted(list(set(expected_cols)))

    # Reindex input yang sudah di-encode agar memiliki kolom yang sama persis dengan model
    # Kolom yang tidak ada di input pengguna akan diisi dengan 0.
    final_input = input_encoded.reindex(columns=expected_cols, fill_value=0)
    
    # Pastikan urutan kolom sama seperti saat training (opsional tapi praktik terbaik)
    # Jika Anda tidak tahu urutan pastinya, reindex di atas sudah cukup 99%
    # final_input = final_input[training_column_order] # <-- Jika Anda punya daftar urutan kolom training
    
    # 3. Lakukan prediksi
    with st.spinner('Memprediksi harga...'):
        prediction = model.predict(final_input)[0]

    # 4. Tampilkan hasil
    st.success(f"### Estimasi Harga Mobil: **${prediction:,.2f}**")

    # Tampilkan metrik fitur tambahan
    st.subheader('📊 Detail Fitur yang Digunakan')
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Usia Mobil", f"{car_age} tahun")
    with col2:
        st.metric("Rata-rata Jarak Tempuh per Tahun", f"{mileage_per_year:,.0f} km/tahun")


# --- Informasi Tambahan dan Footer ---
st.markdown("---")
with st.expander("❓ Tentang Model Prediksi Ini"):
    st.markdown("""
    Model ini menggunakan algoritma **XGBoost** yang telah dilatih dengan data ribuan mobil bekas. 
    Akurasi model sangat bergantung pada kualitas data dan kesamaan fitur input dengan data saat pelatihan.

    **Penting:** Aplikasi ini melakukan pra-pemrosesan data (One-Hot Encoding) pada input Anda agar formatnya sesuai dengan yang diharapkan oleh model yang telah dimuat. Pastikan daftar opsi (merek, model, dll.) di aplikasi ini cocok dengan data yang digunakan untuk melatih model Anda.
    """)

st.markdown(f"""
---
© {CURRENT_YEAR} Prediksi Harga Mobil | Dibuat dengan Streamlit dan XGBoost
""")
