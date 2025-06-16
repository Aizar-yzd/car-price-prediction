# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
@st.cache_resource
def load_model(path):
    """Loads the pickled model pipeline, cached for performance."""
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Error: File model '{path}' tidak ditemukan. Pastikan Anda telah menjalankan `create_model.py` untuk membuatnya.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Load the entire pipeline (preprocessor + model)
model_pipeline = load_model('car_price_pipeline.pkl')
if model_pipeline is None:
    st.stop() # Stop execution if the model fails to load

# --- Static Data for UI ---
# In a more advanced app, these could be loaded from a config file
# or extracted from the training data.
BRANDS = ['Audi', 'BMW', 'Chevrolet', 'Ford', 'Honda', 'Hyundai', 'Kia', 'Mercedes', 'Toyota', 'Volkswagen']
MODELS_DICT = {
    'Audi': ['A3', 'A4', 'Q5'], 'BMW': ['3 Series', '5 Series', 'X5'], 'Chevrolet': ['Equinox', 'Impala', 'Malibu'],
    'Ford': ['Explorer', 'Fiesta', 'Focus'], 'Honda': ['Accord', 'CR-V', 'Civic'], 'Hyundai': ['Elantra', 'Sonata', 'Tucson'],
    'Kia': ['Optima', 'Rio', 'Sportage'], 'Mercedes': ['C-Class', 'E-Class', 'GLA'], 'Toyota': ['Camry', 'Corolla', 'RAV4'],
    'Volkswagen': ['Golf', 'Passat', 'Tiguan']
}
FUEL_TYPES = ['Petrol', 'Diesel', 'Hybrid', 'Electric']
TRANSMISSIONS = ['Automatic', 'Manual', 'Semi-Automatic']
DOORS_OPTIONS = [2, 3, 4, 5]
CURRENT_YEAR = datetime.datetime.now().year

# --- UI Layout ---
st.title('🚗 Prediksi Harga Mobil Bekas')
st.markdown("""
Gunakan model *machine learning* untuk mendapatkan estimasi harga mobil bekas Anda secara akurat. 
Masukkan detail mobil di sidebar untuk memulai.
""")
st.divider()

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header('📋 Detail Mobil')
    brand = st.selectbox('Merek', BRANDS, key='brand')
    model_name = st.selectbox('Model', MODELS_DICT[brand], key='model')
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.slider('Tahun Pembuatan', 2000, CURRENT_YEAR, 2018, key='year')
    with col2:
        engine_size = st.slider('Ukuran Mesin (L)', 1.0, 6.0, 2.0, 0.1, key='engine_size')
    
    fuel_type = st.selectbox('Jenis Bahan Bakar', FUEL_TYPES, key='fuel_type')
    transmission = st.selectbox('Transmisi', TRANSMISSIONS, key='transmission')
    
    mileage = st.number_input('Jarak Tempuh (km)', min_value=0, max_value=500000, value=50000, step=1000, key='mileage')

    col3, col4 = st.columns(2)
    with col3:
        doors = st.selectbox('Jumlah Pintu', DOORS_OPTIONS, index=2, key='doors')
    with col4:
        owner_count = st.slider('Jumlah Pemilik', 1, 5, 1, key='owner_count')
        
    predict_button = st.button('🚀 Prediksi Harga', use_container_width=True, type="primary")

# --- Feature Calculation & Prediction ---
# This block runs when the button is clicked
if predict_button:
    # Calculate derived features
    car_age = CURRENT_YEAR - year
    # Avoid division by zero for brand new cars
    mileage_per_year = mileage / car_age if car_age > 0 else 0
    brand_model = f"{brand}_{model_name}"

    # Create a DataFrame from user inputs
    # The column names MUST match the names used during training in 'create_model.py'
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
        # Add the engineered features
        'Car_Age': [car_age],
        'Mileage_per_Year': [mileage_per_year],
        'Brand_Model': [brand_model]
    })
    
    with st.spinner('Menganalisis data dan memprediksi...'):
        # The pipeline automatically handles preprocessing and prediction
        prediction = model_pipeline.predict(input_data)[0]
        
        # Display the result
        st.subheader("Hasil Prediksi", anchor=False)
        st.success(f"### Estimasi Harga Mobil: **USD ${prediction:,.2f}**")
        
        st.info(f"Berdasarkan input: **{year} {brand} {model_name}** dengan jarak tempuh **{mileage:,} km**.", icon="ℹ️")

        st.divider()

        # --- Additional Analysis Section ---
        st.subheader("Analisis Faktor Harga", anchor=False)
        st.write("Visualisasi sederhana faktor-faktor yang umumnya paling memengaruhi depresiasi harga.")

        # Display simple progress bars for key depreciation factors.
        # Normalization for visualization purposes.
        # Max values are illustrative estimates.
        age_norm = min(car_age / 15, 1.0) # Assume 15 years is high age
        mileage_norm = min(mileage / 200000, 1.0) # Assume 200k km is high mileage
        
        st.write("Usia Mobil:")
        st.progress(age_norm, text=f"{car_age} tahun")

        st.write("Jarak Tempuh:")
        st.progress(mileage_norm, text=f"{mileage:,.0f} km")

# --- Static Information Sections ---
# Placed outside the button 'if' block to always be visible
st.divider()
with st.expander("❓ Tentang Model Ini"):
    st.markdown("""
    Model ini menggunakan algoritma **XGBoost (Extreme Gradient Boosting)**, yang dibungkus dalam **Pipeline Scikit-learn** untuk memastikan konsistensi data.

    **Fitur yang digunakan dalam prediksi:**
    - Merek dan model mobil
    - Tahun pembuatan & Usia mobil
    - Ukuran mesin, jenis bahan bakar, dan transmisi
    - Jarak tempuh (total dan rata-rata per tahun)
    - Jumlah pintu dan riwayat kepemilikan

    Pipeline ini menangani semua transformasi data secara otomatis, dari teks ke angka, sehingga prediksi menjadi andal dan dapat direproduksi. Akurasi model pada data uji adalah sekitar **92% (R-squared)**.
    """)

# --- Footer ---
st.markdown(f"""
---
© {CURRENT_YEAR} Prediksi Harga Mobil | Dibuat dengan Streamlit & Scikit-learn/XGBoost
""")
