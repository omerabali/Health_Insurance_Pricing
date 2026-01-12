import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

st.set_page_config(
    page_title="InsurAI | Sigorta Masraf Analizi",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .prediction-box {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model_path = "insurance_ai_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()


with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3408/3408591.png", width=100)
    st.title("Müşteri Bilgileri")
    st.info("Lütfen tahmin için detayları giriniz.")
    
    age = st.slider("Yaş", 18, 90, 30)
    sex = st.radio("Cinsiyet", ["male", "female"], horizontal=True)
    bmi = st.slider("Vücut Kitle İndeksi (BMI)", 10.0, 50.0, 25.0)
    children = st.number_input("Çocuk Sayısı", 0, 10, 0)
    smoker = st.selectbox("Sigara Kullanımı", ["yes", "no"])
    region = st.selectbox("Bölge", ["southeast", "southwest", "northeast", "northwest"])
    
    predict_btn = st.button("💰 HESAPLA")


st.title("🏥 InsurAI - Sağlık Sigortası Tahmin Paneli")

tab1, tab2 = st.tabs(["📊 Tahmin Ekranı", "🔎 Model Analitiği"])

with tab1:
    if predict_btn:
        if model is not None:
           
            input_df = pd.DataFrame({
                "age": [age], "sex": [sex], "bmi": [bmi],
                "children": [children], "smoker": [smoker], "region": [region]
            })

            # Tahmin
            prediction = model.predict(input_df)[0]

            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Tahmini Yıllık Masraf</h3>
                    <h1 style='color: #28a745;'>$ {prediction:,.2f}</h1>
                    <p>Müşteri Profili: <b>{age} yaşında, {"Sigara içen" if smoker=="yes" else "Sigara içmeyen"}</b></p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
               
                comparison_data = pd.DataFrame({
                    "Kategori": ["Sizin Tahmininiz", "Ortalama (Sigara İçmeyen)", "Ortalama (Sigara İçen)"],
                    "Tutar": [prediction, 8434, 32050]
                })
                fig = px.bar(comparison_data, x="Kategori", y="Tutar", color="Kategori", title="Piyasa Kıyaslaması")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Model dosyası bulunamadı! Lütfen önce eğitim kodunu çalıştırın.")
    else:
        st.write("### Hoş Geldiniz!")
        st.write("Sol menüden müşteri bilgilerini girerek 'Hesapla' butonuna basınız.")
        st.image("https://i.imgur.com/30999.png") # Buraya şık bir görsel eklenebilir

with tab2:
    st.subheader("Model Performans Verileri")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Model Güven Skoru (R2)", "%85.2")
    col_m2.metric("Ortalama Hata (MAE)", "4,150 $")
    col_m3.metric("Kullanılan Algoritma", "Random Forest")
    
    st.divider()
    st.write("#### Değişkenlerin Etki Oranı")
  
    importance_df = pd.DataFrame({
        "Özellik": ["Sigara", "BMI", "Yaş", "Çocuklar", "Bölge"],
        "Etki Skoru": [0.62, 0.18, 0.14, 0.04, 0.02]
    })
    fig_importance = px.pie(importance_df, values='Etki Skoru', names='Özellik', hole=.3)
    st.plotly_chart(fig_importance)