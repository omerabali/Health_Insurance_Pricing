# 🏥 InsurAI: Sağlık Sigortası Masraf Analizi & Tahmin Paneli

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white" />
</p>

---

## 📖 Proje Özeti
**InsurAI**, sigorta şirketlerinin poliçe fiyatlandırma süreçlerini yapay zeka ile modernize etmek için tasarlanmıştır. Bu sistem, uzmanların manuel hesaplamalar yerine; yaş, BMI ve yaşam tarzı (sigara) gibi verileri kullanarak **yıllık sağlık masrafını** yüksek doğrulukla tahmin etmesini sağlar.

---

## 🏗️ Teknik Mimari ve Model Performansı
Proje, veriyi işlemek ve tahmin üretmek için uçtan uca bir **Machine Learning Pipeline** kullanmaktadır.

### 🧠 Model Detayları
- **Algoritma:** Random Forest Regressor
- **Veri Ön İşleme:** Sayısal veriler için `StandardScaler`, kategorik veriler için `OneHotEncoder`.
- **Özellik Mühendisliği:** BMI ve Sigara kullanımı arasındaki korelasyonlar optimize edildi.

### 📊 Başarı Metrikleri
<div align="center">

| Metrik | Değer | Açıklama |
| :--- | :--- | :--- |
| **R2 Skoru** | **%85.2** | Modelin veriyi açıklama başarısı. |
| **MAE** | **4,150 $** | Tahminlerdeki ortalama mutlak hata. |
| **Algoritma** | **Random Forest** | 200 Karar ağacından oluşan topluluk modeli. |

</div>

---

## 🛠️ Özelliklerin Etki Oranı (Feature Importance)
Modelin karar verme sürecinde değişkenlerin ağırlığı şu şekildedir:

* **Sigara Kullanımı:** %62 (En kritik faktör)
* **Vücut Kitle İndeksi (BMI):** %18
* **Yaş:** %14
* **Diğer (Çocuk sayısı, Bölge):** %6



---

## 🎨 Arayüz Özellikleri
> [!TIP]
> **Kullanıcı Paneli Neler Sunar?**
> - **Anlık Tahmin:** Slider ve inputlar değiştikçe masraf tahmini dinamik olarak güncellenir.
> - **Kıyaslama Grafiği:** Tahmin edilen tutar, sigara içen ve içmeyenlerin genel ortalamasıyla anlık kıyaslanır.
> - **Analitik Sekmesi:** Modelin güven skoru ve hata oranları şeffaf bir şekilde dashboard üzerinden paylaşılır.

---

## 📂 Dosya Yapısı
- `app.py`: Streamlit tabanlı kullanıcı arayüzü ve dashboard kodu.
- `model_egit.py`: Modelin eğitimi, Pipeline kurulumu ve `.pkl` kaydı.
- `insurance.csv`: 1338 satırlık ham sigorta veri seti.
- `insurance_ai_model.pkl`: Eğitilmiş hazır model dosyası.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Kütüphaneleri Yükleyin
Projenin çalışması için gerekli olan Python kütüphanelerini terminal üzerinden yükleyin:
```bash
pip install streamlit pandas numpy joblib scikit-learn plotly
