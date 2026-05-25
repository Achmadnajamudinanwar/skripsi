import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import pandas as pd

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="Klasifikasi Wilayah",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_cnn_model():
    model = load_model(
        "best_model.keras",
        compile=False,
        safe_mode=False
    )
    return model

model = load_cnn_model()

# =========================================================
# LABEL KELAS
# =========================================================

class_names = [
    "komersial",
    "pemukiman",
    "ruang_terbuka_hijau"
]

# =========================================================
# CSS TAMPILAN
# =========================================================

st.markdown("""
<style>

.stApp{
    background-color:#eaf9f9;
}

section[data-testid="stSidebar"]{
    background-color:#16213e;
}

section[data-testid="stSidebar"] *{
    color:white;
}

.main-title{
    background: linear-gradient(90deg,#4facfe,#00f2fe);
    padding:25px;
    border-radius:15px;
    text-align:center;
    color:white;
    font-size:38px;
    font-weight:bold;
    box-shadow:0px 4px 15px rgba(0,0,0,0.2);
}

.card{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
    margin-top:20px;
}

.result-box{
    padding:30px;
    border-radius:20px;
    color:white;
    text-align:center;
    font-weight:bold;
    box-shadow:0px 4px 15px rgba(0,0,0,0.2);
}

img{
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

menu = st.sidebar.radio(
    "Menu",
    [
        "Halaman Utama",
        "Pengujian",
        "Info Pembuat"
    ]
)

# =========================================================
# HALAMAN UTAMA
# =========================================================

if menu == "Halaman Utama":

    st.markdown("""
    <div class="main-title">
    KLASIFIKASI CITRA WILAYAH PERKOTAAN <br>
    MENGGUNAKAN CONVOLUTIONAL NEURAL NETWORK
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">

    <h2>Tentang Sistem</h2>

    <p>
    Sistem ini merupakan implementasi Deep Learning menggunakan
    metode Convolutional Neural Network (CNN) dengan arsitektur
    EfficientNet-B2 untuk melakukan klasifikasi citra wilayah
    perkotaan secara otomatis.
    </p>

    <ul>
        <li><b>Komersial</b></li>
        <li><b>Pemukiman</b></li>
        <li><b>Ruang Terbuka Hijau</b></li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PENGUJIAN
# =========================================================

elif menu == "Pengujian":

    st.markdown("""
    <div class="main-title">
    HALAMAN PENGUJIAN
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Gambar Upload",
                use_container_width=True
            )

        with col2:

            img = image.resize((224,224))

            img_array = np.array(img) / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)

            probabilities = prediction[0]

            confidence = np.max(probabilities)

            predicted_class = class_names[np.argmax(probabilities)]

            sorted_probs = np.sort(probabilities)

            selisih = sorted_probs[-1] - sorted_probs[-2]

            if confidence < 0.85 or selisih < 0.30:

                hasil = "GAMBAR TIDAK DIKENALI"
                color = "#ef4444"
                risiko = "DATA DI LUAR DATASET"

            else:

                hasil = predicted_class.upper()

                if predicted_class == "komersial":
                    color = "#facc15"
                    risiko = "SEDANG ⚠️"

                elif predicted_class == "pemukiman":
                    color = "#fb923c"
                    risiko = "TINGGI 🔥"

                else:
                    color = "#22c55e"
                    risiko = "RENDAH 🌳"

            st.markdown(
                f"""
                <div class="result-box"
                style="background:{color};">

                <h1>{hasil}</h1>

                <h2>
                Akurasi : {confidence*100:.2f}%
                </h2>

                <h2>
                Risiko : {risiko}
                </h2>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Confidence")

            st.progress(float(confidence))

            st.subheader("Probabilitas Semua Kelas")

            for i, label in enumerate(class_names):

                st.write(
                    f"{label} : {probabilities[i]*100:.2f}%"
                )

            st.subheader("Grafik Probabilitas")

            chart_data = pd.DataFrame({
                "Kategori": class_names,
                "Probabilitas": probabilities
            })

            st.bar_chart(
                chart_data.set_index("Kategori")
            )

# =========================================================
# INFO PEMBUAT
# =========================================================

elif menu == "Info Pembuat":

    st.markdown("""
    <div class="main-title">
    INFO PEMBUAT
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="text-align:center;">

    <h1>Achmad Najamudin Anwar</h1>

    <p>
    Sistem klasifikasi citra wilayah perkotaan
    menggunakan metode CNN EfficientNet-B2.
    </p>

    </div>
    """, unsafe_allow_html=True)