import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd
import os
import sys
import streamlit.web.cli as stcli

# =========================================================
# AUTO RUN STREAMLIT
# =========================================================

if __name__ == "__main__" and st.runtime.exists() == False:

    sys.argv = [
        "streamlit",
        "run",
        os.path.abspath(__file__)
    ]

    sys.exit(stcli.main())

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

model = load_model("best_model.keras")

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

/* SIDEBAR */

section[data-testid="stSidebar"]{
    background-color:#16213e;
}

section[data-testid="stSidebar"] *{
    color:white;
}

/* TITLE */

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

/* CARD */

.card{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.1);
    margin-top:20px;
}

/* RESULT */

.result-box{
    padding:30px;
    border-radius:20px;
    color:white;
    text-align:center;
    font-weight:bold;
    box-shadow:0px 4px 15px rgba(0,0,0,0.2);
}

/* IMAGE */

img{
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR MENU
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

    <p>
    Sistem mampu mengidentifikasi tiga kategori utama yaitu:
    </p>

    <ul>
        <li><b>Komersial</b> → wilayah gedung perkantoran dan pusat bisnis</li>
        <li><b>Pemukiman</b> → kawasan rumah dan bangunan padat penduduk</li>
        <li><b>Ruang Terbuka Hijau</b> → area hijau seperti taman dan vegetasi</li>
    </ul>

    <p>
    Sistem dikembangkan untuk membantu identifikasi wilayah
    perkotaan serta mendukung analisis kawasan rawan kebakaran
    berdasarkan karakteristik penggunaan lahan.
    </p>

    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        <div class="card" style="text-align:center;">

        <h1>🧠</h1>

        <h3>Deep Learning</h3>

        <p>
        Menggunakan metode CNN untuk klasifikasi otomatis.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card" style="text-align:center;">

        <h1>🚀</h1>

        <h3>EfficientNet-B2</h3>

        <p>
        Arsitektur modern dengan performa klasifikasi yang baik.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown("""
        <div class="card" style="text-align:center;">

        <h1>📊</h1>

        <h3>Analisis Citra</h3>

        <p>
        Menampilkan probabilitas dan hasil klasifikasi citra.
        </p>

        </div>
        """, unsafe_allow_html=True)

# =========================================================
# HALAMAN PENGUJIAN
# =========================================================

elif menu == "Pengujian":

    st.markdown("""
    <div class="main-title">
    HALAMAN PENGUJIAN
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">

    <h2>Cara Penggunaan</h2>

    <ol>
        <li>Upload gambar citra wilayah</li>
        <li>Sistem melakukan proses klasifikasi</li>
        <li>Hasil analisis akan ditampilkan</li>
    </ol>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("## 📥 Input Citra")

        uploaded_file = st.file_uploader(
            "Upload Gambar",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            if image.mode != "RGB":
                image = image.convert("RGB")

            st.image(
                image,
                caption="Gambar Upload",
                use_container_width=True
            )

    with col2:

        st.markdown("## 📈 Hasil Analisis")

        if uploaded_file is not None:

            # =========================================================
            # PREPROCESSING
            # =========================================================

            img = image.resize((224,224))

            img_array = np.array(img)

            img_array = img_array / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            # =========================================================
            # PREDIKSI
            # =========================================================

            prediction = model.predict(img_array)

            probabilities = prediction[0]

            confidence = np.max(probabilities)

            predicted_class = class_names[np.argmax(probabilities)]

            # =========================================================
            # FILTER GAMBAR DI LUAR DATASET
            # =========================================================

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

            # =========================================================
            # HASIL ANALISIS
            # =========================================================

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

            # =========================================================
            # CONFIDENCE BAR
            # =========================================================

            st.subheader("Confidence")

            st.progress(float(confidence))

            # =========================================================
            # DETAIL PROBABILITAS
            # =========================================================

            st.subheader("Probabilitas Semua Kelas")

            for i, label in enumerate(class_names):

                st.write(
                    f"{label} : {probabilities[i]*100:.2f}%"
                )

            # =========================================================
            # GRAFIK
            # =========================================================

            st.subheader("Grafik Probabilitas")

            chart_data = pd.DataFrame({
                "Kategori": class_names,
                "Probabilitas": probabilities
            })

            st.bar_chart(
                chart_data.set_index("Kategori")
            )

        else:

            st.info("Upload gambar terlebih dahulu")

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

    <img src="https://cdn-icons-png.flaticon.com/512/149/149071.png"
    width="150">

    <h1>Achmad Najamudin Anwar 535190047</h1>

    <p>
    Mahasiswa Program Studi Informatika yang mengembangkan
    sistem klasifikasi citra wilayah perkotaan menggunakan
    metode Convolutional Neural Network (CNN)
    dengan arsitektur EfficientNet-B2.
    </p>

    <br>

    <p>
    Sistem dikembangkan sebagai implementasi Deep Learning
    dalam bidang pengolahan citra digital untuk membantu
    identifikasi wilayah perkotaan dan kawasan rawan kebakaran
    secara otomatis.
    </p>

    </div>
    """, unsafe_allow_html=True)