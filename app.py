import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# =========================
# KONFIGURASI HALAMAN
# =========================

st.set_page_config(
    page_title="Klasifikasi Wilayah",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_cnn_model():
    model = load_model("model.h5", compile=False)
    return model

model = load_cnn_model()

# =========================
# LABEL KELAS
# =========================

class_names = [
    "Komersial",
    "Pemukiman",
    "Ruang Terbuka Hijau"
]

# =========================
# TAMPILAN
# =========================

st.title("Klasifikasi Wilayah Perkotaan")

st.write(
    "Upload gambar wilayah untuk diprediksi menggunakan model CNN."
)

# =========================
# UPLOAD GAMBAR
# =========================

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDIKSI
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diupload",
        use_column_width=True
    )

    # Resize gambar
    image = image.resize((224, 224))

    # Ubah ke array
    img_array = np.array(image)

    # Normalisasi
    img_array = img_array / 255.0

    # Tambah dimensi
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # =========================
    # HASIL
    # =========================

    st.success(
        f"Hasil Prediksi: {predicted_class}"
    )

    st.info(
        f"Tingkat Keyakinan: {confidence:.2f}%"
    )