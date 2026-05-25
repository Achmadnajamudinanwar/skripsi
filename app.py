import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model

# =========================
# CONFIG
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
    return load_model("best_model.h5", compile=False)

model = load_cnn_model()

# =========================
# CLASS
# =========================

class_names = [
    "komersial",
    "pemukiman",
    "ruang_terbuka_hijau"
]

# =========================
# TITLE
# =========================

st.title("Klasifikasi Citra Wilayah")

st.write(
    "Upload gambar untuk klasifikasi wilayah perkotaan"
)

# =========================
# UPLOAD
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

    st.image(image, caption="Gambar Upload")

    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    confidence = np.max(prediction)

    predicted_class = class_names[np.argmax(prediction)]

    st.success(
        f"Hasil: {predicted_class}"
    )

    st.info(
        f"Akurasi: {confidence*100:.2f}%"
    )