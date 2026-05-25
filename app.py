import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_cnn_model():

    model = load_model(
        "best_model.h5",
        compile=False
    )

    return model

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
# JUDUL
# =========================

st.title("Klasifikasi Wilayah Perkotaan")

st.write(
    "Upload gambar untuk diprediksi"
)

# =========================
# UPLOAD GAMBAR
# =========================

uploaded_file = st.file_uploader(
    "Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Gambar Upload",
        use_column_width=True
    )

    img = image.resize((224, 224))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = np.max(prediction)

    st.success(
        f"Hasil Prediksi: {predicted_class}"
    )

    st.write(
        f"Confidence: {confidence:.2f}"
    )