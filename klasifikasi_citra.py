import numpy as np
import tensorflow as tf
import streamlit as st
import requests
from base64 import b64encode
from io import BytesIO

# CSS untuk menambahkan background image
def add_bg_from_local(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{image_file}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi untuk mengonversi gambar menjadi base64
def get_image_as_base64(image_path):
    with open(image_path, "rb") as file:
        return b64encode(file.read()).decode()

# Fungsi untuk mengunduh gambar dari URL dan mengubahnya menjadi base64
def get_image_as_base64_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return b64encode(response.content).decode()
    else:
        raise FileNotFoundError(f"Unable to fetch image from {image_url}")

# URL gambar background (gambar di GitHub)
bg_image_url = "https://raw.githubusercontent.com/zaidannn/Zaidan-UAP/main/Images/download%20(6).jpg"
bg_image_base64 = get_image_as_base64_from_url(bg_image_url)
add_bg_from_local(bg_image_base64)

# Judul aplikasi
st.title("Klasifikasi Citra Sayuran")

# Fungsi prediksi
def predict(uploaded_image):
    # Daftar kelas
    class_names = ["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage", "Capsicum", "Carrot",
                   "Cauliflower", "Cucumber", "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"]

    # Muat dan preprocess citra
    img = tf.keras.utils.load_img(uploaded_image, target_size=(224, 224))  # Pastikan ukuran sesuai dengan model
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch

    # Unduh model dari GitHub
    model_url = "https://github.com/zaidannn/Zaidan-UAP/raw/main/Model/Image/vegetable_classifier.h5"
    response = requests.get(model_url)
    
    # Cek jika permintaan sukses
    if response.status_code == 200:
        model = tf.keras.models.load_model(BytesIO(response.content))
    else:
        raise FileNotFoundError(f"Model tidak dapat ditemukan di {model_url}")

    # Prediksi
    output = model.predict(img)
    score = tf.nn.softmax(output[0])  # Hitung probabilitas
    return class_names[np.argmax(score)], 100 * np.max(score)  # Prediksi label dan confidence

# Komponen file uploader untuk banyak file
uploads = st.file_uploader("Unggah citra untuk mendapatkan hasil prediksi", type=["png", "jpg"], accept_multiple_files=True)

# Tombol prediksi
if st.button("Predict", type="primary"):
    if uploads:
        st.subheader("Hasil prediksi:")

        for upload in uploads:
            # Tampilkan setiap citra yang diunggah
            st.image(upload, caption=f"Citra yang diunggah: {upload.name}", use_column_width=True)

            with st.spinner(f"Memproses citra {upload.name} untuk prediksi..."):
                # Panggil fungsi prediksi
                try:
                    label, confidence = predict(upload)
                    st.write(f"**{upload.name}** - Label: **{label}**, Confidence: **{confidence:.5f}%**")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses {upload.name}: {e}")
    else:
        st.error("Unggah setidaknya satu citra terlebih dahulu!")
