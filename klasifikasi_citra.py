import numpy as np
import tensorflow as tf
from pathlib import Path
import streamlit as st
import requests
from base64 import b64encode

# CSS untuk menambahkan background image dan tata letak
def add_bg_from_local(image_file):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{image_file}");
            background-size: cover;  /* Menyesuaikan gambar dengan ukuran layar */
            background-position: center center;  /* Menempatkan gambar di tengah */
            background-repeat: no-repeat;  /* Gambar tidak akan terulang */
            background-attachment: fixed;  /* Agar gambar tetap saat scroll */
        }}

        /* Gaya untuk judul aplikasi */
        h1 {{
            font-size: 3em;
            color: #FF5733;  /* Mengganti warna judul menjadi oranye */
            text-align: center;
            font-weight: bold;
            margin-top: 50px;
        }}

        /* Gaya untuk tombol */
        .stButton {{
            background-color: #4CAF50;  /* Hijau untuk tombol */
            color: white;
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 10px;
            border: none;
            width: 250px;
            margin: 20px auto;
            display: block;
        }}

        .stButton:hover {{
            background-color: #45a049;
        }}

        /* Gaya untuk uploader file */
        .stFileUploader {{
            display: block;
            margin: 0 auto;
            width: 80%;
        }}

        /* Gaya untuk gambar yang diunggah */
        .stImage {{
            border-radius: 15px;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
            margin: 15px 0;
        }}

        /* Gaya untuk hasil prediksi */
        .result-container {{
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
            gap: 20px;
        }}

        .result-item {{
            flex: 1 1 calc(33.33% - 20px);
            box-sizing: border-box;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
            text-align: center;
        }}

        /* Gaya untuk teks hasil prediksi */
        .result-item p {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Fungsi untuk mengunduh gambar dari URL dan mengubahnya menjadi base64
def get_image_as_base64_from_url(image_url):
    response = requests.get(image_url)
    if response.status_code == 200:
        return b64encode(response.content).decode()
    else:
        raise FileNotFoundError(f"Unable to fetch image from {image_url}")

# URL gambar background (gambar di GitHub)
bg_image_url = "https://raw.githubusercontent.com/zaidannn/Zaidan-UAP/main/Images/download%20(10).jpg"
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

    # Muat model
    model_path = Path(__file__).parent / "Model/Image/vegetable_classifier.h5"
    model = tf.keras.models.load_model(model_path)

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

        # Membuat container untuk hasil prediksi dengan tata letak yang lebih rapi
        st.markdown('<div class="result-container">', unsafe_allow_html=True)

        for upload in uploads:
            # Tampilkan setiap citra yang diunggah dalam tampilan rapi
            st.markdown('<div class="result-item">', unsafe_allow_html=True)
            st.image(upload, caption=f"Citra yang diunggah: {upload.name}", use_column_width=True, channels="RGB")

            with st.spinner(f"Memproses citra {upload.name} untuk prediksi..."):
                # Panggil fungsi prediksi
                try:
                    label, confidence = predict(upload)
                    st.write(f"**{upload.name}** - Label: **{label}**, Confidence: **{confidence:.5f}%**")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat memproses {upload.name}: {e}")
            st.markdown('</div>', unsafe_allow_html=True)  # Tutup div result-item

        st.markdown('</div>', unsafe_allow_html=True)  # Tutup div result-container
    else:
        st.error("Unggah setidaknya satu citra terlebih dahulu!")
