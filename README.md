# Klasifikasi Jenis Sayuran Dengan MobileNetV2 dan InceptionV3

<p align="center">
  <img src="https://raw.githubusercontent.com/zaidannn/Zaidan-UAP/main/Images/download%20(10).jpg" alt="Logo" width="300"/>
</p>

## Overview Project
Proyek ini bertujuan untuk mengembangkan sebuah sistem klasifikasi gambar yang dapat mengenali dan membedakan berbagai jenis sayuran, termasuk kategori seperti 'Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', dan 'Tomato'. Sistem ini dirancang untuk mendukung berbagai aplikasi, seperti pengelompokan otomatis hasil panen, pendataan inventaris sayuran, atau komponen dalam aplikasi yang memerlukan kemampuan pengenalan jenis sayuran berbasis gambar.

Link Dataset yang digunakan : " https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset "

Model yang digunakan : MobileNetV2 dan InceptionV3 dengan Architecture Model seperti gambar berikut.

### MobileNetV2 

MobileNetV2 adalah arsitektur deep learning yang dirancang untuk efisiensi tinggi dengan menggunakan **depthwise separable convolution**, yang memisahkan proses convolusi berdasarkan channel dan mengurangi jumlah parameter secara signifikan. Model ini memperkenalkan **bottleneck residual blocks** dengan **inverted residuals** dan **linear bottlenecks** untuk mempertahankan informasi penting sambil tetap efisien. Setiap blok dimulai dengan ekspansi channel untuk menangkap lebih banyak fitur, diikuti dengan kompresi untuk menjaga efisiensi. Dengan lapisan pooling global dan dense di akhir, MobileNetV2 sangat cocok untuk aplikasi klasifikasi gambar pada perangkat dengan sumber daya terbatas, seperti smartphone dan IoT.

<p align="center">
  <img src="https://raw.githubusercontent.com/zaidannn/Zaidan-UAP/main/Images/Mobilenetv2.jpg" alt="MobileNetV2 Architecture" width="600"/>
</p>

### InceptionV3 

InceptionV3 adalah arsitektur deep learning yang dirancang untuk efisiensi dan akurasi tinggi dalam klasifikasi gambar. Model ini merupakan pengembangan dari versi sebelumnya, dengan memperkenalkan beberapa inovasi seperti **factorized convolutions**, yang memecah convolusi besar menjadi convolusi yang lebih kecil untuk mengurangi jumlah parameter. InceptionV3 juga menggunakan teknik **batch normalization** untuk mempercepat konvergensi dan **auxiliary classifiers** sebagai regularisasi tambahan. Struktur model ini menggabungkan berbagai ukuran kernel dalam satu layer untuk menangkap informasi dari berbagai skala. Dengan desain modular yang efisien, InceptionV3 sangat cocok untuk berbagai aplikasi pengenalan gambar, termasuk yang membutuhkan performa tinggi dengan kompleksitas komputasi yang optimal.

<p align="center">
  <img src="https://github.com/zaidannn/Zaidan-UAP/blob/main/Images/Inceptionv3.jpg" alt="MobileNetV2 Architecture" width="600"/>
</p>

## Overview Dataset
Dataset terdiri atas 21000 gambar sayuran yang terbagi menjadi 15 kelas. Data terbagi menjadi 3 folder yaitu data train dengan 15000 gambar, data validation dengan 3000 gambar, dan data test dengan 3000 gambar.

## Model Evaluation
### MobileNetV2

<p align="center">
  <img src="https://github.com/zaidannn/Zaidan-UAP/blob/main/Images/learning%20curve%20mobilnetv2.png" alt="MobileNetV2 Architecture" width="600"/>
</p>

<p align="center">
  <img src="https://github.com/zaidannn/Zaidan-UAP/blob/main/Images/cr%20mobilenetv2.jpg" alt="MobileNetV2 Architecture" width="600"/>
</p>

### InceptionV3

<p align="center">
  <img src="https://github.com/zaidannn/Zaidan-UAP/blob/main/Images/learning%20curve%20inceptionv3.png" alt="MobileNetV2 Architecture" width="600"/>
</p>

<p align="center">
  <img src="https://github.com/zaidannn/Zaidan-UAP/blob/main/Images/cr%20inceptionv3.jpg" alt="MobileNetV2 Architecture" width="600"/>
</p>

## Deployment Web
### Tampilan Web

<p align="center">
  <img src="https://github.com/zaidannn/Zaidan-UAP/blob/main/Images/tampilanapp1.jpg" alt="MobileNetV2 Architecture" width="600"/>
</p>

<p align="center">
  <img src="https://github.com/zaidannn/Zaidan-UAP/blob/main/Images/tampilanapp2.jpg" alt="MobileNetV2 Architecture" width="600"/>
</p>

Link : https://zaidan-uap-jqdrdnw4efyuvjf6wfigyk.streamlit.app/

- Author @Zaidannn
