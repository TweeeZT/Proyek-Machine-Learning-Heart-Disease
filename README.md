# Proyek Machine Learning 

## Heart Disease Project

**Latar Belakang**

Penyakit jantung merupakan salah satu penyebab utama kematian di dunia. Deteksi dini terhadap risiko penyakit jantung sangat penting untuk mencegah komplikasi serius dan meningkatkan harapan hidup pasien. Dengan banyaknya faktor risiko seperti tekanan darah tinggi, kadar kolesterol, diabetes, dan gaya hidup tidak sehat, penggunaan teknologi seperti machine learning dapat membantu tenaga medis dalam memprediksi kondisi pasien secara lebih cepat dan akurat.

Dalam proyek ini, dibangun model klasifikasi biner untuk memprediksi apakah seseorang memiliki penyakit jantung berdasarkan sejumlah fitur kesehatan dan gaya hidup yang telah dikumpulkan dalam sebuah dataset.

## Business Understanding

### Problem Statements

- Fitur-fitur apa saja yang paling berpengaruh dalam menentukan seseorang memiliki penyakit jantung?

- Bagaimana menyiapkan dataset agar siap digunakan dalam model klasifikasi penyakit jantung?

- Algoritma machine learning apa yang memberikan performa terbaik dalam klasifikasi penyakit jantung?

### Goals

- Menjelaskan hubungan antara fitur-fitur dengan label “Heart Disease Status”.

- Menyiapkan dataset yang bersih dan seimbang untuk pelatihan model klasifikasi.

- Membangun dan membandingkan performa beberapa model machine learning dan deep learning.



### Solution statements
- Melakukan EDA dan visualisasi untuk menggali pola pada data.

- Menerapkan preprocessing, termasuk scaling, dan oversampling.

- Menggunakan beberapa model seperti KNN, Random Forest, XGBoost, dan Naive Bayes.


## Data Understanding
Dataset [Kaggle](https://www.kaggle.com/datasets/oktayrdeki/heart-disease) terdiri dari 10000 baris dan 21 fitur. Berisi berbagai indikator kesehatan dan faktor risiko yang berhubungan dengan penyakit jantung. Parameter seperti usia, jenis kelamin, tekanan darah, kadar kolesterol, kebiasaan merokok, dan pola olahraga dikumpulkan untuk menganalisis risiko penyakit jantung serta berkontribusi dalam penelitian kesehatan.

Variabel pada dataset ini sebagai berikut:

**Age**: Usia individu.

**Gender**:  Jenis kelamin individu (Male or Female).

**Blood Pressure**: Tekanan darah individu. (systolic).

**Cholesterol Level**: Total kadar kolesterol dalam tubuh individu.

**Exercise Habits**: Tingkat kebiasaan olahraga individu  (Low, Medium, High).

**Smoking**: Apakah individu merokok atau tidak (Yes or No).

**Family Heart Disease**: Apakah terdapat riwayat penyakit jantung dalam keluarga (Yes or No).

**Diabetes**: Apakah individu menderita diabetes (Yes or No).

**BMI**: Nilai indeks massa tubuh individu.

**High Blood Pressure**: Apakah individu memiliki tekanan darah tinggi  (Yes or No).

**Low HDL Cholesterol**: Apakah individu memiliki kadar HDL (kolesterol baik) yang rendah(Yes or No).

**High LDL Cholesterol**: Apakah individu memiliki kadar LDL (kolesterol jahat) yang tinggi (Yes or No).

**Alcohol Consumption**: Tingkat konsumsi alkohol individu (None, Low, Medium, High).

**Stress Level**: Tingkat stres individu (Low, Medium, High).

**Sleep Hours**:  Jumlah jam tidur individu setiap hari.

**Sugar Consumption**: Tingkat konsumsi gula individu(Low, Medium, High).

**Triglyceride Level**: Tingkat trigliserida dalam darah individu.

**Fasting Blood Sugar**:  Tingkat gula darah saat puasa.

**CRP Level**: Tingkat protein C-reaktif, penanda peradangan dalam tubuh.

**Homocysteine Level**: Tingkat homosistein, asam amino yang memengaruhi kesehatan pembuluh darah.

**Heart Disease Status**: Status individu terkait penyakit jantung (Yes or No).



### **Exploratory Data Analysis (EDA)**
Proses awal dalam analisis data yang bertujuan untuk memahami struktur, karakteristik, pola, dan hubungan dalam data sebelum dilakukan pemodelan atau analisis statistik lebih lanjut.

```
print("\nInformasi dataset:")
df.info()
```

```
Informasi dataset:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 21 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   Age                   9971 non-null   float64
 1   Gender                9981 non-null   object 
 2   Blood Pressure        9981 non-null   float64
 3   Cholesterol Level     9970 non-null   float64
 4   Exercise Habits       9975 non-null   object 
 5   Smoking               9975 non-null   object 
 6   Family Heart Disease  9979 non-null   object 
 7   Diabetes              9970 non-null   object 
 8   BMI                   9978 non-null   float64
 9   High Blood Pressure   9974 non-null   object 
 10  Low HDL Cholesterol   9975 non-null   object 
 11  High LDL Cholesterol  9974 non-null   object 
 12  Alcohol Consumption   7414 non-null   object 
 13  Stress Level          9978 non-null   object 
 14  Sleep Hours           9975 non-null   float64
 15  Sugar Consumption     9970 non-null   object 
 16  Triglyceride Level    9974 non-null   float64
 17  Fasting Blood Sugar   9978 non-null   float64
 18  CRP Level             9974 non-null   float64
 19  Homocysteine Level    9980 non-null   float64
 20  Heart Disease Status  10000 non-null  object 
dtypes: float64(9), object(12)
```

- **Melihat missing values**

```
print("\nMissing values per fitur:")
print(df.isnull().sum())
```

```
Missing values per fitur:
Age                       29
Gender                    19
Blood Pressure            19
Cholesterol Level         30
Exercise Habits           25
Smoking                   25
Family Heart Disease      21
Diabetes                  30
BMI                       22
High Blood Pressure       26
Low HDL Cholesterol       25
High LDL Cholesterol      26
Alcohol Consumption     2586
Stress Level              22
Sleep Hours               25
Sugar Consumption         30
Triglyceride Level        26
Fasting Blood Sugar       22
CRP Level                 26
Homocysteine Level        20
Heart Disease Status       0
```
Berdasarkan output di atas, hampir semua kolom memiliki missing value dengan "Alcohol Consumption" yang memiliki missing value paling banyak yaitu sebesar 2586

- **Melihat apakah ada data duplikat**
```
duplicates = df.duplicated()
duplicate_count = duplicates.sum()
print(f"Number of duplicate rows: {duplicate_count}")
```

```
Number of duplicate rows: 0
```
Berdasarkan output diatas, tidak ditemukan ada data yang duplikat
```
print("\nInformasi dataset:")
df.info()
```

**Visualisasi data**

- **Melihat distribusi data numerik**

![Image](https://github.com/user-attachments/assets/f1b9c8a5-3b91-471d-8add-0d507bd12d0f)

Data menunjukkan bahwa rata-rata individu berada pada usia paruh baya (49 tahun) dengan tekanan darah, kolesterol, trigliserida, gula darah puasa, dan kadar CRP yang relatif tinggi—mengindikasikan risiko besar terhadap penyakit kardiovaskular dan metabolik. Rata-rata BMI berada di kisaran overweight (29), dan kadar homosistein juga berada mendekati batas risiko. Satu-satunya faktor yang cukup ideal adalah durasi tidur, dengan rata-rata hampir 7 jam per hari. Secara keseluruhan, mayoritas individu memiliki profil kesehatan yang mengarah pada kondisi metabolik yang kurang baik.

- **Melihat distribusi data kategorikal**

![Image](https://github.com/user-attachments/assets/7c0c1509-d6a6-48bd-af9c-2f93316649d7)

Sebagian besar individu dalam data ini adalah pria dan sekitar separuh dari populasi memiliki kebiasaan merokok serta riwayat penyakit jantung dalam keluarga. Aktivitas fisik cenderung bervariasi, tetapi cukup banyak yang berada pada tingkat rendah hingga sedang. Sekitar separuh responden juga memiliki tekanan darah tinggi, kadar LDL tinggi, dan HDL rendah, yang merupakan faktor risiko utama penyakit jantung. Tingkat stres dan konsumsi gula cukup tinggi, sementara konsumsi alkohol lebih merata. Meskipun hanya 20% individu yang tercatat mengidap penyakit jantung, faktor-faktor risiko kesehatan yang signifikan tersebar luas dalam populasi.

- **Melihat heatmap korelasi antar fitur**

![Image](https://github.com/user-attachments/assets/cc950a9e-825a-4005-a3fd-18ef59e91e85)

Berdasarkan heatmap korelasi, tidak terdapat hubungan yang kuat antar variabel; seluruh nilai korelasi mendekati nol. Ini menunjukkan bahwa faktor-faktor seperti tekanan darah, kolesterol, BMI, gula darah, dan lainnya saling independen atau hanya memiliki hubungan yang sangat lemah. Artinya, setiap variabel kemungkinan memberi kontribusi unik terhadap kondisi kesehatan responden, dan tidak ada satu pun yang secara langsung berkorelasi tinggi dengan variabel lainnya dalam dataset ini.

- **Melihat heatmap korelasi untuk fitur penyakit jantung**

![Image](https://github.com/user-attachments/assets/164eb25a-d284-4a0a-bc7f-7fd30178fb0c)

Heatmap ini menunjukkan bahwa tidak ada variabel numerik yang memiliki korelasi kuat terhadap status penyakit jantung. Seluruh nilai korelasi sangat rendah, baik positif maupun negatif, dengan nilai tertinggi hanya sekitar 0,02 pada BMI. Ini mengindikasikan bahwa status penyakit jantung dalam data ini tidak dipengaruhi secara signifikan oleh satu pun variabel numerik secara langsung, sehingga kemungkinan besar faktor risiko lebih kompleks atau tersembunyi dalam interaksi antar variabel atau faktor kategorikal.

- **Jumlah penderita penyakit vs tidak**

![Image](https://github.com/user-attachments/assets/d3d46b70-478b-47b8-9d44-f3affb3943cb)

Distribusi data menunjukkan bahwa mayoritas invidividu (sekitar 80%) tidak memiliki penyakit jantung, sedangkan hanya sekitar 20% yang terdiagnosis.

- **Melihat distribusi numerik berdasarkan memiliki penyakit vs tidak**

![Image](https://github.com/user-attachments/assets/a5660a6e-3f7a-4f3f-a4ea-c13cda992a31)

Gambar di atas menunjukkan distribusi berbagai fitur terhadap status penyakit jantung. Secara umum, individu dengan penyakit jantung ("Yes") cenderung memiliki nilai yang sedikit lebih tinggi pada tekanan darah, kadar kolesterol, BMI, gula darah puasa, CRP, dan homosistein dibandingkan yang tidak terkena. Namun, perbedaannya tidak terlalu mencolok, mengindikasikan bahwa tidak ada satu fitur pun yang dominan sebagai indikator tunggal penyakit jantung.

- **Distribusi kategorikal terhadap penyakit jantung**

![Image](https://github.com/user-attachments/assets/093874c7-9d17-4489-b2be-f0eaf523dd52)

Berdasarkan visualisasi distribusi fitur kategorikal terhadap status penyakit jantung, terlihat bahwa faktor-faktor seperti kebiasaan olahraga rendah, riwayat penyakit jantung dalam keluarga, tekanan darah tinggi, diabetes, kadar kolesterol LDL tinggi dan HDL rendah, serta tingkat stres yang tinggi, cenderung berasosiasi dengan peningkatan kasus penyakit jantung. Selain itu, merokok, konsumsi alkohol dan gula dalam jumlah tinggi juga menunjukkan keterkaitan meskipun tidak sekuat faktor-faktor utama tadi. Secara umum, kondisi kesehatan dan gaya hidup tampak berperan penting dalam status penyakit jantung.
 
## Data Preparation
Data preparation adalah langkah penting untuk memastikan kualitas data sebelum digunakan dalam model machine learning. Data mentah sering kali mengandung nilai kosong, duplikasi, atau rentang nilai yang tidak konsisten, yang dapat memengaruhi kinerja model. Oleh karena itu, proses ini bertujuan untuk membersihkan dan mempersiapkan data agar analisis berjalan optimal.

- **Menghapus missing value**
```
jumlah_sebelum = len(df)
df = df.dropna()
jumlah_setelah = len(df)

print(f"Jumlah data sebelum drop missing value: {jumlah_sebelum}")
print(f"Jumlah data setelah drop missing value: {jumlah_setelah}")
print(f"Jumlah data yang dihapus: {jumlah_sebelum - jumlah_setelah}")
```

```
Jumlah data sebelum drop missing value: 10000
Jumlah data setelah drop missing value: 7067
Jumlah data yang dihapus: 2933
```
Dari 10000 data yang ada pada dataset, ternyata ada **2933** data yang memiliki missing value. Setelah di drop, sekarang jumlah data ada di **7067** data

- **Menghilangkan data duplikat**
```
len_before = len(df)
df = df.drop_duplicates()
len_after = len(df)

print(f"Jumlah data sebelum drop duplicate: {len_before}")
print(f"Jumlah data setelah drop duplicate: {len_after}")
print(f"Jumlah data yang dihapus (duplikat): {len_before - len_after}")
```

```
Jumlah data sebelum drop duplicate: 7067
Jumlah data setelah drop duplicate: 7067
Jumlah data yang dihapus (duplikat): 0
```
Tidak ditemukan data duplikat, maka dari itu data tetap berada di **7067**

- **Menghilangkan outlier dari data**
```
# Menghapus outlier
for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("Ukuran data setelah menghapus outlier:", df.shape)
```

```
Ukuran data setelah menghapus outlier: (7067, 21)

```
Tidak ditemukan data yang memiliki outlier, maka dari itu data tetap berada di **7067**

- **Oversampling data**
Oversampling perlu dilakukan karena dataset memiliki ketidakseimbangan, Metode oversampling yang dilakukan adalah SMOTE, yaitu menciptakan data sintetis berdasarkan interpolasi data kelas minoritas.

```
print("Sebelum Oversampling:")
print(df['Heart Disease Status'].value_counts())
print("\nPersentase:")
print((df['Heart Disease Status'].value_counts(normalize=True) * 100).round(2))

X = pd.get_dummies(df.drop('Heart Disease Status', axis=1))
y = df['Heart Disease Status']

# Terapkan SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

df = pd.concat([X_res, y_res], axis=1)

print("\nSetelah Oversampling:")
print(df['Heart Disease Status'].value_counts())
print("\nPersentase:")
print((df['Heart Disease Status'].value_counts(normalize=True) * 100).round(2))
```

```
Sebelum Oversampling:
Heart Disease Status
0    5632
1    1435
Name: count, dtype: int64

Persentase:
Heart Disease Status
0    79.69
1    20.31
Name: proportion, dtype: float64

Setelah Oversampling:
Heart Disease Status
0    5632
1    5632
Name: count, dtype: int64

Persentase:
Heart Disease Status
0    50.0
1    50.0
```
Sekarang dataset memiliki jumlah data yang seimbang, dengan masing-masing kelas memiliki jumlah data yang seimbang yaitu 50% untuk kelas 0 dan 50% untuk kelas 1

- **MinMax Scaling**

```
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

df = pd.concat([X, y.reset_index(drop=True)], axis=1)
```
MinMax scaling dilakukan untuk memastikan data asli dan sintetis memiliki skala yang sama, selain itu MinMax scaling juga berguna untuk algoritma machine learning yang sensitif seperti KNN

- **Mengisi data kosong dengan median**
```
imputer = SimpleImputer(strategy='median')
df[df.columns] = imputer.fit_transform(df)
```

**Split data untuk train dan test (80% train/20% test)**
```
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```


## Modeling 
### Modeling Machine Learning
- Random Forest

Random Forest adalah algoritma machine learning berbasis ensemble learning yang digunakan untuk tugas klasifikasi dan regresi. Algoritma ini bekerja dengan membangun banyak pohon keputusan (decision trees) selama proses pelatihan, lalu menggabungkan hasil prediksi dari pohon-pohon tersebut untuk menghasilkan keputusan akhir. Pada kasus klasifikasi, Random Forest mengambil keputusan berdasarkan suara terbanyak (mayoritas voting) dari semua pohon. Keunggulan utama dari Random Forest adalah akurasinya yang tinggi, ketahanannya terhadap overfitting, serta kemampuannya dalam menangani data dengan banyak fitur. Namun, kekurangan Random Forest terletak pada ukurannya yang besar, proses pelatihan yang bisa memakan waktu, serta hasil model yang sulit diinterpretasikan dibandingkan pohon keputusan tunggal. 

**Parameter**
- **n_estimator:50**= Jumlah pohon dalam hutan.

**Hasil**

Random Forest menunjukkan performa yang sangat baik dengan precision tinggi (99.53%) dan f1 score yang seimbang. Hal ini menunjukkan bahwa model sangat jarang memberikan false positive, namun recall-nya masih bisa ditingkatkan (75.31%), yang berarti ada beberapa kasus positif yang belum berhasil ditangkap.

- K Nearest Neighbour (KNN)

K-Nearest Neighbors (KNN) adalah algoritma machine learning yang digunakan untuk tugas klasifikasi dan regresi, dan termasuk dalam kelompok algoritma berbasis instance-based learning atau lazy learning. KNN bekerja dengan cara mencari sejumlah tetangga terdekat (K tetangga) dari data yang akan diprediksi, kemudian menentukan kelas atau nilai berdasarkan mayoritas (untuk klasifikasi) ]dari tetangga-tetangga tersebut. Jarak antar data biasanya dihitung menggunakan metrik seperti Euclidean distance. Karena KNN tidak membangun model secara eksplisit selama pelatihan, proses prediksi menjadi lebih lambat karena setiap prediksi melibatkan pencarian terhadap seluruh data latih. Keunggulan KNN terletak pada kesederhanaannya dan kemampuannya menangani data non-linear. Namun, algoritma ini dapat menjadi tidak efisien untuk dataset besar dan sensitif terhadap skala fitur, sehingga normalisasi data sering diperlukan.

**Parameter**
- **metric: manhattan**= Menggunakan jarak Manhattan untuk mengukur seberapa jauh dua data, yaitu menjumlahkan selisih absolut antar fitur.
- **n_neighbors: 13** = Model mempertimbangkan 13 tetangga terdekat saat membuat prediksi.
- **weights: distance** =  Tetangga yang lebih dekat memiliki pengaruh lebih besar daripada yang jauh dalam menentukan hasil prediksi.

**Hasil**
KNN memiliki performa yang sedikit di bawah Random Forest, baik dari segi accuracy maupun f1 score. Meskipun precision-nya sangat tinggi (98.82%), recall masih rendah (74.51%), yang menyebabkan f1 score sedikit lebih kecil. Model ini masih sangat baik, tetapi mungkin kurang optimal untuk data yang kompleks atau memiliki dimensi tinggi, karena KNN cenderung sensitif terhadap skala dan distribusi data.

- XGBoost (Extreme Gradient Boosting) 

XGBoost (Extreme Gradient Boosting) adalah algoritma machine learning yang berbasis pada teknik boosting, yaitu metode ensemble yang menggabungkan banyak model lemah (biasanya pohon keputusan sederhana) secara bertahap untuk membentuk model yang kuat. XGBoost dirancang untuk menjadi efisien, fleksibel, dan akurat, serta sangat populer karena kemampuannya menghasilkan performa tinggi dalam banyak kompetisi data science. Prinsip utamanya adalah membangun pohon-pohon secara berurutan, di mana setiap pohon baru berusaha memperbaiki kesalahan dari pohon-pohon sebelumnya dengan meminimalkan fungsi kerugian menggunakan pendekatan gradient descent. XGBoost juga memiliki berbagai fitur unggulan seperti regularisasi untuk menghindari overfitting, kemampuan menangani data kosong secara otomatis, serta paralelisasi dalam proses pelatihannya, yang membuatnya lebih cepat dibanding implementasi boosting lainnya.

**Parameter**
- **learning_rate: 0.1** = Mengontrol seberapa besar kontribusi setiap pohon baru terhadap model akhir. Nilai kecil  membuat model belajar secara perlahan tapi lebih stabil, sehingga mengurangi risiko overfitting. Nilai ini sering disebut juga sebagai step size shrinkage.
- **n_estimators: 100** = Menentukan jumlah pohon keputusan (trees) yang akan dibangun secara berurutan. Semakin besar nilainya, semakin kompleks dan akurat modelnya

**Hasil**
XGBoost adalah model dengan performa terbaik secara keseluruhan berdasarkan accuracy dan f1 score tertinggi. Precision-nya sempurna (100%), artinya tidak ada false positive sama sekali, dan recall cukup tinggi. Ini menjadikan XGBoost model paling seimbang dan kuat di antara semua model yang di uji, terutama dalam konteks data yang mungkin tidak seimbang. Performa tinggi ini menjadikan XGBoost pilihan unggulan jika resource pelatihan memadai.


- Naive Bayes

Naive Bayes adalah algoritma machine learning berbasis teori probabilitas Bayes yang digunakan untuk tugas klasifikasi. Algoritma ini disebut “naive” (naif) karena mengasumsikan bahwa setiap fitur dalam data bersifat independen satu sama lain, padahal dalam kenyataannya fitur-fitur tersebut seringkali saling berkaitan.  Cara kerjanya adalah dengan menghitung probabilitas dari setiap kelas berdasarkan fitur-fitur yang ada, lalu memilih kelas dengan probabilitas tertinggi sebagai hasil prediksi. Keunggulan utama Naive Bayes adalah kecepatannya dalam pelatihan dan prediksi, kemampuannya bekerja dengan baik pada data berdimensi tinggi, serta tidak memerlukan banyak data untuk mencapai akurasi yang memadai. Namun, karena mengandalkan asumsi independensi antar fitur, performanya bisa menurun jika hubungan antar fitur sangat kuat dan kompleks.

**Hasil**
Naive Bayes tampil cukup kompetitif dengan accuracy dan precision tinggi. Namun, recall-nya paling rendah bersama KNN. Seperti XGBoost, precision sempurna menunjukkan model sangat konservatif dalam memprediksi positif, sehingga minim false positive. Namun, pendekatan ini membuat model melewatkan lebih banyak kasus positif, menurunkan recall.

### Modeling Deep Learning
**Arsitektur dan Parameter Model**
Struktur model yang dibangun meliputi:

- **Input Layer:** Menerima sejumlah fitur numerik sesuai jumlah kolom pada dataset (berdasarkan X_train.shape[1]).

- **Lapisan Tersembunyi (Hidden Layers):**

 - Dense layer dengan 32 neuron dan aktivasi ReLU.

 - Dropout layer (rate 0.4) untuk mengurangi risiko overfitting.

 - Dense layer dengan 64 neuron dan aktivasi ReLU.

 - Dropout layer (rate 0.3) untuk regularisasi tambahan.

 - Dense layer dengan 128 neuron dan aktivasi ReLU guna menangkap pola lebih kompleks.

- **Output Layer:** Satu neuron dengan aktivasi sigmoid untuk memproduksi probabilitas klasifikasi biner.

**Konfigurasi Training**
- Optimizer: Adam (learning_rate=0.001)

- Fungsi Loss: Binary Cross-Entropy

- Metrik Evaluasi: Accuracy, Precision, dan Recall

**Detail Training**

- **epochs = 100** : Pelatihan dilakukan dalam 100 iterasi penuh.

- **batch_size = 32** : Data dibagi menjadi batch berisi 32 sampel sebelum model memperbarui bobotnya.

- **validation_split = 0.2** : Sebanyak 20% dari data latih dipakai sebagai data validasi untuk mengevaluasi performa selama pelatihan.

- **EarlyStopping** : Menghentikan pelatihan secara otomatis jika tidak ada penurunan pada val_loss selama 10 epoch berturut-turut.

- **ModelCheckpoint** : Menyimpan model terbaik berdasarkan nilai validasi terendah.

- **ReduceLROnPlateau** : Menurunkan learning rate ketika val_loss tidak menunjukkan perbaikan dalam beberapa epoch, untuk membantu model belajar lebih optimal.

**Proses Training**
Proses training diawali dengan meng split dataset ke 80% train dan 20% validasi. Model dilatih selama 100 epoch dan diterapkan callback untuk menyimpan model terbaik  berdasarkan akurasi validasi tertinggi untuk hasil yang optimal.

**Hasil Training**
Training hanya memerlukan 35 epoch untuk mencari hasil model yang terbaik

## Evaluation
**Dalam proyek klasifikasi status penyakit jantung ini, pemilihan metrik evaluasi didasarkan pada relevansinya terhadap konteks medis dan kebutuhan analisis yang telah dijelaskan pada tahap Business Understanding.**

**Metrik Evaluasi**

Metrik yang digunakan antara lain:

- **Accuracy (Akurasi):**
Mengukur proporsi keseluruhan prediksi yang benar. Akurasi memberikan gambaran umum seberapa baik model dalam membedakan antara individu dengan dan tanpa penyakit jantung berdasarkan fitur numerik yang tersedia.

- **Precision (Presisi):**
Mengukur seberapa banyak prediksi positif (individu memiliki penyakit jantung) yang benar-benar tepat. Precision menjadi penting dalam konteks medis untuk menghindari false positive, yakni kesalahan dalam mengklasifikasikan orang sehat sebagai sakit, yang bisa menyebabkan intervensi medis yang tidak perlu.

- **Recall (Sensitivitas):**
Menilai seberapa baik model dalam mengenali semua kasus aktual penyakit jantung. Nilai recall yang tinggi menunjukkan bahwa model mampu meminimalkan false negative, yaitu kasus di mana pasien yang sebenarnya sakit tidak terdeteksi. Hal ini sangat krusial untuk mencegah risiko kesehatan yang tidak tertangani.

- **F1-Score:**
Merupakan rata-rata harmonik dari precision dan recall, digunakan sebagai indikator keseimbangan performa model, terutama ketika data tidak sepenuhnya seimbang. F1-Score menjadi metrik penting untuk menilai kemampuan model dalam situasi trade-off antara presisi dan sensitivitas.

## Analisis Hasil Training

**Performa Model Machine Learning**
```
              Accuracy  Precision  Recall  F1 Score
Model                                              
RandomForest    0.8748     0.9953  0.7531    0.8574
KNN             0.8682     0.9882  0.7451    0.8496
XGBoost         0.8762     1.0000  0.7522    0.8586
NaiveBayes      0.8726     1.0000  0.7451    0.8539
```

**Performa Model Deep Learning**
```
Accuracy:  0.8735
Precision: 1.0000
Recall:    0.7469
F1 Score:  0.8551
```
Berdasarkan hasil evaluasi, model deep learning menunjukkan performa yang kompetitif dibandingkan dengan model machine learning seperti Random Forest dan XGBoost. Dengan nilai precision 100%, model deep learning mampu menghindari false positive, yang penting dalam konteks medis untuk mencegah kesalahan diagnosis pada pasien sehat. Meskipun nilai recall-nya (74,69%) sedikit lebih rendah dibandingkan Random Forest dan XGBoost, nilai F1 Score yang dihasilkan (85,51%) tetap menunjukkan keseimbangan yang baik antara precision dan recall. Akurasinya (87,35%) juga sebanding dengan model-model terbaik lainnya. Secara keseluruhan, model deep learning tidak hanya setara dari sisi performa, tetapi juga menawarkan fleksibilitas untuk pengembangan lebih lanjut guna meningkatkan akurasi diagnosis penyakit jantung.

## Hubungan dengan Business Understanding & Mengatasi Problem Statements

- Proyek ini berhasil menjawab seluruh problem statements yang telah ditetapkan. Melalui proses eksplorasi data (EDA), ditemukan bahwa faktor-faktor seperti kebiasaan olahraga rendah, riwayat penyakit jantung dalam keluarga, tekanan darah tinggi, diabetes, stres tinggi, dan kolesterol tidak normal memiliki asosiasi kuat dengan status penyakit jantung. Ini menjawab problem pertama mengenai fitur paling berpengaruh.

- Proses data preparation dilakukan secara menyeluruh dengan menghapus missing value, menghindari duplikasi, melakukan oversampling menggunakan SMOTE untuk menangani ketidakseimbangan kelas, serta menerapkan MinMax scaling. Hal ini menjawab problem kedua tentang bagaimana menyiapkan dataset agar siap digunakan untuk pelatihan.

- Dalam menjawab problem ketiga mengenai model terbaik, dilakukan perbandingan antara beberapa model machine learning (Random Forest, KNN, XGBoost, dan Naive Bayes) serta model deep learning. Hasil menunjukkan bahwa model deep learning memberikan performa yang sangat kompetitif, dengan precision sempurna dan F1-score tinggi, menandakan kemampuannya yang kuat dalam klasifikasi status penyakit jantung.

## Solution Statements
Beberapa langkah yang dijabarkan dalam solution statements terbukti memberikan dampak signifikan terhadap kualitas data dan performa model:

- **EDA dan Visualisasi:**
Analisis univariat dan multivariat membantu mengungkap pola penting dalam data, seperti hubungan antara gaya hidup (merokok, olahraga, konsumsi gula) dan status penyakit jantung. Visualisasi distribusi data numerik dan kategorikal memperkuat pemahaman tentang faktor risiko, mendukung proses pemilihan fitur dan strategi pemodelan.

- **Preprocessing dan Penyeimbangan Data:**
Penanganan missing value, penghapusan duplikasi, dan penggunaan SMOTE untuk oversampling berhasil menyeimbangkan distribusi kelas dalam dataset. Hal ini penting untuk meningkatkan recall dan F1-score, terutama pada kasus minoritas (penderita penyakit jantung), dan secara langsung menjawab tantangan dari data yang tidak seimbang.

- **Scaling Fitur:**
Penerapan MinMaxScaler membantu algoritma yang sensitif terhadap skala (seperti KNN dan neural network) untuk belajar secara optimal, meningkatkan performa keseluruhan model.

- **Pemodelan Machine Learning dan Deep Learning:**
Implementasi dan evaluasi beberapa algoritma memberikan pemahaman yang jelas tentang kekuatan dan kelemahan masing-masing pendekatan. Meskipun model seperti XGBoost unggul dalam akurasi dan F1-score, pendekatan deep learning menunjukkan fleksibilitas tinggi serta precision sempurna (100%), memperkuat potensinya untuk digunakan dalam konteks prediksi medis.

- **Evaluasi Multi-Metrik:**
Penggunaan metrik yang beragam (accuracy, precision, recall, F1-score) memberikan penilaian yang komprehensif atas performa model. Evaluasi ini menunjukkan bahwa model deep learning tidak hanya akurat, tetapi juga mampu meminimalkan kesalahan klasifikasi yang berisiko dalam konteks diagnosis penyakit jantung.

## Author
Name : Cahya Abdurrahman
 
Email : cahyaabd@upi.edu