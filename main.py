# Import Library yang digunakan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from scipy.stats import uniform
import xgboost as xgb

# Membaca dataset
df = pd.read_csv('/content/drive/MyDrive/Colab/loan_data_2007_2014.csv')

# Melihat 5 baris pertama dataset
print('\nBerikut tabel data: ')
print(df.head(100))

# Melihat dimensi data
print('\nDimensi data: ')
print(df.shape)

# Melihat nama kolom
print('\nNama kolom: ')
print(df.columns)

# Melihat info data
print('\nBerikut info data: ')
print(df.info())

# Melihat jenis tipe data setiap kolom
print('\nBerikut jenis tipe data setiap kolom: ')
print(df.dtypes)
print('\nJumlah setiap tipe data:\n',df.dtypes.value_counts())

# Cek missing value
df_mv = df.isna().sum()
print('\nApakah ada missing value ?')
if df_mv.sum() == 0:
    print('Tidak ada missing value')
else:
    print('Ada missing value sebanyak', df_mv.sum())
    print('\nAdapun sebagai berikut:\n', df_mv)
#
# Cek kolom yang mengandung missing value
col_null = df.columns[df.isna().any()]
print('\nKolom yang memiliki missing value: ')
print(col_null)
print('Jumlah kolom yang memiliki missing value:', len(col_null))
#
# Cek kolom yang mengandung missing value 100%, lebih dari 75%, lebih dari 50% dari total baris
#
#
# Cek kolom yang mengandung missing value 100% dari baris
col_null_100_percent = df.columns[df.isna().sum() == 1.0 * len(df)]
print('\nKolom yang memiliki missing value 100%: ')
print(col_null_100_percent)
print('Jumlah kolom yang memiliki missing value 100%: ', len(col_null_100_percent))
#
# Cek kolom yang mengandung missing value lebih dari 75% baris
col_null_75_percent = df.columns[df.isna().sum() > 0.75 * len(df)]
print('Jumlah kolom yang memiliki missing value lebih dari 75%: ', len(col_null_75_percent))
#
# Cek kolom yang mengandung missing value lebih dari 50% baris
col_null_50_percent = df.columns[df.isna().sum() >= 0.50 * len(df)]
print('\nKolom yang memiliki missing value lebih dari 50%: ')
print(col_null_50_percent)
print('Jumlah kolom yang memiliki missing value lebih dari 50%: ', len(col_null_50_percent))
# ================================================================

# Cek duplikat
df_dup = df.duplicated().sum()
print('\nApakah ada data duplikat ?')
if df_dup == 0:
    print('Tidak ada data duplikat')
else:
    print('Ada data duplikat sejumlah', df_dup.sum())
# ====================================================

# Cek konsistensi data - menampilkan nilai unik untuk seluruh kolom
for col in df.columns:
    grouped_df = df.groupby([col])[col].count()
    print(f"\nGrouped data for column '{col}':\n")
    print(grouped_df)
    print("-" * 30) # Separator for better readability
# ===================================================

# Deteksi Outlier menggunakan boxplot
print('\nDeteksi Outlier pada kolom atau fitur yang akan digunakan:')
for col_used in ['acc_now_delinq', 'addr_state', 'annual_inc', 'collection_recovery_fee',
       'collections_12_mths_ex_med', 'delinq_2yrs', 'dti', 'emp_length',
       'funded_amnt', 'funded_amnt_inv', 'grade', 'home_ownership',
       'initial_list_status', 'inq_last_6mths', 'installment', 'int_rate',
       'last_pymnt_amnt', 'loan_amnt', 'open_acc', 'out_prncp',
       'out_prncp_inv', 'pub_rec', 'purpose', 'pymnt_plan', 'recoveries',
       'revol_bal', 'revol_util', 'sub_grade', 'term', 'tot_coll_amt',
       'tot_cur_bal', 'total_acc', 'total_pymnt', 'total_pymnt_inv',
       'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp',
       'total_rev_hi_lim', 'verification_status']:
      plt.figure(figsize=(4, 2))
      sns.boxplot(x=df[col_used])
      plt.title(f"Deteksi Outlier Pada {col_used}")
      plt.show()

# Identifikasi Outlier menggunakan (IQR)
for col_used in df.columns[:-1]:  # Fitur numerik
    # Cek apakah kolom bertipe data numerik sebelumn melakukan perhitungan
    if pd.api.types.is_numeric_dtype(df[col_used]):
        Q1 = df[col_used].quantile(0.25)
        Q3 = df[col_used].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col_used] < lower_bound) | (df[col_used] > upper_bound)]
        print(f"\nOutlier pada kolom {col_used}:")
        print(outliers)
    else:
        print(f"\nKolom '{col_used}' dilewati (bukan tipe numerik).")

  # Melihat statistik deskriptif untuk seluruh kolom yang tidak memiliki missing value lebih dari 50% dari total baris
not_null_columns = df.columns[df.notnull().sum() > 0.5 * len(df)] # notnull berarti tidak mengandung missing value
df_sd = df[not_null_columns].describe()
print('Berikut statistik data: \n')
print(df_sd)
# ==============================================================

df_original = pd.read_csv('/content/drive/MyDrive/Colab/loan_data_2007_2014.csv')

# Melihat distribusi
distribusi_col = df_original.select_dtypes(include='number').columns

for col in distribusi_col:
    plt.figure(figsize=(8,6))
    sns.histplot(data=df_original, x=col, kde=True)
    plt.title(f"Distribusi {col}")
    plt.show()

# Melihat perbandingan jumlah dari setiap nilai pada kolom term dalam bentuk pie chart
term_counts = df_original['term'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(term_counts, labels=term_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Loan Terms')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Melihat perbandingan jumlah dari setiap nilai pada kolom grade dalam bentuk pie chart
# Melihat dalam artian grade mana yang paling banyak dipilih oleh peminjam
grade_counts= df_original['grade'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(grade_counts, labels=grade_counts.index, autopct='%.1f%%', startangle=140)
plt.title('Distribution of Loan Grades')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Melihat perbandingan jumlah dari setiap nilai pada kolom subgrade dalam bentuk pie chart
# Melihat dalam artian subgrade mana yang paling banyak dipilih oleh peminjam
sub_grade_counts= df_original['sub_grade'].value_counts().nlargest(5)
plt.figure(figsize=(8,8))
plt.pie(sub_grade_counts, labels=sub_grade_counts.index, autopct='%.1f%%', startangle=140)
plt.title('Distribution of Loan Sub Grades')
plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

# Melihat perbandingan jumlah dari setiap nilai pada kolom add_state
# Dalam artian melihat daerah mana yang paling banyak nasabahnya
addr_state_counts = df_original['addr_state'].value_counts().nlargest(5) #n.largest untuk mengambil top 5
plt.figure(figsize=(10,8))
plt.bar(addr_state_counts.index.astype(str), addr_state_counts.values) #addr_state_counts.index.astype(str) sebagai x dan addr_state_counts. value sebagai y
plt.title('Distribusi berdasarkan addr_state')
plt.xlabel('Address')
plt.ylabel('Jumlah')
plt.show()

# Membuat chart garis pertumbuhan loan_ammount dari tahun 2007 sampai tahun 2014
df_original['issue_d'] = pd.to_datetime(df_original['issue_d'], format='%b-%y')
df_original['year'] = df_original['issue_d'].dt.year

loan_growth_by_year = df_original.groupby('year')['loan_amnt'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=loan_growth_by_year, x='year', y='loan_amnt')
plt.title('Pertumbuhan Jumlah Pinjaman per Tahun (2007-2014)')
plt.xlabel('Tahun')
plt.ylabel('Jumlah Pinjaman')
plt.grid(True)
plt.show()

# Membuat chart scatter hubungan pendapatan tahunan dengan jumlah pinjaman yang diambil berdasarkan perbandingan jangka waktu (term) pinjaman
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_original, x='annual_inc', y='loan_amnt', hue='term', alpha=0.5)
plt.title('Hubungan Pendapatan Tahunan dengan Jumlah Pinjaman Berdasarkan Jangka Waktu')
plt.xlabel('Pendapatan Tahunan')
plt.ylabel('Jumlah Pinjaman')
plt.show()

'''  Membersihkan kolom data yang mengandung outlier '''
for col_used in df.columns[:-1]:
  if pd.api.types.is_numeric_dtype(df[col_used]):
    Q1 = df[col_used].quantile(0.25)
    Q3 = df[col_used].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col_used] = df[col_used].clip(lower_bound, upper_bound)

# Membuang data dengan memilih kolom yang ingin dipertahankan
columns_to_remove = ['desc', 'mths_since_last_delinq', 'mths_since_last_record',
                     'mths_since_last_major_derog', 'annual_inc_joint', 'dti_joint',
                     'verification_status_joint', 'open_acc_6m', 'open_il_6m', 'open_il_12m',
                     'open_il_24m', 'mths_since_rcnt_il', 'total_bal_il', 'il_util',
                     'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi',
                     'total_cu_tl', 'inq_last_12m', 'next_pymnt_d','Column1', 'id', 'member_id', 'emp_title',
                     'url', 'title', 'policy_code', 'application_type', 'Unnamed: 0'
                     , 'zip_code']

# Dapatkan semua nama kolom
all_columns = df.columns

# Dapatkan nama kolom yang ingin dipertahankan (semua kolom dikurangi kolom yang akan dihapus)
columns_to_keep = all_columns.difference(columns_to_remove)

# Buat DataFrame baru dengan hanya kolom yang dipertahankan
df = df[columns_to_keep]

print('\nInformasi DataFrame setelah menghapus kolom:')
print(df.info())
print('\nDimensi DataFrame setelah menghapus kolom:')
print(df.shape, '\n')
#=====================================================================

# Mentransformasikan data objek menjadi waktu
datetime_col = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'last_credit_pull_d']
# Mengecek apakah date time masih berada di kolom df
datetime_col_present = [col for col in datetime_col if col in df.columns]
if datetime_col_present:
  print(df[datetime_col_present].info())
  print(df[datetime_col_present])
  for col in datetime_col_present:
      df[col] = pd.to_datetime(df[col], format='%b-%y', errors='coerce')
else:
  print("\nDatetime columns were already removed.")


# Merubah dari data objek menjadi numeric
'''> Catatan
---> Penambahan kolom baru untuk label atau target klasifikasi ['loan_status' = 'good_bad_status']
---> Label Encoding kolom ['home_ownership', 'purpose', 'addr_state', 'initial_list_status'
      , '']
---> Ordinal Encoding kolom ['term', 'grade', 'sub_grade', 'emp_length', 'verification_status'
      , 'pymnt_plan']
'''
# Melakukan encoding loan_status untuk menjadi kolom good_bad_status
# Encoding loan_status
loan_status_mapping = {
    'Fully Paid': 1,
    'Charged Off': 0,
    'Current': 1,
    'Default': 0,
    'Late (31-120 days)': 0,
    'In Grace Period': 1,
    'Late (16-30 days)': 0,
    'Does not meet the credit policy. Status:Fully Paid': 1,
    'Does not meet the credit policy. Status:Charged Off': 0
}

# Menambahkan kolom baru 'good_bad_status' berdasarkan mapping
df['good_bad_status'] = df['loan_status'].map(loan_status_mapping)

# Menampilkan hasil
print('\nBerikut value terbaru dari kolom good_bad_status:', df['good_bad_status'])
print('\nBerikut jumlah dari setiap value kolom good_bad_status', df['good_bad_status'].value_counts(), '\n')


# Menghapus kolom loan_status karena sudah diganti menjadi good_bad_status
df = df.drop('loan_status', axis=1)
#===========================================================================

# Melakukan label encoding dan ordinal encoding
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Melakukan label encoding
# Kolom yang ingin di-label encode

Label_Encoding_kolom = ['home_ownership', 'purpose', 'addr_state', 'initial_list_status']

# Dictionary untuk menyimpan encoder tiap kolom
encoders = {}

for col in Label_Encoding_kolom:
    le = preprocessing.LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Menampilkan hasil label encoding
print('\nIni adalah hasil label encoding\n', df)
# ==========================================================

# Melakukan ordinal encoding
# Definisikan urutan nilai untuk setiap kolom ordinal
ordinal_mapping = {
    'term': {' 36 months': 0, ' 60 months': 1},
    'grade': {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6},
    'sub_grade': {
        'A1':0,'A2':1,'A3':2,'A4':3,'A5':4,
        'B1':5,'B2':6,'B3':7,'B4':8,'B5':9,
        'C1':10,'C2':11,'C3':12,'C4':13,'C5':14,
        'D1':15,'D2':16,'D3':17,'D4':18,'D5':19,
        'E1':20,'E2':21,'E3':22,'E4':23,'E5':24,
        'F1':25,'F2':26,'F3':27,'F4':28,'F5':29,
        'G1':30,'G2':31,'G3':32,'G4':33,'G5':34
    },
    'emp_length': {
        '< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,
        '5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10
    },
    'verification_status': {'Not Verified':0, 'Verified':1, 'Source Verified':2},
    'pymnt_plan': {'n':0, 'y':1}
}

# Kolom yang ingin di-ordinal encode
Ordinal_Encoding_kolom = list(ordinal_mapping.keys())

# Terapkan mapping ke DataFrame
for col in Ordinal_Encoding_kolom:
    df[col] = df[col].map(ordinal_mapping[col])

# Menampilkan hasil ordinal encoding
print('\nIni adalah hasil ordinal encoding\n', df)
# ================================================

# Melihat kekuatan hubungan antar variabel
numerical_data = df.select_dtypes(include=np.number).copy() # Membuat sebuah copy untuk menghindari SettingWithCopyWarning
# Mengeceualikan kolom bertipe data date time dari tipe data numerik sebelum menghitung korelasi
numerical_data = numerical_data.select_dtypes(exclude=['datetime64[ns]'])

# Menghapus kolom dengan 0 variansi sebelum menghitung korelasi
variance = numerical_data.var()
constant_columns = variance[variance == 0].index
numerical_data = numerical_data.drop(columns=constant_columns)
print(f"\nKolom dengan varians nol yang dihapus sebelum menghitung korelasi: {list(constant_columns)}")


correlation_matrix = numerical_data.corr()
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriks Korelasi")
plt.show()

# Hasil typedata terbaru dari dataframe
print('\nBerikut Info Tipe Data terbaru dari dataframe: \n')
print(df.dtypes)
print(df.dtypes.value_counts())
#
# ==================================================================

# Data Frame sebelum handling missing value
print('Data kolom yang memiliki missing value')
print(df.isna().sum())
print('\nJumlah missing value:', df.isna().sum().sum())
print('Dimensi data:', df.shape)

# Membuang seluruh baris yang mengandung missing value
df.dropna(inplace = True) # inplace = True menetapkan secara permanent data frame baru dengan kondisi sudah membuang missing value (karena digunakan pada fungsi .dropna)

# Data Frame setelah handling missing value
print('\nCek ulang data terbaru setelah dropna')
print(df.isna().sum())
print('\nCek ulang jumlah missing value data terbaru sebanyak:', df.isna().sum().sum())
print('Dimensi ulang data:', df.shape)

# Normalisasikan data
# Sebelum mendefinisikan pastikan memisahkan kolom yang numberik dan yang bukan
#
# Pisahkan kolom bertipe bukan numberik dan yang numberik
numerical_cols = df.select_dtypes(include=np.number).columns
non_numerical_cols = df.select_dtypes(exclude=np.number).columns
#
# Membuang kolom bertipe data date time di dalam kolom tipe non-numerik karena tidak diperlukan dalam model
datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
non_numerical_cols = non_numerical_cols.drop(datetime_cols, errors='ignore')

# Definisikan MinMaxScaler
scaler = preprocessing.MinMaxScaler()

# Reset indeks untuk menghindari masalah ketidaksejajaran selama penggabungan.
df = df.reset_index(drop=True)

# Scale numerical columns atau kolom numberik
df_norm_numerical = scaler.fit_transform(df[numerical_cols])
df_norm_numerical = pd.DataFrame(df_norm_numerical, columns=numerical_cols)

# Gabungkan kolom numerik yang telah diskalakan dengan kolom non-numerik asli.
df_norm = pd.concat([df_norm_numerical, df[non_numerical_cols]], axis=1)
df = df_norm

# Mencetak dataframe
print(df.head())
print(df.dtypes)
print(df.shape)
print(df.isnull().sum())
print(df.isnull().sum().sum())

# Melakukan pengecekan ulang data setelah berbagai proses
# Cek konsistensi data - menampilkan nilai unik untuk seluruh kolom
for col in df.columns:
    grouped_df = df.groupby([col])[col].count()
    print(f"\nGrouped data for column '{col}':\n")
    print(grouped_df)
    print("-" * 30) # Pemisah untuk meningkatkan keterbacaan

# Mengidentifikasi kolom yang memiliki variansi 0 atau kosong
zero_variance_cols = df.columns[df.var() == 0]

# Buang kolom dengan variansi 0 atau kosong
df = df.drop(columns=zero_variance_cols)

print(f"Kolom yang dihapus karena memiliki varians nol: {list(zero_variance_cols)}")
print("\nInformasi DataFrame setelah menghapus kolom dengan varians nol:")
print(df.info())
print("\nDimensi DataFrame setelah menghapus kolom dengan varians nol:")
print(df.shape)

# Pisahkan data menjadi nilai X dan y
X = df.drop('good_bad_status', axis=1)
y = df['good_bad_status']

# Stratified Sampling
X_sample, _, y_sample, _= train_test_split(
    X, y,
    train_size=0.3,       # Jumlah persen data yang diambil
    stratify=y,           # stratified berdasarkan distribusi target
    random_state=42
)
print("Jumlah data asli :", len(y))
print("Jumlah data sampel:", len(y_sample))
print("Distribusi target sampel:")
print(y_sample.value_counts(normalize=True))
print(y_sample.value_counts())


# Melihat perbandingan jumlah dari setiap nilai pada kolom good_bad_status dalam bentuk bar
grade_counts= y_sample.value_counts()
plt.figure(figsize=(4,4))
plt.bar(grade_counts.index.astype(str), grade_counts.values)
plt.title('Perbandingan nilai 0 dan 1 good_bad_status')
plt.xlabel('Good/Bad Status')
plt.ylabel('Proportion')
plt.show()

# Menentukan fitur yang digunakan
selector = SelectKBest(score_func=f_classif, k=32)
selector.fit(X_sample, y_sample)
selected_features = X.columns[selector.get_support()]
print("\nFitur terpilih:", selected_features)

# Menggunakan sampel data yang sudah distratifikasi untuk pembagian data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X_sample[selected_features], y_sample, test_size=0.2, random_state=42
)

print('\nDimensi X latih:', X_train.shape)
print('Dimensi X uji:', X_test.shape)
print('Dimensi y latih:', y_train.shape)
print('Dimensi y uji:', y_test.shape)
print('\nSebelum oversampling:', pd.Series(y_train).value_counts())

# Oversampling dengan SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)
print("Setelah oversampling:", pd.Series(y_train_over).value_counts())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
'''
# Library dibawah digunakan jika ingin menggunakan hyperparameter tuning terkhusus pada random search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
'''

# Definisi model
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_over, y_train_over)

# --- Evaluasi dari data latih ---
y_train_pred_proba = model.predict_proba(X_train_over)[:, 1]
y_train_pred = (y_train_pred_proba >= 0.5).astype(int) # Asumsikan threshold sama seperti data uji
accuracy_train = accuracy_score(y_train_over, y_train_pred)
roc_auc_train = roc_auc_score(y_train_over, y_train_pred_proba)

print("\nHasil Evaluasi pada Data Latih:")
print(f"- Akurasi: {accuracy_train}")
print(f"- ROC-AUC Score: {roc_auc_train}")


# --- Evaluasi dari data uji ---
# Dapatkan probabilitas yang diprediksi untuk ROC-AUC
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# Definisikan threshold
threshold = 0.5 # Tentukan threshold yang diinginkan

# Lakukan prediksi dengan data testing menggunakan model terbaik
y_test_pred = (y_test_pred_proba >= threshold).astype(int)


# Evaluasi model
accuracy_test = accuracy_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)

# Catatan hasil eksekusi dan parameter model
print("\nHasil Eksekusi pada Data Uji:")
print(f"- Algoritma yang digunakan: Logistic Regression")
print(f"- Fitur yang digunakan: {selected_features}")
print(f"- Akurasi model terbaik: {accuracy_test}")
print(f"- ROC-AUC Score terbaik: {roc_auc_test}")
print("- Laporan Klasifikasi model terbaik:")
print(classification_report(y_test, y_test_pred))

import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
'''
# Library dibawah digunakan jika ingin menggunakan hyperparameter tuning terkhusus pada random search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
'''

# Definisikan model
model = xgb.XGBClassifier(random_state=42, class_weight='balanced', use_label_encoder=False, eval_metric='logloss') # Corrected: use XGBClassifier
model.fit(X_train_over, y_train_over)

# --- Evaluaasi dari data latih ---
y_train_pred_proba = model.predict_proba(X_train_over)[:, 1]
y_train_pred = (y_train_pred_proba >= 0.5).astype(int) # Asumsikan threshold sama seperti data uji
accuracy_train = accuracy_score(y_train_over, y_train_pred)
roc_auc_train = roc_auc_score(y_train_over, y_train_pred_proba)

print("\nHasil Evaluasi pada Data Latih:")
print(f"- Akurasi: {accuracy_train}")
print(f"- ROC-AUC Score: {roc_auc_train}")

# --- Evaluasi dari data uji ---
# Prediksi probabilitas untuk ROC-AUC
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

# Definisikan threshold
threshold = 0.5 # Tentukan threshold yang diinginkan

# Lakukan prediksi dengan data testing menggunakan model terbaik
y_test_pred = (y_test_pred_proba >= threshold).astype(int)

# Evaluasi model
accuracy_test = accuracy_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, y_test_pred_proba)

# Catatan hasil eksekusi dan parameter model
print("\nHasil Eksekusi pada Data Uji:")
print(f"- Algoritma yang digunakan: XGBoost")
print(f"- Fitur yang digunakan: {selected_features}")
print(f"- Akurasi model terbaik: {accuracy_test}")
print(f"- ROC-AUC Score terbaik: {roc_auc_test}")
print("- Laporan Klasifikasi model terbaik:")
print(classification_report(y_test, y_test_pred))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Definisikan model Logistic Regression
model_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

# Definisikan parameter grid untuk Grid Search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Inisialisasi GridSearchCV
grid_search = GridSearchCV(estimator=model_lr, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)

# Lakukan Grid Search pada data training yang sudah di oversample
grid_search.fit(X_train_over, y_train_over)

# Dapatkan model terbaik dari Grid Search
best_model_lr = grid_search.best_estimator_

# --- Evaluasi Model Terbaik dari data latih ---
y_train_pred_proba_lr_gs = best_model_lr.predict_proba(X_train_over)[:, 1]
y_train_pred_lr_gs = (y_train_pred_proba_lr_gs >= 0.5).astype(int) # Asumsikan threshold sama seperti data uji
accuracy_train_lr_gs = accuracy_score(y_train_over, y_train_pred_lr_gs)
roc_auc_train_lr_gs = roc_auc_score(y_train_over, y_train_pred_proba_lr_gs)

print("\nHasil Evaluasi Model Terbaik (Grid Search LR) pada Data Latih:")
print(f"- Akurasi: {accuracy_train_lr_gs}")
print(f"- ROC-AUC Score: {roc_auc_train_lr_gs}")


# --- Evaluasi Model Terbaik dari data uji ---
# Lakukan prediksi dengan data testing menggunakan model terbaik
y_test_pred_proba_lr_gs = best_model_lr.predict_proba(X_test)[:, 1]

threshold = 0.5 # threshold
y_test_pred_lr_gs = (y_test_pred_proba_lr_gs >= threshold).astype(int) # Menggunakan threshold

# Evaluasi model terbaik
accuracy_test_lr_gs = accuracy_score(y_test, y_test_pred_lr_gs)
roc_auc_test_lr_gs = roc_auc_score(y_test, y_test_pred_proba_lr_gs)

# Catatan hasil eksekusi dan parameter model terbaik
print("\nHasil Grid Search - Logistic Regression:")
print(f"- Parameter terbaik: {grid_search.best_params_}")
print(f"- ROC-AUC terbaik pada cross-validation: {grid_search.best_score_}")
print("\nHasil Evaluasi Model Terbaik pada Data Testing:")
print(f"- Akurasi: {accuracy_test_lr_gs}")
print(f"- ROC-AUC Score: {roc_auc_test_lr_gs}")
print("- Laporan Klasifikasi:")
print(classification_report(y_test, y_test_pred_lr_gs))

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Definisikan model XGBoost
model_xgb = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Definisikan parameter grid untuk Grid Search (contoh parameter)
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Inisialisasi GridSearchCV
grid_search_xgb = GridSearchCV(estimator=model_xgb, param_grid=param_grid_xgb, cv=3, scoring='roc_auc', n_jobs=-1)

# Lakukan Grid Search pada data training yang sudah di oversample
grid_search_xgb.fit(X_train_over, y_train_over)

# Dapatkan model terbaik dari Grid Search
best_model_xgb = grid_search_xgb.best_estimator_

# --- Evaluasi Model Terbaik dari data latih ---
y_train_pred_proba_xgb_gs = best_model_xgb.predict_proba(X_train_over)[:, 1]
y_train_pred_xgb_gs = (y_train_pred_proba_xgb_gs >= 0.5).astype(int) # Samakan threshold dengan data uji
accuracy_train_xgb_gs = accuracy_score(y_train_over, y_train_pred_xgb_gs)
roc_auc_train_xgb_gs = roc_auc_score(y_train_over, y_train_pred_proba_xgb_gs)

print("\nHasil Evaluasi Model Terbaik (Grid Search XGBoost) pada Data Latih:")
print(f"- Akurasi: {accuracy_train_xgb_gs}")
print(f"- ROC-AUC Score: {roc_auc_train_xgb_gs}")


# --- Evaluasi Model Terbaik dari data uji ---
# Lakukan prediksi dengan data testing menggunakan model terbaik
y_test_pred_proba_xgb_gs = best_model_xgb.predict_proba(X_test)[:, 1]
threshold = 0.5
y_test_pred_xgb_gs = (y_test_pred_proba_xgb_gs >= threshold).astype(int) # Menggunakan threshold

# Evaluasi model terbaik
accuracy_test_xgb_gs = accuracy_score(y_test, y_test_pred_xgb_gs)
roc_auc_test_xgb_gs = roc_auc_score(y_test, y_test_pred_proba_xgb_gs)

# Catatan hasil eksekusi dan parameter model terbaik
print("\nHasil Grid Search - XGBoost:")
print(f"- Parameter terbaik: {grid_search_xgb.best_params_}")
print(f"- ROC-AUC terbaik pada cross-validation: {grid_search_xgb.best_score_}")
print("\nHasil Evaluasi Model Terbaik pada Data Testing:")
print(f"- Akurasi: {accuracy_test_xgb_gs}")
print(f"- ROC-AUC Score: {roc_auc_test_xgb_gs}")
print("- Laporan Klasifikasi:")
print(classification_report(y_test, y_test_pred_xgb_gs))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Definisikan daftar dari model prediksi dan nama-nama mereka
# Pastikan variabel prediksi tersebut sesuai dengan output dari sell evaluasi model
model_predictions = [
    (y_test_pred_lr_gs, 'Logistic Regression GS'), # Menggunakan prediksi LR dituning
    (y_test_pred_xgb_gs, 'XGBoost GS'),           # Menggunakan prediksi XGB dituning
    (y_test_pred, 'Logistic Regression'),   # Menggunakan prediksi LR awal/asal
    (y_test_pred, 'XGBoost')                 # Menggunakan prediksi XGB awal/asal 
]

# Lakukan loop/iterasi melalui prediksi dan plot confusion matrix
for predictions, model_name in model_predictions:
    # Membuat confusion metrik
    cm = confusion_matrix(y_test, predictions)
    print(f"\nConfusion Matrix - {model_name}:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# --- Membandingkan akurasi (Mengasumsi akurasi variabel-variabel tersedia) ---
# Kamu akan membutuhkan untuk memastikan akurasi variabel (accuracy_test, accuracy_test_xgb, accuracy_test_lr_gs, accuracy_test_xgb_gs)
# Sel-sel dari evaluasi sebelumnya masih berada dalam namespace kernel. 

# Buat sebuah chart bar untuk membandingkan akurasi
# Mendefinisikan nama models sebagai nilai x pada chart bar
models = ['Logistic Regression', 'XGBoost', 'Logistic Regression GS', 'XGBoost GS']
#
# Mendefinisikan nilai akurasi sebagai nilai y pada chart bar
# Pastikan variabel-variabel tersebut diisi dengan benar dari sel sebelumnya
accuracies = [accuracy_test, accuracy_test, accuracy_test_lr_gs, accuracy_test_xgb_gs] # Catatan: Pastikan bahwa accuracy_test dari LR awal digunakan.
plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'orange', 'salmon'])
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1) # Atur y-axis batas dari 0 hingga 1 untuk accuracy
plt.xticks(rotation=45, ha='right') # Putar label untuk meningkatkan keterbacaan
plt.tight_layout() # Sesuaikan tata letak untuk mencegah label tumpang tindih.
plt.show()

