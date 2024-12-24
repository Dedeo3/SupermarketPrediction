import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE  # Untuk menangani ketidakseimbangan kelas
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
dataset = 'supermarket_sales.csv'
read = pd.read_csv(dataset)

# Fungsi untuk mengklasifikasikan Rating
def classify_rating(value):
    if value <= 2.5:
        return "Buruk"
    elif 2.6 <= value <= 4.0:
        return "Biasa saja"
    else:
        return "Baik"

# Terapkan fungsi classify_rating ke kolom 'Rating'
read['RatingCategory'] = read['Rating'].apply(classify_rating)

# Cek distribusi RatingCategory
print("Distribusi RatingCategory:\n", read['RatingCategory'].value_counts())

# Encoding target RatingCategory
label_encoder = LabelEncoder()
read['RatingCategory_encoded'] = label_encoder.fit_transform(read['RatingCategory'])

# Pisahkan fitur dan target
X = read[['Rating']]  # Fitur
y = read['RatingCategory_encoded']  # Target

# Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling ketidakseimbangan kelas dengan SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi model
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))

# Visualisasi distribusi kategori Rating
sns.countplot(x='RatingCategory', data=read)
plt.title("Distribusi Kategori Rating")
plt.show()

# Cek distribusi prediksi
print("Distribusi prediksi:", pd.Series(y_pred).value_counts())
