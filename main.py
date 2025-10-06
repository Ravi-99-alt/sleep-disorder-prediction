import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load dataset
df = pd.read_csv("Sleep_Data.csv")

# Rename columns to match expected format
columns = [
    "Person ID", "Gender", "Age", "Occupation", "Sleep Duration",
    "Quality of Sleep", "Physical Activity Level", "Stress Level",
    "BMI Category", "Blood Pressure", "Heart Rate", "Daily Steps",
    "Sleep Disorder"
]
df.columns = columns

# Drop irrelevant column
df = df.drop(columns=["Person ID"])

# Optional: Split Blood Pressure into Systolic and Diastolic
def split_blood_pressure(bp):
    try:
        systolic, diastolic = map(int, bp.split("/"))
        return systolic, diastolic
    except:
        return np.nan, np.nan

df[["Systolic_BP", "Diastolic_BP"]] = df["Blood Pressure"].apply(
    lambda x: pd.Series(split_blood_pressure(x))
)

# Drop original BP column
df = df.drop(columns=["Blood Pressure"])

# Convert numeric columns safely
numeric_cols = [
    "Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
    "Stress Level", "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values
df = df.dropna().reset_index(drop=True)

# Encode categorical features
gender_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])

occupation_encoder = LabelEncoder()
df['Occupation'] = occupation_encoder.fit_transform(df['Occupation'])

bmi_category_encoder = LabelEncoder()
df['BMI Category'] = bmi_category_encoder.fit_transform(df['BMI Category'])

target_encoder = LabelEncoder()
df['Sleep Disorder'] = target_encoder.fit_transform(df['Sleep Disorder'])

# Save encoders
os.makedirs("models", exist_ok=True)
joblib.dump(gender_encoder, 'models/gender_encoder.pkl')
joblib.dump(occupation_encoder, 'models/occupation_encoder.pkl')
joblib.dump(bmi_category_encoder, 'models/bmi_category_encoder.pkl')
joblib.dump(target_encoder, 'models/target_encoder.pkl')

# Features and Target
X = df[[
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
    'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps',
    'BMI Category', 'Systolic_BP', 'Diastolic_BP'
]]
y = df['Sleep Disorder']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.pkl')

# Balance classes using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/model.pkl')

# Evaluate
acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Model trained with Accuracy: {acc * 100:.2f}%")
print("✅ Classification Report:\n", classification_report(y_test, model.predict(X_test)))

# Optional: Print label mapping
print("\n🔢 Sleep Disorder Class Mapping:")
for i, label in enumerate(target_encoder.classes_):
    print(f"{i} => {label}")

# Optional: Show sample data for verification
print("\n📌 First 5 Rows of Cleaned Data:")
print(df.head())

print("\n🧮 Sleep Disorder Class Distribution:")
print(df['Sleep Disorder'].value_counts())