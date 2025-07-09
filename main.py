import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("blood_availability.csv")
print("Unique Hospitals:", df["Hospital Name"].unique())

# Encode categorical variables
label_encoders = {}
for col in ["Blood Group", "Hospital Name", "Location"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Get encoded values for hospitals
hn_map = label_encoders["Hospital Name"]
hn_jnm = hn_map.transform(["JNM"])[0]
hn_aiims = hn_map.transform(["AIIMS Kalyani"])[0]

# Force all blood groups at JNM to be Not Available
df.loc[df["Hospital Name"] == hn_jnm, "Units Available"] = 0
df.loc[df["Hospital Name"] == hn_jnm, "Availability"] = 0

# Force all blood groups at AIIMS Kalyani to be Available
df.loc[df["Hospital Name"] == hn_aiims, "Units Available"] = 5
df.loc[df["Hospital Name"] == hn_aiims, "Availability"] = 1

# Print class balance
print("Class Distribution:")
print(df["Availability"].value_counts())

# Feature selection
feature_names = ["Blood Group", "Hospital Name", "Location", "Units Available"]
X = df[feature_names]
y = df["Availability"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "blood_availability_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("✅ Model and encoders saved successfully!")
