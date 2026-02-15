import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
df = pd.read_csv("heart.csv")

# Identify numeric and categorical columns
X = df.drop(columns=['id', 'dataset', 'num'], errors='ignore')
y = df['num'].apply(lambda x: 1 if x > 0 else 0)

numeric_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(exclude=['number']).columns

# Fill missing values
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
X[categorical_cols] = X[categorical_cols].fillna('Unknown')

# One-hot encode
X = pd.get_dummies(X)
feature_cols = X.columns.tolist()

# Scale and Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "features.pkl")
print("✅ Files generated: model.pkl, scaler.pkl, features.pkl")