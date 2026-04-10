import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


print("[LOG] Loading data...")
df = pd.read_csv("heart.csv")

X = df.drop(columns=['id', 'dataset', 'num'], errors='ignore')
y = df['num'].apply(lambda x: 1 if x > 0 else 0)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

# Ensure all categorical features are strings (fixes TypeError in OneHotEncoder)
X[categorical_features] = X[categorical_features].astype(str)

print(f"[LOG] Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

print("[LOG] Tuning model (GridSearchCV)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"[LOG] Model trained. Accuracy: {accuracy:.4f}")
print(f"[LOG] Best params: {grid_search.best_params_}")

combined_data = {
    'model': best_model,
    'metadata': {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_names_in': X.columns.tolist(),
        'accuracy': accuracy,
        'categorical_values': {col: df[col].astype(str).unique().tolist() for col in categorical_features}
    }
}

joblib.dump(combined_data, "model.pkl")
print("[LOG] Unified file saved: model.pkl (Contains both model and metadata)")