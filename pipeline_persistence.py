import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Generate Synthetic Dataset
X, y = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=6, 
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Build Scikit-Learn Pipeline (Preprocessing + Estimator)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf_classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
])

# 3. Fit Pipeline on Training Data
print("Fitting complete end-to-end pipeline...")
pipeline.fit(X_train, y_train)

# 4. Save Model Artifact to Disk
model_filename = 'rf_pipeline.joblib'
joblib.dump(pipeline, model_filename)
print(f"✓ Trained pipeline saved to disk as '{model_filename}'")

# 5. Simulate Production Inference by Reloading Saved Model
print("\n--- Production Inference Simulation ---")
loaded_pipeline = joblib.load(model_filename)
print("✓ Loaded saved pipeline successfully.")

# Run Inference on Unseen Test Data using Loaded Artifact
predictions = loaded_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"✓ Production Model Test Accuracy: {accuracy:.4f}")