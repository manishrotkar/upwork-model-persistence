# End-to-End Pipeline & Model Persistence (Joblib)

This module demonstrates how to encapsulate data preprocessing (`StandardScaler`) and estimator training (`RandomForestClassifier`) inside an integrated `sklearn.pipeline.Pipeline`, serialize the model object to disk using `joblib`, and reload it to perform production inference.

## Key Features
- **End-to-End Scikit-Learn Pipeline**: Combines feature scaling and classification to eliminate data leakage.
- **Model Serialization**: Uses `joblib.dump()` to produce lightweight binary model artifacts (`rf_pipeline.joblib`).
- **Production Inference**: Uses `joblib.load()` to load serialized pipeline artifacts and execute prediction pipelines without retraining.

## Project Structure
```text
topic6_model_persistence/
├── pipeline_persistence.py
├── rf_pipeline.joblib
├── README.md
└── requirements.txt