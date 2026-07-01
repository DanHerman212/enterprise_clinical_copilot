# Extend the Vertex AI pre-built XGBoost training container.
# This gives us Google-verified compatible versions of xgboost,
# scikit-learn, pandas, pyarrow, and all Google Cloud client
# libraries.  We add only the packages not included: shap,
# evidently, optuna, matplotlib, and joblib.
FROM us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest

RUN pip install --no-cache-dir \
    shap>=0.45 \
    evidently==0.4.34 \
    optuna>=3.6 \
    matplotlib>=3.9 \
    joblib>=1.4 \
    "google-cloud-bigquery[pandas]>=3.20" \
    google-cloud-bigquery-storage>=2.25
