# Extend the Vertex AI pre-built XGBoost training container.
# This gives us Google-verified compatible versions of xgboost,
# scikit-learn, pandas, pyarrow, and all Google Cloud client
# libraries.  We add only the packages not included: shap,
# evidently, optuna, matplotlib, joblib, and the GCP clients used
# by the components.  Build + push this, then point components at it
# via TRAINING_IMAGE_URI so runs use pinned deps and skip per-step installs.
FROM us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.2-1:latest

RUN pip install --no-cache-dir \
    "numpy<2" \
    shap>=0.45 \
    evidently==0.4.34 \
    optuna>=3.6 \
    matplotlib>=3.9 \
    joblib>=1.4 \
    pyarrow>=15 \
    "google-cloud-bigquery[pandas]>=3.20" \
    google-cloud-bigquery-storage>=2.25 \
    google-cloud-aiplatform>=1.60
