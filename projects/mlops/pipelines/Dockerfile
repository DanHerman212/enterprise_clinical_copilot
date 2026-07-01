# Use the Vertex AI pre-built XGBoost training image as the base.
# This gives us Google-verified compatible versions of xgboost,
# scikit-learn, pandas, pyarrow, google-cloud-*, and libgomp1.
# We add only the four packages not included: shap, evidently,
# optuna, matplotlib, and joblib.
FROM us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-7:latest

RUN pip install --no-cache-dir \
    shap>=0.45 \
    evidently>=0.4 \
    optuna>=3.6 \
    matplotlib>=3.9 \
    joblib>=1.4
