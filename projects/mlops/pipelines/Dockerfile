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
    google-cloud-aiplatform>=1.60 \
    kfp==2.16.1

# Bake the project source into the image so the thin @dsl.component wrappers can
# `import` their `run_*` helpers at runtime. KFP lightweight components only
# serialize the decorated function body, so shared module-level code must be
# present on PYTHONPATH inside the container (build context is projects/mlops).
WORKDIR /app
COPY src /app/src
COPY pipelines /app/pipelines
ENV PYTHONPATH=/app
