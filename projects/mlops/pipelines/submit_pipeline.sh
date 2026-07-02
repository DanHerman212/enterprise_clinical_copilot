#!/bin/bash
set -e

echo "Setting up Python 3.11 environment for KFP submission..."
if [ -d "/Users/danherman/Desktop/enterprise_clinical_copilot/.venv-311" ]; then
    source /Users/danherman/Desktop/enterprise_clinical_copilot/.venv-311/bin/activate
else
    # Assume python3.11 is available
    python3.11 -m venv /Users/danherman/Desktop/enterprise_clinical_copilot/.venv-311
    source /Users/danherman/Desktop/enterprise_clinical_copilot/.venv-311/bin/activate
    pip install kfp==2.16.1 google-cloud-aiplatform pandas scikit-learn xgboost shap evidently joblib pyarrow
fi

export PYTHONPATH="/Users/danherman/Desktop/enterprise_clinical_copilot/projects/mlops:$PYTHONPATH"

echo "Submitting pipeline to Vertex AI..."
python /Users/danherman/Desktop/enterprise_clinical_copilot/projects/mlops/pipelines/training_pipeline.py submit
echo "Submission script complete."
