# Data and Model Lifecycle

## Data representation

Dataform in `definitions/` builds the cohort, label, patient-level splits, feature families,
and encoded mart in BigQuery. Split assignment is deterministic by patient identifier and
a Dataform assertion rejects patients appearing in more than one split.

`mlops/data/encoding.py` is the feature-contract source for the fixed-order numeric vector.
Categorical encoding is materialized in BigQuery; numeric nulls remain available to native
XGBoost missing-value handling. The same feature order and parent-group mapping are carried
into the serving bundle for TreeSHAP aggregation.

## Training

`mlops/training/` contains the Vertex AI pipeline and components. The workflow validates
data, gates a benchmark against the implemented HOSPITAL baseline, tunes XGBoost with
Optuna, calibrates the operating threshold, evaluates the held-out test split, produces
SHAP and fairness artifacts, and registers the model only after the required stages pass.

## Serving

`mlops/serving/` contains the Custom Prediction Routine and deployment scripts. The serving
bundle contains the native booster, feature manifest, decision threshold, checksums, and
gate metrics. The endpoint returns a probability, threshold decision, and parent-aggregated
TreeSHAP factors.

## Retrieval

`services/mcp/pipelines/` chunks and embeds discharge notes before indexing them in Vector
Search. `services/mcp/retrieval/` contains deterministic parsing, chunking, embedding, and
evaluation helpers. The retrieval tool resolves original passage text from BigQuery and
never treats an embedding score as clinical evidence.

## Data boundary

The public demonstration uses synthetic notes and synthetic structured features. MIMIC-IV
records are used for training and evaluation only under the applicable data use agreement.
