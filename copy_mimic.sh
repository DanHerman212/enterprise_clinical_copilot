#!/bin/bash
PROJECT="trim-icon-498815-a0"

echo "Creating datasets..."
bq mk --dataset --project_id=$PROJECT $PROJECT:mimiciv_3_1_hosp || true
bq mk --dataset --project_id=$PROJECT $PROJECT:mimiciv_ed || true

TABLES=(
  "mimiciv_3_1_hosp.admissions"
  "mimiciv_3_1_hosp.patients"
  "mimiciv_3_1_hosp.diagnoses_icd"
  "mimiciv_3_1_hosp.procedures_icd"
  "mimiciv_3_1_hosp.prescriptions"
  "mimiciv_3_1_hosp.labevents"
  "mimiciv_ed.edstays"
)

echo "Copying tables..."
for TABLE in "${TABLES[@]}"; do
  echo "Copying physionet-data:$TABLE to $PROJECT:$TABLE..."
  bq cp --force --project_id=$PROJECT physionet-data:$TABLE $PROJECT:$TABLE
done
echo "All tables copied successfully!"
