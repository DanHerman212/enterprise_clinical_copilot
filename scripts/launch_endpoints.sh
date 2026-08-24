#!/usr/bin/env bash
# launch_endpoints.sh — stand the demo serving endpoints back up IN PARALLEL.
#
# Original lived at /tmp/launch_endpoints.sh (Aug 19); persisted here 2026-08-24
# after it kept being forgotten. Never store this only in /tmp again.
#
# Prediction endpoint:
#   projects/mlops/scripts/deploy_cpr.py
#     → reuses the cached CPR image (content-hash tag), auto-discovers the
#       newest readmission-final-* bundle, deploys to readmission-endpoint
#       (n1-standard-2, ~5-10 min).
# RAG index endpoint:
#   projects/agent-harness/scripts/deploy_synthetic_rag.py
#     → auto-picks the NEWEST rag-tree-ah-* index (the synthetic demo cohort,
#       ~560 vectors — NOT the 555k real corpus), deploys it to
#       readmission-rag-index on a cheap e2-standard-2 (~$0.09/hr).
#
# NOTE: do NOT call deploy_rag_endpoint.py directly without INDEX_ID /
# INDEX_MACHINE_TYPE — its defaults are the 555k real index on e2-standard-16
# (~$270/mo). deploy_synthetic_rag.py sets the safe values for us.
#
# Logs: /tmp/deploy_cpr.log and /tmp/deploy_rag.log
set -u
REPO=/Users/danherman/Desktop/enterprise_clinical_copilot
PYTHON="$REPO/.venv/bin/python"
export PYTHONWARNINGS=ignore::FutureWarning
cd "$REPO"

echo "[$(date +%H:%M:%S)] launching prediction endpoint deploy (deploy_cpr.py)"
"$PYTHON" -u projects/mlops/scripts/deploy_cpr.py > /tmp/deploy_cpr.log 2>&1 &
P1=$!

echo "[$(date +%H:%M:%S)] launching RAG index endpoint deploy (deploy_synthetic_rag.py)"
"$PYTHON" -u projects/agent-harness/scripts/deploy_synthetic_rag.py > /tmp/deploy_rag.log 2>&1 &
P2=$!

echo "[$(date +%H:%M:%S)] both launched (cpr=$P1 rag=$P2) — waiting..."
wait "$P1"; C1=$?
wait "$P2"; C2=$?
echo "[$(date +%H:%M:%S)] DONE: cpr exit=$C1  rag exit=$C2"
echo "--- deploy_cpr.log (tail) ---"
tail -4 /tmp/deploy_cpr.log
echo "--- deploy_rag.log (tail) ---"
tail -4 /tmp/deploy_rag.log
