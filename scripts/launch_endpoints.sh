#!/usr/bin/env bash
# launch_endpoints.sh — stand the demo serving endpoints back up IN PARALLEL.
#
# Original lived at /tmp/launch_endpoints.sh (Aug 19); persisted here 2026-08-24
# after it kept being forgotten. Never store this only in /tmp again.
#
# Prediction endpoint:
#   mlops/serving/deploy_cpr.py
#     → reuses the cached CPR image (content-hash tag), auto-discovers the
#       newest readmission-final-* bundle, deploys to readmission-endpoint
#       (n1-standard-2, ~5-10 min).
# RAG index endpoint:
#   scripts/agent/deploy_rag.py
#     → auto-picks the NEWEST rag-tree-ah-* index (the demo cohort, ~454
#       vectors — NOT the 555k real corpus), refuses anything above the
#       synthetic-scale limit, deploys it under a staging id on the machine the
#       corpus config names, and promotes it to the live id only if the recall
#       report passes. RAG_CORPUS=demo below is what selects the cheap
#       e2-standard-2 (~$0.09/hr); the config's active corpus is the real one,
#       which would size this at e2-standard-16 (~$0.38/hr).
#
# Logs: /tmp/deploy_cpr.log and /tmp/deploy_rag.log
set -u
REPO=/Users/danherman/Desktop/enterprise_clinical_copilot
PYTHON="$REPO/.venv/bin/python"
export PYTHONWARNINGS=ignore::FutureWarning
# The RAG deploy sizes its machine and its recall thresholds from the corpus,
# and the committed default is the real one. This launcher stands up the demo.
export RAG_CORPUS="${RAG_CORPUS:-demo}"
cd "$REPO"

echo "[$(date +%H:%M:%S)] launching prediction endpoint deploy (deploy_cpr.py)"
"$PYTHON" -u mlops/serving/deploy_cpr.py > /tmp/deploy_cpr.log 2>&1 &
P1=$!

echo "[$(date +%H:%M:%S)] launching RAG index endpoint deploy (deploy_rag.py)"
"$PYTHON" -u scripts/agent/deploy_rag.py > /tmp/deploy_rag.log 2>&1 &
P2=$!

echo "[$(date +%H:%M:%S)] both launched (cpr=$P1 rag=$P2) — waiting..."
wait "$P1"; C1=$?
wait "$P2"; C2=$?
echo "[$(date +%H:%M:%S)] DONE: cpr exit=$C1  rag exit=$C2"
echo "--- deploy_cpr.log (tail) ---"
tail -4 /tmp/deploy_cpr.log
echo "--- deploy_rag.log (tail) ---"
tail -4 /tmp/deploy_rag.log
