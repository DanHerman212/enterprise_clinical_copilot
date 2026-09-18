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
#   scripts/agent/build_rag_image.sh   (RUN FIRST, synchronously)
#     → rebuilds the `rag-ingest` image, which is the component the gate
#       measures an index with. It is built here rather than left to memory
#       because on 2026-09-18 the deployed image was four weeks older than the
#       source: it predated the `empty_result_rate` metric, so the recall job
#       could not report a threshold the gate enforced, and the promotion was
#       refused while saying the corpus had failed. It had not; nothing had been
#       measured. Building here means the gate always measures with the code in
#       this checkout.
#   scripts/agent/deploy_rag.py
#     → auto-picks the NEWEST rag-tree-ah-* index (the demo cohort, ~454
#       vectors — NOT the 555k real corpus), refuses anything above the
#       synthetic-scale limit, deploys it under a staging id on the machine the
#       corpus config names, and promotes it to the live id only if the recall
#       report passes. RAG_CORPUS=demo below is what selects the cheap
#       e2-standard-2 (~$0.09/hr); the config's active corpus is the real one,
#       which would size this at e2-standard-16 (~$0.38/hr).
#     → REUSES an existing measurement when the index contents and the
#       thresholds are unchanged, so standing the endpoints back up does not
#       re-run a 20-minute recall job to re-derive a verdict already on file.
#       Pass --force-measure to scripts/agent/deploy_rag.py to insist on a fresh
#       one.
#
# Logs: /tmp/deploy_cpr.log, /tmp/build_rag_image.log and /tmp/deploy_rag.log
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

# The image build is the RAG deploy's precondition, so it runs to completion
# before that deploy starts — and in parallel with the CPR deploy, which does
# not depend on it. A build failure stops the RAG half rather than letting it
# measure with whatever image happens to be tagged `latest`.
echo "[$(date +%H:%M:%S)] building the RAG ingest image (build_rag_image.sh)"
if bash scripts/agent/build_rag_image.sh > /tmp/build_rag_image.log 2>&1; then
  echo "[$(date +%H:%M:%S)] image built — launching RAG index endpoint deploy (deploy_rag.py)"
  "$PYTHON" -u scripts/agent/deploy_rag.py "$@" > /tmp/deploy_rag.log 2>&1 &
  P2=$!
else
  echo "[$(date +%H:%M:%S)] IMAGE BUILD FAILED — not deploying the RAG index. Last lines:"
  tail -15 /tmp/build_rag_image.log
  P2=""
  C2=1
fi

if [ -n "$P2" ]; then
  echo "[$(date +%H:%M:%S)] both launched (cpr=$P1 rag=$P2) — waiting..."
  wait "$P1"; C1=$?
  wait "$P2"; C2=$?
else
  wait "$P1"; C1=$?
fi
echo "[$(date +%H:%M:%S)] DONE: cpr exit=$C1  rag exit=$C2"
echo "--- deploy_cpr.log (tail) ---"
tail -4 /tmp/deploy_cpr.log
if [ -n "$P2" ]; then
  echo "--- deploy_rag.log (tail) ---"
  tail -6 /tmp/deploy_rag.log
else
  echo "--- build_rag_image.log (tail) ---"
  tail -6 /tmp/build_rag_image.log
fi
exit $(( C1 != 0 || C2 != 0 ))
