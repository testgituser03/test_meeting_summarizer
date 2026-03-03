#!/usr/bin/env bash
# run_app.sh — Launch the Meeting Summarizer Streamlit demo.
#
# Activates the project venv, sets MPS fallback, and starts the server.
# Usage:
#   bash scripts/run_app.sh
#   bash scripts/run_app.sh --server.port 8502   # custom port

set -euo pipefail

source ~/.venvs/meeting-summarizer/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1

streamlit run scripts/app.py \
    --server.port 8501 \
    --server.headless false \
    "$@"
