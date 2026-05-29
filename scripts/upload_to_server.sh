#!/bin/bash
# Upload gptq_plus codebase to server via rsync
# Excludes: outputs, datasets, cache, venv, pycache, IDE/editor files, PDFs

set -e

LOCAL_DIR=${LOCAL_DIR:-/mnt/d/gptq_plus}
SERVER_USER=${SERVER_USER:-root}
SERVER_HOST=${SERVER_HOST:-your.server.ip}
SERVER_DIR=${SERVER_DIR:-/root/lh/gptq_plus}

echo "Uploading ${LOCAL_DIR} -> ${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}"
echo ""

rsync -avz --progress \
    --exclude='outputs/' \
    --exclude='datasets/' \
    --exclude='cache/' \
    --exclude='.venv_wsl/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.idea/' \
    --exclude='.claude/' \
    --exclude='papers/' \
    --exclude='论文参考/' \
    --exclude='${OUT}/' \
    --exclude='*.pdf' \
    --exclude='*.png' \
    --exclude='ablation_results.md' \
    --exclude='formal_baseline_results*.md' \
    --exclude='formal_baseline_commands.md' \
    --exclude='memory_profile.md' \
    --exclude='qwen3_0p6b_refined_lr_results.md' \
    "${LOCAL_DIR}/" \
    "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/"

echo ""
echo "Done. Code is at ${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}"
