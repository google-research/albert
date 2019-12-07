#!/bin/bash
# Small integration test script.

set -e
set -x

virtualenv -p python3 .
source ./bin/activate

OUTPUT_DIR_BASE="$(mktemp -d)"
OUTPUT_DIR="${OUTPUT_DIR_BASE}/output"
EXPORT_DIR="${OUTPUT_DIR_BASE}/export"

pip install numpy
pip install -r requirements.txt
python -m run_pretraining_test \
    --output_dir="${OUTPUT_DIR}" \
    --export_dir="${EXPORT_DIR}" \
    --do_train \
    --do_eval \
    --nouse_tpu \
    --train_batch_size=2 \
    --eval_batch_size=1 \
    --max_seq_length=4 \
    --num_train_steps=2 \
    --max_eval_steps=3


