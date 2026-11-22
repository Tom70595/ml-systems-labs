#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python lab06/train.py --config lab06/configs/base.yaml --tag base
python lab06/profile_epoch.py --config lab06/configs/base.yaml
python lab06/study_workers.py --config lab06/configs/base.yaml
python lab06/study_batch.py --config lab06/configs/base.yaml --epochs 5
python lab06/study_optim.py --config lab06/configs/base.yaml --epochs 10
echo "All Lab 6 runs complete."