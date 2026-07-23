#!/bin/bash

# Ensure Conda is available in the script
eval "$(conda shell.bash hook)"
conda activate zs_ssl_new

# Check if an argument was provided
if [ -z "$1" ]; then
    echo "Error: No .mat file provided."
    echo "Usage: ./run_single_test.sh path/to/file.mat"
    exit 1
fi

MAT_FILE="$1"

if [ ! -f "$MAT_FILE" ]; then
    echo "Error: File '$MAT_FILE' does not exist."
    exit 1
fi

# Generate a single random seed for this run so all 4 models initialize identically
RANDOM_SEED=$RANDOM
echo "Using Universal Random Seed: $RANDOM_SEED"

echo "=========================================================="
echo "Starting Parallel Processing For: $MAT_FILE"
echo "=========================================================="

# =========================================================================
# GPU 0: Baseline + UFLoss 0.5
# =========================================================================
(
  echo "[GPU 0] Starting Baseline + UFLoss 0.5..."
  CUDA_VISIBLE_DEVICES=0 python recon_code.py --data_opt AXFLAIR --data_dir "$MAT_FILE" --out_dir "saved_models_20.7" --lambda_uf 0.5 --seed $RANDOM_SEED --use_delta_es --epochs 100 > /dev/null 2>&1
  echo "[GPU 0] Baseline + UFLoss 0.5 completed!"
) &

# =========================================================================
# GPU 1: Baseline
# =========================================================================
(
  echo "[GPU 1] Starting Baseline..."
  CUDA_VISIBLE_DEVICES=1 python recon_code.py --data_opt AXFLAIR --data_dir "$MAT_FILE" --out_dir "saved_models_20.7" --seed $RANDOM_SEED --use_delta_es --epochs 100 > /dev/null 2>&1
  echo "[GPU 1] Baseline completed!"
) &

# =========================================================================
# GPU 2: Curriculum Only
# =========================================================================
(
  echo "[GPU 2] Starting Curriculum Only..."
  CUDA_VISIBLE_DEVICES=2 python recon_code.py --data_opt AXFLAIR --data_dir "$MAT_FILE" --out_dir "saved_models_20.7" --use_multi_masks --multi_mask_mode frequency_curriculum --multi_mask_k 25 --seed $RANDOM_SEED --use_delta_es --epochs 100 > /dev/null 2>&1
  echo "[GPU 2] Curriculum Only completed!"
) &

# =========================================================================
# GPU 3: Curriculum + UFLoss 0.5
# =========================================================================
(
  echo "[GPU 3] Starting Curriculum + UFLoss 0.5..."
  CUDA_VISIBLE_DEVICES=3 python recon_code.py --data_opt AXFLAIR --data_dir "$MAT_FILE" --out_dir "saved_models_20.7" --use_multi_masks --multi_mask_mode frequency_curriculum --multi_mask_k 25 --lambda_uf 0.5 --seed $RANDOM_SEED --use_delta_es --epochs 100 > /dev/null 2>&1
  echo "[GPU 3] Curriculum + UFLoss completed!"
) &

echo "All 4 GPU jobs have been dispatched to the background!"
echo "Waiting for them to finish..."
wait
echo "ALL TESTS COMPLETED SUCCESSFULLY FOR $MAT_FILE!"
