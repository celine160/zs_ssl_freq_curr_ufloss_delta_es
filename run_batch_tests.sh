#!/bin/bash

# Ensure Conda is available in the script
eval "$(conda shell.bash hook)"
conda activate zs_ssl_new

DATA_DIR="data/processed_data_19_7"

echo "Spawning parallel training jobs across GPUs 1, 2, and 3..."

# Generate a fixed array of random seeds for each file to ensure identical initialization
# across all 3 GPUs for the same file, while maintaining randomness across different files.
SEED_ARRAY=()
FILES=("$DATA_DIR"/*.mat)
for i in "${!FILES[@]}"; do
  SEED_ARRAY[$i]=$RANDOM
done

# =========================================================================
# GPU 1: Baseline
# =========================================================================
(
  for i in "${!FILES[@]}"; do
      MAT_FILE="${FILES[$i]}"
      SEED="${SEED_ARRAY[$i]}"
      echo "[GPU 1] Baseline -> $MAT_FILE (Seed: $SEED)"
      CUDA_VISIBLE_DEVICES=1 python recon_code.py --data_opt AXFLAIR --data_dir "$MAT_FILE" --seed $SEED --use_delta_es > /dev/null 2>&1
  done
  echo "[GPU 1] Baseline fully completed!"
) &

# =========================================================================
# GPU 2: Curriculum Only
# =========================================================================
(
  for i in "${!FILES[@]}"; do
      MAT_FILE="${FILES[$i]}"
      SEED="${SEED_ARRAY[$i]}"
      echo "[GPU 2] Curriculum Only -> $MAT_FILE (Seed: $SEED)"
      CUDA_VISIBLE_DEVICES=2 python recon_code.py --data_opt AXFLAIR --data_dir "$MAT_FILE" --use_multi_masks --multi_mask_mode frequency_curriculum --multi_mask_k 25 --seed $SEED --use_delta_es > /dev/null 2>&1
  done
  echo "[GPU 2] Curriculum Only fully completed!"
) &

# =========================================================================
# GPU 3: Curriculum + UFLoss 0.5
# =========================================================================
(
  for i in "${!FILES[@]}"; do
      MAT_FILE="${FILES[$i]}"
      SEED="${SEED_ARRAY[$i]}"
      echo "[GPU 3] Curriculum + UFLoss 0.5 -> $MAT_FILE (Seed: $SEED)"
      CUDA_VISIBLE_DEVICES=3 python recon_code.py --data_opt AXFLAIR --data_dir "$MAT_FILE" --use_multi_masks --multi_mask_mode frequency_curriculum --multi_mask_k 25 --lambda_uf 0.5 --seed $SEED --use_delta_es > /dev/null 2>&1
  done
  echo "[GPU 3] Curriculum + UFLoss fully completed!"
) &

echo "All 3 GPU jobs have been dispatched to the background!"
wait
echo "ALL BATCH JOBS COMPLETED SUCCESSFULLY!"
