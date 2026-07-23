#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate zs_ssl_new

OUTPUT_DIR="data/processed_mat_201_datafiles"

echo "Processing first 4 files from data/201_datafiles..."

python preprocess_h5_to_mat --input data/201_datafiles/file_brain_AXFLAIR_201_6002871.h5 --output_dir $OUTPUT_DIR --slice_idx 0
python preprocess_h5_to_mat --input data/201_datafiles/file_brain_AXFLAIR_201_6002876.h5 --output_dir $OUTPUT_DIR --slice_idx 0
python preprocess_h5_to_mat --input data/201_datafiles/file_brain_AXFLAIR_201_6002878.h5 --output_dir $OUTPUT_DIR --slice_idx 0
python preprocess_h5_to_mat --input data/201_datafiles/file_brain_AXFLAIR_201_6002881.h5 --output_dir $OUTPUT_DIR --slice_idx 0

echo "Processing complete! Output saved to $OUTPUT_DIR"
