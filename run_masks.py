import subprocess
import os

print("Running Original Baseline...")
subprocess.run([
    "python", "recon_code.py", 
    "--data_opt", "AXFLAIR", 
    "--data_dir", "data/processed_mat_201_6002867/file_brain_AXFLAIR_201_6002867_slice000_R4_ACS24.mat"
], env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"))

print("Running Frequency-Balanced Multi-Mask...")
subprocess.run([
    "python", "recon_code.py", 
    "--data_opt", "AXFLAIR", 
    "--data_dir", "data/processed_mat_201_6002867/file_brain_AXFLAIR_201_6002867_slice000_R4_ACS24.mat",
    "--use_multi_masks",
    "--multi_mask_mode", "frequency_balanced"
], env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"))
