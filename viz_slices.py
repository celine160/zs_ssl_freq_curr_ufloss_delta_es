import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Load one of the h5 files
filepath = 'UFLoss/Data/file_brain_AXFLAIR_200_6002425.h5'
with h5py.File(filepath, 'r') as f:
    kspace = f['kspace'][()]

num_slices = kspace.shape[0]

# Reconstruct all slices using Root-Sum-of-Squares (RSS)
# FastMRI data is complex-valued.
# kspace shape: (num_slices, num_coils, height, width)
recons = []
for i in range(num_slices):
    kspace_slice = kspace[i]
    # 2D IFFT for each coil
    k_shifted = np.fft.ifftshift(kspace_slice, axes=(-2, -1))
    im_coils = np.fft.ifft2(k_shifted, axes=(-2, -1))
    im_coils = np.fft.ifftshift(im_coils, axes=(-2, -1))
    
    # RSS coil combination
    rss_image = np.sqrt(np.sum(np.abs(im_coils)**2, axis=0))
    recons.append(rss_image)

# Calculate grid size
cols = 4
rows = int(np.ceil(num_slices / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
axes = axes.flatten()

for i in range(num_slices):
    # Normalize for visualization
    img = recons[i]
    vmax = np.percentile(img, 99)
    axes[i].imshow(img, cmap='gray', vmax=vmax)
    axes[i].set_title(f'Slice {i}')
    axes[i].axis('off')

for i in range(num_slices, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
os.makedirs('/home/celine.abutareef/.gemini/antigravity/brain/f0c51c3e-57e5-4299-b5d3-efff698b2827/artifacts', exist_ok=True)
plt.savefig('/home/celine.abutareef/.gemini/antigravity/brain/f0c51c3e-57e5-4299-b5d3-efff698b2827/artifacts/all_slices_viz.png', bbox_inches='tight', dpi=150)
print(f"Total slices: {num_slices}")
