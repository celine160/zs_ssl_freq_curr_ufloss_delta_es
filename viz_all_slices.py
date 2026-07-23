import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

def ifft2c(kspace):
    """Centered 2D inverse FFT."""
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )

def rss_combine(coil_images):
    """Root-sum-of-squares coil combination."""
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))

# We will use the first test file
h5_path = "data/data_19_7/file_brain_AXFLAIR_200_6002629.h5"

with h5py.File(h5_path, 'r') as f:
    kspace = f['kspace'][()]

print(f"K-space shape: {kspace.shape}")
num_slices = kspace.shape[0]

# Calculate grid size
cols = 4
rows = int(np.ceil(num_slices / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 4))
axes = axes.flatten()

for i in range(num_slices):
    slice_kspace = kspace[i]
    slice_img_coils = ifft2c(slice_kspace)
    slice_img_rss = rss_combine(slice_img_coils)
    
    # Normalize for display
    display_img = np.abs(slice_img_rss)
    vmax = np.percentile(display_img, 99)
    
    axes[i].imshow(display_img, cmap='gray', vmax=vmax, vmin=0)
    axes[i].set_title(f"Slice {i}")
    axes[i].axis('off')

for j in range(num_slices, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("all_slices_viz_test.png", dpi=150)
print("Saved all_slices_viz_test.png")
