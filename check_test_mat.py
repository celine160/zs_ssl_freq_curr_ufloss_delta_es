import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def ifft2(x):
    """Centered 2D inverse FFT."""
    return np.fft.fftshift(
        np.fft.ifft2(
            np.fft.ifftshift(x, axes=(-2, -1)),
            axes=(-2, -1),
            norm="ortho",
        ),
        axes=(-2, -1),
    )

mat_file = "data/processed_test_data/file_brain_AXFLAIR_200_6002441_slice000_R4_ACS24.mat"
data = scipy.io.loadmat(mat_file)

kspace_full = data['kspace']
sens_maps = data['sens_maps']

print(f"K-space full shape: {kspace_full.shape}")
print(f"Sens maps shape: {sens_maps.shape}")

# Reconstruct Reference Image using SENSE combination
image_coils = ifft2(kspace_full)
ref_image_sense = np.sum(np.conj(sens_maps) * image_coils, axis=0) # SENSE combination

# Reconstruct Reference Image using RSS combination
ref_image_rss = np.sqrt(np.sum(np.abs(image_coils)**2, axis=0))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.abs(ref_image_sense), cmap='gray')
axes[0].set_title("SENSE Combined Ref Image")
axes[0].axis('off')

axes[1].imshow(ref_image_rss, cmap='gray')
axes[1].set_title("RSS Combined Ref Image")
axes[1].axis('off')

plt.savefig("test_mat_viz.png", dpi=150)
print("Saved test_mat_viz.png")
