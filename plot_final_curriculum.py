import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models import UnrollNet
from models import utils
from models.parser_ops import get_parser

def main():
    parser = get_parser()
    args = parser.parse_args([])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. Load Data & Normalize with np.max ---
    data_path = 'data/processed_mat_201_6002867/file_brain_AXFLAIR_201_6002867_slice000_R4_ACS24.mat'
    data = sio.loadmat(data_path)
    kspace_test, sens_maps, original_mask = data['kspace'], data['sens_maps'], data['mask']
    args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB = kspace_test.shape
    
    kspace_test = kspace_test / np.max(np.abs(kspace_test[:]))
    
    test_mask = np.complex64(original_mask)
    if args.data_opt == 'Coronal_PD':
        test_mask[:, 0:17] = np.ones((args.nrow_GLOB, 17))
        test_mask[:, 352:args.ncol_GLOB] = np.ones((args.nrow_GLOB, 16))
        
    ref_image = utils.sense1(kspace_test, sens_maps)
    nw_input = utils.sense1(kspace_test * np.tile(test_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB)), sens_maps)
    
    nw_input_tensor = utils.complex2real(nw_input[np.newaxis])
    nw_input_tensor = torch.from_numpy(nw_input_tensor).permute(0,3,1,2).to(device)
    test_mask_tensor = torch.from_numpy(test_mask[np.newaxis]).to(device)
    sens_maps_tensor = torch.from_numpy(np.transpose(sens_maps[np.newaxis], (0, 3, 1, 2))).to(device)
    
    # --- 2. Load the Best Model ---
    model_dir = 'saved_models_freq_curr_17_7/ZS_SSL_Model_300Epochs_Rate4_10Unrolls_Original_No_UFLoss_Multimask_frequency_curriculum_K25'
    model_path = os.path.join(model_dir, 'best.pth')
    
    model = UnrollNet.UnrolledNet(args, device=device).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    with torch.no_grad():
        dummy_loss = torch.zeros_like(test_mask_tensor)
        nw_output, _, _ = model(nw_input_tensor, test_mask_tensor, dummy_loss, sens_maps_tensor)
        nw_output = nw_output.cpu().numpy()
        
    recon_img = nw_output[0, 0, :, :] + 1j * nw_output[0, 1, :, :]
    
    psnr = peak_signal_noise_ratio(np.abs(ref_image), np.abs(recon_img), data_range=np.max(np.abs(ref_image)))
    ssim = structural_similarity(np.abs(ref_image), np.abs(recon_img), data_range=np.max(np.abs(ref_image)))
    
    # --- 3. Parse Training Log for Loss Curve ---
    log_file = os.path.join(model_dir, 'training_log.txt')
    epochs, trn_losses, val_losses = [], [], []
    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch:' in line and 'Val Total:' in line:
                parts = line.split('|')
                ep = int(parts[0].split(':')[1].strip())
                trn = float(parts[2].split(':')[1].strip())
                val = float(parts[-1].split(':')[1].strip())
                epochs.append(ep)
                trn_losses.append(trn)
                val_losses.append(val)
                
    # --- 4. Plot Everything ---
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Loss Curve
    ax1 = plt.subplot(2, 2, (1, 2))
    ax1.plot(epochs, trn_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('Frequency Curriculum Learning Curve', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Reconstructed Image
    ax2 = plt.subplot(2, 2, 3)
    im2 = ax2.imshow(np.abs(recon_img), cmap='gray', vmax=np.max(np.abs(ref_image)))
    ax2.set_title(f'Curriculum Reconstruction\nPSNR: {psnr:.2f} | SSIM: {ssim:.4f}', fontsize=14)
    ax2.axis('off')
    
    # Plot 3: Error Map
    ax3 = plt.subplot(2, 2, 4)
    error_map = np.abs(np.abs(ref_image) - np.abs(recon_img))
    im3 = ax3.imshow(error_map, cmap='jet', vmin=0, vmax=np.max(np.abs(ref_image))*0.2)
    ax3.set_title('Absolute Error Map', fontsize=14)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('zs_ssl_ufloss_gridsearch/frequency_curriculum_final_results.png', dpi=150, bbox_inches='tight')
    print("Saved final results plot to zs_ssl_ufloss_gridsearch/frequency_curriculum_final_results.png")

if __name__ == '__main__':
    main()
