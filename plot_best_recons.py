import os
import argparse
import re
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
import utils
import UnrollNet
from modules import Dataset_Inference, test

def normalize_for_display(img, norm_factor):
    return np.clip(img / norm_factor, 0, 1)

def compute_nmse(gt, pred):
    return np.linalg.norm(gt - pred)**2 / np.linalg.norm(gt)**2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing the model runs')
    parser.add_argument('--data_file', type=str, default=None, help='Path to the original .mat data file (optional)')
    args = parser.parse_args()
    
    # Locate the original mat file
    if args.data_file and os.path.exists(args.data_file):
        mat_path = args.data_file
    else:
        basename = os.path.basename(args.base_dir.rstrip('/'))
        mat_filename = basename.replace('saved_models_', '') + '.mat'
        
        # Try multiple data directories just in case
        mat_path = None
        possible_dirs = ['data/processed_mat_201_datafiles', 'data/processed_mat_201_6002867']
        for pd in possible_dirs:
            p = os.path.join(pd, mat_filename)
            if os.path.exists(p):
                mat_path = p
                break
                
    if mat_path is None:
        print(f"Could not find original .mat file. Please provide --data_file")
        return
        
    print(f"Loaded {mat_path}")
    data = sio.loadmat(mat_path)
    kspace_train = data['kspace']
    sens_maps = data['sens_maps']
    original_mask = data['mask']
    
    nrow, ncol, ncoil = kspace_train.shape
    
    # Normalize kspace
    kspace_train = kspace_train / np.max(np.abs(kspace_train[:]))
    
    test_mask = np.complex64(original_mask)
    sens_transposed = sens_maps  # Already (640, 320, 24)
    
    nw_input_inference = utils.sense1(kspace_train * np.tile(test_mask[..., np.newaxis], (1, 1, ncoil)), sens_transposed)
    ref_image = utils.sense1(kspace_train, sens_transposed)
    
    ref_abs = np.abs(ref_image)
    disp_norm = np.percentile(ref_abs, 99) if np.percentile(ref_abs, 99) > 1e-8 else ref_abs.max()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    class DummyArgs:
        nb_res_blocks = 15
        nb_unroll_blocks = 10
        CG_Iter = 10
        batchSize = 1
    args_dummy = DummyArgs()
    
    model = UnrollNet.UnrolledNet(args_dummy, device=device).to(device)
    
    sens_for_network = np.transpose(sens_maps, (2, 0, 1))[np.newaxis]
    test_data = Dataset_Inference(utils.complex2real(nw_input_inference[np.newaxis]), test_mask[np.newaxis], test_mask[np.newaxis], sens_for_network)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # We need to evaluate:
    # 1. Baseline best.pth
    # 2. Baseline best_delta.pth
    # 3. Frequency best_delta.pth
    # 4. Frequency+UFLoss best_delta.pth
    
    # Define how to find the directories
    def find_dir(pattern1, pattern2=None, exclude=None):
        for d in os.listdir(args.base_dir):
            if pattern1 in d:
                if pattern2 and pattern2 not in d:
                    continue
                if exclude and exclude in d:
                    continue
                return os.path.join(args.base_dir, d)
        return None
        
    baseline_dir = find_dir('Original_No_UFLoss', exclude='Multimask')
    freq_dir = find_dir('Original_No_UFLoss', pattern2='Multimask')
    freq_ufloss_dir = find_dir('UFLoss_0.5', pattern2='Multimask')
    
    models_to_eval = [
        ('Baseline', baseline_dir, 'best.pth'),
        ('Freq Curriculum', freq_dir, 'best_delta.pth'),
        ('Freq Curr + UFLoss', freq_ufloss_dir, 'best_delta.pth')
    ]
        
    results = {}
    for title, mdir, pth_name in models_to_eval:
        if not mdir:
            print(f"Skipping {title}: directory not found")
            continue
        pth_path = os.path.join(mdir, pth_name)
        if not os.path.exists(pth_path):
            pth_path = os.path.join(mdir, 'best.pth')
            if not os.path.exists(pth_path):
                print(f"Skipping {title}: {pth_name} and best.pth not found")
                continue
            
        print(f"Evaluating {title}...")
        checkpoint = torch.load(pth_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        epoch = checkpoint.get('epoch', -1) + 1  # 0-indexed usually
        
        recon = test(test_loader, model, device)
        recon_complex = utils.real2complex(recon.to('cpu').numpy())
        
        recon_complex = np.squeeze(recon_complex)
        recon_abs = np.abs(recon_complex)
        
        p = psnr(ref_abs, recon_abs, data_range=ref_abs.max())
        s = ssim(ref_abs, recon_abs, data_range=ref_abs.max())
        n = compute_nmse(ref_abs, recon_abs)
        
        results[title] = {
            'epoch': epoch,
            'recon': recon_abs,
            'psnr': p,
            'ssim': s,
            'nmse': n
        }
        
    fig, axes = plt.subplots(2, 4, figsize=(12, 12))
    
    # Target
    axes[0, 0].imshow(normalize_for_display(ref_abs, disp_norm), cmap='gray')
    axes[0, 0].set_title('Target', fontsize=16)
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')  # No error map for target
    
    col = 1
    for title, mdir, pth_name in models_to_eval:
        ax_img = axes[0, col]
        ax_err = axes[1, col]
        
        res = results.get(title)
        if res:
            ax_img.imshow(normalize_for_display(res['recon'], disp_norm), cmap='gray')
            ax_img.set_title(title, fontsize=14)
            ax_img.axis('off')
            
            err = (np.abs(ref_abs - res['recon']) / disp_norm) * 10
            im_err = ax_err.imshow(err, cmap='inferno', vmin=0, vmax=1)
            ax_err.set_title('Error Map (x10)', fontsize=14)
            ax_err.axis('off')
        else:
            ax_img.axis('off')
            ax_err.axis('off')
            ax_img.set_title(f"{title}\nNOT FOUND", fontsize=14)
            
        col += 1
        
    plt.subplots_adjust(left=0.0, right=0.9, bottom=0.0, top=0.88, wspace=0.0, hspace=0.15)
    
    # Add Figure Title
    fig.suptitle("Model Reconstructions Comparison", fontsize=24, y=0.98)
    
    # Add a single colorbar for the error maps on the right side of the bottom row
    # The bottom row goes from y=0 to y~0.36
    cbar_ax = fig.add_axes([0.91, 0.0, 0.015, 0.36])
    fig.colorbar(im_err, cax=cbar_ax)
    
    out_png = os.path.join(args.base_dir, 'all_models_recons_comparison.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved plot to {out_png}")
    plt.close(fig)
    
    # ------------------------------------------------------------------
    # Generate Zoomed ROI Plot
    # ------------------------------------------------------------------
    y1, y2 = 185, 345
    x1, x2 = 70, 250
    
    # Calculate aspect ratio to perfectly glue them
    crop_w = x2 - x1
    crop_h = y2 - y1
    fig_h = 5.0
    fig_w = fig_h * (4 * crop_w) / (2 * crop_h)
    
    fig_z, axes_z = plt.subplots(2, 4, figsize=(fig_w, fig_h))
    
    # Target Zoomed
    axes_z[0, 0].imshow(normalize_for_display(ref_abs[y1:y2, x1:x2], disp_norm), cmap='gray')
    axes_z[0, 0].set_title('Target (Zoom)', fontsize=16)
    axes_z[0, 0].axis('off')
    axes_z[1, 0].axis('off')
    
    # Arrow parameters
    arr_x_zoom = 155 - x1
    arr_y_zoom = 235 - y1
    arrow_kwargs = dict(xy=(arr_x_zoom, arr_y_zoom), xytext=(arr_x_zoom + 25, arr_y_zoom - 25),
                        arrowprops=dict(facecolor='red', edgecolor='red', shrink=0, width=2, headwidth=8))
                        
    axes_z[0, 0].annotate('', **arrow_kwargs)
    
    col = 1
    for title, mdir, pth_name in models_to_eval:
        ax_z_img = axes_z[0, col]
        ax_z_err = axes_z[1, col]
        
        res = results.get(title)
        if res:
            ax_z_img.imshow(normalize_for_display(res['recon'][y1:y2, x1:x2], disp_norm), cmap='gray')
            ax_z_img.set_title(title, fontsize=14)
            ax_z_img.axis('off')
            ax_z_img.annotate('', **arrow_kwargs)
            
            err_z = (np.abs(ref_abs[y1:y2, x1:x2] - res['recon'][y1:y2, x1:x2]) / disp_norm) * 10
            im_z_err = ax_z_err.imshow(err_z, cmap='inferno', vmin=0, vmax=1)
            ax_z_err.set_title('Error Map (x10)', fontsize=14)
            ax_z_err.axis('off')
        else:
            ax_z_img.axis('off')
            ax_z_err.axis('off')
            ax_z_img.set_title(f"{title}\nNOT FOUND", fontsize=14)
            
        col += 1
        
    plt.subplots_adjust(left=0.0, right=0.9, bottom=0.0, top=0.88, wspace=0.0, hspace=0.15)
    
    # Add Colorbar for Zoomed Error Maps
    cbar_ax_z = fig_z.add_axes([0.91, 0.0, 0.015, 0.36])
    fig_z.colorbar(im_z_err, cax=cbar_ax_z)
    
    out_png_z = os.path.join(args.base_dir, 'zoomed_recons_comparison.png')
    plt.savefig(out_png_z, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"Saved zoomed plot to {out_png_z}")
    plt.close(fig_z)

if __name__ == '__main__':
    main()
