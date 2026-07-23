import os
import re
import matplotlib.pyplot as plt

def plot_log(log_path, title, save_name):
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return

    epochs = []
    val_loss = []
    psnrs = []
    ssims = []
    nmses = []

    with open(log_path, 'r') as f:
        for line in f:
            if "PSNR" in line:
                # Extract numbers using regex
                # Example: Epoch: 43 | ... | Val Total: 0.4052 | PSNR: 42.7958 | SSIM: 0.9843 | NMSE: 0.001420
                ep = int(re.search(r'Epoch:\s*(\d+)', line).group(1))
                vl = float(re.search(r'Val Total:\s*([\d\.]+)', line).group(1))
                psnr = float(re.search(r'PSNR:\s*([\d\.]+)', line).group(1))
                ssim = float(re.search(r'SSIM:\s*([\d\.]+)', line).group(1))
                nmse_match = re.search(r'NMSE:\s*([\d\.]+)', line)
                nmse = float(nmse_match.group(1)) if nmse_match else 0.0

                epochs.append(ep)
                val_loss.append(vl)
                psnrs.append(psnr)
                ssims.append(ssim)
                nmses.append(nmse)

    if not epochs:
        print("No valid lines found with PSNR/SSIM metrics.")
        return

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Validation Loss', color=color)
    ax1.plot(epochs, val_loss, color=color, label='Validation Loss', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('True PSNR', color=color)  
    ax2.plot(epochs, psnrs, color=color, label='PSNR', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    # Mark the PSNR Peak
    best_psnr_idx = psnrs.index(max(psnrs))
    best_psnr_epoch = epochs[best_psnr_idx]
    best_psnr_val = psnrs[best_psnr_idx]
    ax2.plot(best_psnr_epoch, best_psnr_val, 'bo', markersize=10)
    ax2.annotate(f'Peak: {best_psnr_val:.2f} (Ep {best_psnr_epoch})', 
                 (best_psnr_epoch, best_psnr_val), textcoords="offset points", xytext=(0,10), ha='center', color='blue', weight='bold')

    plt.title(f'Validation Loss vs. True PSNR\n{title}')
    fig.tight_layout()  
    plt.grid(alpha=0.3)
    plt.savefig(save_name, dpi=300)
    print(f"Saved plot to {save_name}")

if __name__ == '__main__':
    log1 = 'saved_models_freq_curr_17_7/ZS_SSL_Model_300Epochs_Rate4_10Unrolls_UFLoss_0.5_Multimask_frequency_curriculum_K25_Tight/training_log.txt'
    plot_log(log1, 'UFLoss 0.5 + Curriculum', 'val_vs_psnr_ufloss05.png')
    
    log2 = 'saved_models_freq_curr_17_7/ZS_SSL_Model_300Epochs_Rate4_10Unrolls_Original_No_UFLoss_Tight/training_log.txt'
    plot_log(log2, 'Baseline ZS-SSL', 'val_vs_psnr_baseline.png')
