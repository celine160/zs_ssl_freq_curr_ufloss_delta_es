import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import torch

def get_delta_es_epoch(model_dir):
    pth_path = os.path.join(model_dir, 'best_delta.pth')
    if os.path.exists(pth_path):
        try:
            checkpoint = torch.load(pth_path, map_location='cpu')
            return checkpoint.get('epoch', -1) + 1
        except Exception as e:
            print(f"Error loading {pth_path}: {e}")
    return None

def parse_log(log_path):
    epochs = []
    val_losses = []
    psnrs = []
    ssims = []
    nmses = []
    
    if not os.path.exists(log_path):
        print(f"File not found: {log_path}")
        return epochs, val_losses, psnrs, ssims, nmses
        
    with open(log_path, 'r') as f:
        for line in f:
            if "PSNR" in line:
                ep = int(re.search(r'Epoch:\s*(\d+)', line).group(1))
                vl = float(re.search(r'Val Total:\s*([\d\.]+)', line).group(1))
                psnr = float(re.search(r'PSNR:\s*([\d\.]+)', line).group(1))
                ssim = float(re.search(r'SSIM:\s*([\d\.]+)', line).group(1))
                
                nmse_match = re.search(r'NMSE:\s*([\d\.]+)', line)
                nmse = float(nmse_match.group(1)) if nmse_match else 0.0
                
                epochs.append(ep)
                val_losses.append(vl)
                psnrs.append(psnr)
                ssims.append(ssim)
                nmses.append(nmse)
                
    return epochs, val_losses, psnrs, ssims, nmses

def main():
    parser = argparse.ArgumentParser(description="Plot early stopping metrics.")
    parser.add_argument('--base_dir', type=str, required=True, help="Base directory containing the runs.")
    args = parser.parse_args()
    
    base_dir = args.base_dir
    runs = {
        'Baseline (DeltaES)': 'ZS_SSL_Model_100Epochs_Rate4_10Unrolls_Original_No_UFLoss_DeltaES_Tight/training_log.txt',
        'Baseline + UFLoss 0.5 (DeltaES)': 'ZS_SSL_Model_100Epochs_Rate4_10Unrolls_UFLoss_0.5_DeltaES_Tight/training_log.txt',
        'Curriculum Only (DeltaES)': 'ZS_SSL_Model_100Epochs_Rate4_10Unrolls_Original_No_UFLoss_Multimask_frequency_curriculum_K25_DeltaES_Tight/training_log.txt',
        'Curriculum + UFLoss 0.5 (DeltaES)': 'ZS_SSL_Model_100Epochs_Rate4_10Unrolls_UFLoss_0.5_Multimask_frequency_curriculum_K25_DeltaES_Tight/training_log.txt'
    }
    
    data = {}
    model_dirs = {}
    for name, path in runs.items():
        full_path = os.path.join(base_dir, path)
        data[name] = parse_log(full_path)
        model_dirs[name] = os.path.dirname(full_path)
        
    # Create figure with extra space at bottom for the table
    fig = plt.figure(figsize=(16, 16))
    
    # 2x2 grid for the plots, taking up top portion of the figure
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.6])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), 
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    
    metrics = [
        ('Validation Loss', 1),
        ('PSNR', 2),
        ('SSIM', 3),
        ('NMSE', 4)
    ]
    
    colors = ['#d62728', '#9467bd', '#2ca02c', '#1f77b4']  # Red, Purple, Green, Blue
    
    table_data = []
    row_colors = []
    
    for color, (name, d) in zip(colors, data.items()):
        epochs, val_losses, psnrs, ssims, nmses = d
        if not epochs:
            continue
            
        # Get Delta ES epoch
        delta_ep = get_delta_es_epoch(model_dirs[name])
        delta_idx = None
        if delta_ep is not None and delta_ep in epochs:
            delta_idx = epochs.index(delta_ep)

        # Find original early stopping epoch (min val loss)
        stop_idx = val_losses.index(min(val_losses))
        stop_epoch = epochs[stop_idx]
        
        # Best metrics
        best_psnr_idx = psnrs.index(max(psnrs))
        best_ssim_idx = ssims.index(max(ssims))
        best_nmse_idx = nmses.index(min(nmses))
        
        delta_str = "N/A"
        if delta_idx is not None:
            delta_str = f"Ep {delta_ep}\nPSNR: {psnrs[delta_idx]:.4f}\nSSIM: {ssims[delta_idx]:.4f}\nNMSE: {nmses[delta_idx]:.6f}"

        # Build table row with grouped multi-line text
        table_data.append([
            name,
            f"Ep {stop_epoch}\nPSNR: {psnrs[stop_idx]:.4f}\nSSIM: {ssims[stop_idx]:.4f}\nNMSE: {nmses[stop_idx]:.6f}",
            delta_str,
            f"Ep {epochs[best_psnr_idx]}\nPSNR: {psnrs[best_psnr_idx]:.4f}\nSSIM: {ssims[best_psnr_idx]:.4f}\nNMSE: {nmses[best_psnr_idx]:.6f}",
            f"Ep {epochs[best_ssim_idx]}\nPSNR: {psnrs[best_ssim_idx]:.4f}\nSSIM: {ssims[best_ssim_idx]:.4f}\nNMSE: {nmses[best_ssim_idx]:.6f}"
        ])
        row_colors.append(color)

        for ax, (metric_name, idx) in zip(axs, metrics):
            if metric_name == 'Validation Loss':
                y = val_losses
            elif metric_name == 'PSNR':
                y = psnrs
            elif metric_name == 'SSIM':
                y = ssims
            elif metric_name == 'NMSE':
                y = nmses
                
            ax.plot(epochs, y, label=name, color=color, linewidth=2)
            
            # 1. Circle for Original Early Stopping Point
            ax.plot(stop_epoch, y[stop_idx], marker='o', markersize=10, markeredgecolor='black', color='white', zorder=5)
            ax.plot(stop_epoch, y[stop_idx], marker='o', markersize=6, color=color, zorder=6)
            
            # 1.5. Star for Delta Early Stopping Point
            if delta_idx is not None:
                ax.plot(delta_ep, y[delta_idx], marker='*', markersize=16, markeredgecolor='black', color='gold', zorder=7)
            
            # 2. Square for True Metric Peak (only on metric plots, not val loss)
            if metric_name != 'Validation Loss':
                if metric_name == 'PSNR': best_idx = best_psnr_idx
                elif metric_name == 'SSIM': best_idx = best_ssim_idx
                elif metric_name == 'NMSE': best_idx = best_nmse_idx
                
                best_epoch = epochs[best_idx]
                best_val = y[best_idx]
                
                # Plot square only if it's different from the stopping point
                if best_epoch != stop_epoch and best_epoch != delta_ep:
                    ax.plot(best_epoch, best_val, marker='s', markersize=11, markeredgecolor='black', color=color, zorder=4)
                else:
                    ax.plot(best_epoch, best_val, marker='s', markersize=14, markeredgecolor=color, fillstyle='none', markeredgewidth=2, zorder=3)
            
            ax.set_title(f'{metric_name} vs Epochs', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epochs', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.grid(True, alpha=0.3)
            if metric_name == 'Validation Loss':
                # Custom legend to explain markers
                from matplotlib.lines import Line2D
                handles, labels = ax.get_legend_handles_labels()
                marker_circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=10, label='Original ES (Min Val)')
                marker_star = Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markeredgecolor='black', markersize=14, label='Delta ES Point')
                marker_square = Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=10, label='True Metric Peak')
                handles.extend([marker_circle, marker_star, marker_square])
                ax.legend(handles=handles, fontsize=11)
                
    # Add Table
    if table_data:
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        
        col_labels = ['Model', 'Original ES (Min Val)', 'Delta ES (Sig. Drop)', 'Best Overall (Peak PSNR & NMSE)', 'Absolute Peak SSIM']
        table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(13)
        table.scale(1, 5)  # Make rows extremely tall so 4 lines of text fit comfortably
        
        # Color the first column
        for i, color in enumerate(row_colors):
            table[(i+1, 0)].set_text_props(color=color, weight='bold')
            
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#f0f0f0')
                
    plt.suptitle('Early Stopping Disconnect: True Metrics vs. Validation Loss', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(base_dir, 'early_stopping_metrics_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == '__main__':
    main()
