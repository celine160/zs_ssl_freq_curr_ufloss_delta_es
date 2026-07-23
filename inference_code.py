import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from models import utils, parser_ops, UnrollNet
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = parser_ops.get_parser()
args = parser.parse_args()

file_dir = os.path.join(os.getcwd(),'saved_models_201_6002867')
saved_model_dir = os.path.join(file_dir,args.saved_model_name)
trn_loss=sio.loadmat(os.path.join(saved_model_dir,'TrainingLog.mat'))['trn_loss']
val_loss=sio.loadmat(os.path.join(saved_model_dir,'TrainingLog.mat'))['val_loss']
plt.plot(np.asarray(trn_loss).T)
plt.plot(np.asarray(val_loss).T)
plt.title('Loss Curves'), plt.xlabel('Epochs'), plt.ylabel('Loss')
plt.legend(['trn loss', 'val loss'])
plt.grid()
plt.savefig(os.path.join(saved_model_dir, 'inference_loss_curves.png'))
plt.close()

# load the data, pad the mask and normalize k-space
data = sio.loadmat(args.data_dir) 
kspace_test,sens_maps, original_mask= data['kspace'], data['sens_maps'], data['mask']
nrow_GLOB, ncol_GLOB, ncoil_GLOB  = kspace_test.shape

# %%  zeropadded outer edges of k-space with no signal- check readme file for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency
test_mask = np.complex64(original_mask)
if args.data_opt=='Coronal_PD':
    test_mask[ :, 0:17] = np.ones((nrow_GLOB, 17))
    test_mask[:, 352:ncol_GLOB] = np.ones((nrow_GLOB, 16))
    
# Normalize the kspace using the 95th percentile of the zero-filled image (fastMRI standard)
zf_img = utils.sense1(kspace_test * np.tile(test_mask[..., np.newaxis], (1, 1, ncoil_GLOB)), sens_maps)
scale = np.percentile(np.abs(zf_img), 95)
kspace_test = kspace_test / scale

#generate network input and reference image
nw_input = utils.sense1(kspace_test * np.tile(test_mask[..., np.newaxis], (1, 1, ncoil_GLOB)),sens_maps)
ref_image = utils.sense1(kspace_test,sens_maps)

model =UnrollNet.UnrolledNet(args,device=device).to(device)
model.load_state_dict(torch.load(os.path.join(saved_model_dir,'best.pth'))["model_state"])

model.eval()
with torch.no_grad():
    sens_maps  = torch.from_numpy(np.transpose(sens_maps[np.newaxis], (0, 3, 1, 2))).to(device)
    input_to_nw = torch.from_numpy(utils.complex2real(nw_input)[np.newaxis]).permute(0,3,1,2).to(device)
    trn_mask = torch.from_numpy(test_mask[np.newaxis]).to(device)
    nw_img_output, lamdas,nw_kspace_output = model(input_to_nw,trn_mask,trn_mask,sens_maps)

zs_ssl_recon = utils.real2complex(nw_img_output.permute(0,2,3,1).squeeze().to('cpu').numpy())

if args.data_opt == 'Coronal_PD':
    """window levelling in presence of fully-sampled data"""
    factor = np.max(np.abs(ref_image[:]))
else:
    factor = 1

ref_image = np.abs(ref_image) / factor
nw_input = np.abs(nw_input) / factor
zs_ssl_recon = np.abs(zs_ssl_recon) / factor

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

data_range = np.max(ref_image) - np.min(ref_image)
psnr_val = psnr(ref_image, zs_ssl_recon, data_range=data_range)
ssim_val = ssim(ref_image, zs_ssl_recon, data_range=data_range)
mse_val = np.mean((ref_image - zs_ssl_recon)**2)
nmse_val = np.linalg.norm(ref_image - zs_ssl_recon)**2 / np.linalg.norm(ref_image)**2

print(f"\n======================================")
print(f"Inference Results:")
print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM: {ssim_val:.4f}")
print(f"MSE:  {mse_val:.6f}")
print(f"NMSE: {nmse_val:.4f}")
print(f"======================================\n")

diff_image = np.abs(ref_image - zs_ssl_recon)

plt.figure(figsize=(20,5))
plt.subplot(1,4,1),plt.imshow(ref_image,cmap='gray',vmax=0.6*np.max(ref_image[:])), plt.title('Ref Image')
plt.subplot(1,4,2),plt.imshow(nw_input,cmap='gray',vmax=0.6*np.max(ref_image[:])), plt.title('Zero-Filled')
plt.subplot(1,4,3),plt.imshow(zs_ssl_recon,cmap='gray',vmax=0.6*np.max(ref_image[:])), plt.title(f'ZS-SSL Recon\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}\nMSE: {mse_val:.6f}, NMSE: {nmse_val:.4f}')
plt.subplot(1,4,4)
im = plt.imshow(diff_image,cmap='jet',vmax=0.1*np.max(ref_image[:]))
plt.title('Error Map')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.savefig(os.path.join(saved_model_dir, 'inference_output.png'))
plt.close()
