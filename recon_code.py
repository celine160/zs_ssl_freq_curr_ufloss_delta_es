import os

import time

import numpy as np

import scipy.io as sio

import torch

from torch.utils.data import DataLoader 

from models import utils, parser_ops, UnrollNet

from models.modules import MixL1L2Loss, Dataset, Dataset_Inference, train, validation, test

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'UFLoss', 'DL_Recon_UFLoss', 'models', 'unrolled2D'))
from networks.clean_ufloss.model import Model
import networks.resnet as resnet

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

parser = parser_ops.get_parser()
args = parser.parse_args()

import random

seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = sio.loadmat(args.data_dir) 

kspace_train,sens_maps, original_mask= data['kspace'], data['sens_maps'], data['mask']

args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB  = kspace_train.shape



# Normalize the kspace to 0-1 region (Original ZS-SSL normalization)
kspace_train = kspace_train / np.max(np.abs(kspace_train[:]))

# Compute the scaling factor needed to push the normalized image back up to the 95th percentile for UFLoss
zf_img = utils.sense1(kspace_train * np.tile(original_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB)), sens_maps)
ufloss_scale = np.percentile(np.abs(zf_img), 95)
#..................Generate validation mask....................................

if args.use_multi_masks:
    from models import multimask
    
    # Protect the ACS block from being masked out!
    center_kx = int(utils.find_center_ind(kspace_train, axes=(1, 2)).item())
    center_ky = int(utils.find_center_ind(kspace_train, axes=(0, 2)).item())
    small_acs_block = (4, 4)
    acs_mask = np.zeros_like(original_mask)
    acs_mask[center_kx - small_acs_block[0] // 2: center_kx + small_acs_block[0] // 2,
             center_ky - small_acs_block[1] // 2: center_ky + small_acs_block[1] // 2] = 1
    acs_mask = acs_mask * original_mask  # Only keep ACS where it was originally sampled

    safe_omega = np.copy(original_mask)
    safe_omega[acs_mask == 1] = 0

    bank = multimask.build_mask_triplet(
        omega_mask=safe_omega,
        rho_train=args.rho_train,
        rho_val=args.rho_val,
        mode=args.multi_mask_mode,
        seed=args.seed,
        K=args.multi_mask_k
    )
    cv_val_mask = bank["gamma_mask"]
    cv_trn_mask = bank["omega_mask"] - bank["gamma_mask"] + acs_mask
    remainder_mask = np.copy(cv_trn_mask)
else:
    cv_trn_mask, cv_val_mask = utils.uniform_selection(kspace_train,original_mask, rho=args.rho_val)
    remainder_mask, cv_val_mask=np.copy(cv_trn_mask),np.copy(np.complex64(cv_val_mask))




#..............................validation data..................................

ref_kspace_val = np.empty((args.num_reps,args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)

nw_input_val = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)



nw_input_val = utils.sense1(kspace_train * np.tile(cv_trn_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB)),sens_maps)[np.newaxis]

ref_kspace_val=kspace_train*np.tile(cv_val_mask[:, :, np.newaxis], (1, 1, args.ncoil_GLOB))[np.newaxis]



print('size of kspace: ', kspace_train[np.newaxis,...].shape, ', maps: ', sens_maps.shape, ', mask: ', original_mask.shape)



#..............................train data.....................................

nw_input_trn = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

ref_kspace = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)



trn_mask, loss_mask = np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                      np.empty((args.num_reps, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)


for jj in range(args.num_reps):

    if args.use_multi_masks:
        from models import multimask
        trn_mask[jj, ...], loss_mask[jj, ...], _ = multimask.select_round_robin(bank, jj, args.multi_mask_k)
        trn_mask[jj, ...] += acs_mask  # Add ACS block back!
    else:
        trn_mask[jj, ...], loss_mask[jj, ...] = utils.uniform_selection(kspace_train,remainder_mask, rho=args.rho_train)


    sub_kspace = kspace_train * np.tile(trn_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))

    ref_kspace[jj, ...] = kspace_train * np.tile(loss_mask[jj][..., np.newaxis], (1, 1, args.ncoil_GLOB))

    nw_input_trn[jj, ...] = utils.sense1(sub_kspace,sens_maps)

# %%  zeropadded outer edges of k-space with no signal- check readme file for further explanations

# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal

# in the training mask we set corresponding columns as 1 to ensure data consistency

if args.data_opt=='Coronal_PD' :

    trn_mask[:, :, 0:17] = np.ones((args.num_reps, args.nrow_GLOB, 17))

    trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((args.num_reps, args.nrow_GLOB, 16))

# %% Prepare the data for the training

# Compute pseudo target for UFLoss using Iterative CG-SENSE
import sigpy.mri as sigmri
if args.lambda_uf > 0.0:
    print("Generating clean pseudo-target for UFLoss using CG-SENSE...")
    kspace_under = kspace_train * np.tile(remainder_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB))
    k_sig = np.transpose(kspace_under, (2, 0, 1)).astype(np.complex64)
    s_sig = np.transpose(sens_maps, (2, 0, 1)).astype(np.complex64)
    pseudo_target_img = sigmri.app.SenseRecon(k_sig, s_sig, lamda=0.005, max_iter=30, show_pbar=False).run()
    print("CG-SENSE pseudo-target generated!")
    pseudo_target_img = utils.complex2real(pseudo_target_img[np.newaxis])
    pseudo_target_trn = np.tile(pseudo_target_img, (args.num_reps, 1, 1, 1))
else:
    pseudo_target_trn = None

sens_maps = np.tile(sens_maps[np.newaxis],(args.num_reps,1,1,1))

sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))

ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))

nw_input_trn = utils.complex2real(nw_input_trn)



# %% validation data 

ref_kspace_val = utils.complex2real(np.transpose(ref_kspace_val, (0, 3, 1, 2)))

nw_input_val = utils.complex2real(nw_input_val)
train_data= Dataset(nw_input_trn,trn_mask, loss_mask, sens_maps, ref_kspace, pseudo_target=pseudo_target_trn)

do_shuffle = not (args.use_multi_masks and args.multi_mask_mode == 'frequency_curriculum')
train_loader = DataLoader(train_data, batch_size=args.batchSize, shuffle=do_shuffle,num_workers = 6)



val_data = Dataset(nw_input_val,cv_trn_mask[np.newaxis], cv_val_mask[np.newaxis],  sens_maps[0][np.newaxis], ref_kspace_val)

val_loader = DataLoader(val_data, batch_size=args.batchSize, shuffle=False,num_workers = 6)

model_name = 'ZS_SSL_Model_'+str(args.epochs)+'Epochs_Rate'+ str(args.acc_rate) + '_' + str(args.nb_unroll_blocks) + 'Unrolls'
if args.lambda_uf > 0.0:
    model_name += f'_UFLoss_{args.lambda_uf}'
else:
    model_name += '_Original_No_UFLoss'
if args.use_multi_masks:
    model_name += f'_Multimask_{args.multi_mask_mode}_K{args.multi_mask_k}'
if args.use_delta_es:
    model_name += '_DeltaES'
    
# Dynamically create the base directory based on the input filename
data_file_name = os.path.basename(args.data_dir).split('.')[0]
if args.out_dir:
    base_out_dir = os.path.join(args.out_dir, f"saved_models_{data_file_name}")
else:
    base_out_dir = f"saved_models_{data_file_name}"

directory = os.path.join(base_out_dir, model_name + '_Tight')

if not os.path.exists(directory):
    os.makedirs(directory)
model =UnrollNet.UnrolledNet(args,device=device).to(device)

ufloss_path = os.path.join(os.path.dirname(__file__), 'UFLoss', 'Training_Logs', 'checkpoints_ufloss_mapping', 'train_UFLoss_feature_128_date_20260714_temp_0.07_lr_1e-5', 'checkpoints', 'ckpt200.pth')
model_re = Model(resnet.resnet18, feature_dim=128, data_length=18800)
model_re.load_state_dict(torch.load(ufloss_path, map_location="cpu")["state_dict"])
model_ufloss = model_re.network.to(device)
model_ufloss.requires_grad_ = False
model_ufloss.eval()

loss_fn = MixL1L2Loss()

optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate)


total_train_loss, total_val_loss = [], []
ep, val_loss_tracker_orig, val_loss_tracker_delta = 0, 0, 0
valid_loss_min_orig = np.inf
valid_loss_min_significant = np.inf
delta_es_locked = False

#train the model

start_time=time.time()

print("Preparing live-evaluation data...")
eval_test_mask = np.complex64(original_mask)
eval_nw_input = utils.sense1(kspace_train * np.tile(eval_test_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB)),np.transpose(sens_maps[0],(1,2,0)))
eval_ref_image = utils.sense1(kspace_train,np.transpose(sens_maps[0],(1,2,0)))
if args.data_opt=='Coronal_PD':
    eval_test_mask[:, 0:17] = np.ones((args.nrow_GLOB, 17))
    eval_test_mask[:, 352:args.ncol_GLOB] = np.ones((args.nrow_GLOB, 16))
    eval_factor = np.max(np.abs(eval_ref_image[:]))
else:
    eval_factor = 1.0

eval_test_data = Dataset_Inference(utils.complex2real(eval_nw_input[np.newaxis]),eval_test_mask[np.newaxis], eval_test_mask[np.newaxis], sens_maps[0][np.newaxis])
eval_test_loader = DataLoader(eval_test_data, batch_size=args.batchSize, shuffle=False, num_workers=6)
eval_ref_image_scaled = np.abs(eval_ref_image) / eval_factor

print("Starting training loop. This may take a moment if PyTorch is falling back to CPU...")
while ep<args.epochs and val_loss_tracker_orig<args.stop_training:



    tic = time.time()

    trn_loss, kspace_loss, ufloss, lamdas = train(train_loader, model, loss_fn, optimizer,device=device, model_ufloss=model_ufloss, alpha=args.lambda_uf, image_scale=ufloss_scale)

    val_loss = validation(val_loader, model, loss_fn, device=device)

    total_train_loss.append(trn_loss)    

    total_val_loss.append(val_loss)

    

    #save the best checkpoint

    checkpoint = {

            "epoch": ep,

            "valid_loss_min":val_loss,

            "model_state": model.state_dict(),

            "optim_state": optimizer.state_dict()

        }

    # --- ORIGINAL ES LOGIC ---
    if val_loss <= valid_loss_min_orig:
        valid_loss_min_orig = val_loss
        torch.save(checkpoint, os.path.join(directory,"best.pth")) 
        val_loss_tracker_orig = 0
    else:
        val_loss_tracker_orig += 1

    # --- DELTA ES LOGIC (Saves the model at the exact moment of significant improvement) ---
    if not delta_es_locked:
        # Calculate absolute improvement: val_loss must be lower than the previous best by at least args.min_delta
        is_significant_improvement = val_loss < (valid_loss_min_significant - args.min_delta)
        
        if ep >= args.warmup_epochs:
            if is_significant_improvement:
                valid_loss_min_significant = val_loss
                val_loss_tracker_delta = 0
                # Save the delta model exactly at this significant peak!
                torch.save(checkpoint, os.path.join(directory,"best_delta.pth"))
            else:
                val_loss_tracker_delta += 1
                
        if val_loss_tracker_delta >= args.stop_training:
            print(f"--> Delta ES Condition Met at Epoch {ep+1}! Locking in best_delta.pth from Epoch {ep+1 - args.stop_training}")
            delta_es_locked = True



    toc = time.time() - tic

    # Calculate live PSNR/SSIM/NMSE
    model.eval()
    with torch.no_grad():
        zs_ssl_recon = test(eval_test_loader, model, device)
        zs_ssl_recon = utils.real2complex(zs_ssl_recon.to('cpu').numpy())
        zs_ssl_recon_scaled = np.abs(zs_ssl_recon) / eval_factor
        live_psnr = peak_signal_noise_ratio(eval_ref_image_scaled, zs_ssl_recon_scaled, data_range=eval_ref_image_scaled.max())
        live_ssim = structural_similarity(eval_ref_image_scaled, zs_ssl_recon_scaled, data_range=eval_ref_image_scaled.max())
        live_nmse = np.linalg.norm(eval_ref_image_scaled - zs_ssl_recon_scaled)**2 / np.linalg.norm(eval_ref_image_scaled)**2

    log_str = f"Epoch: {ep+1} | Time: {toc:.2f}s | Trn Total: {trn_loss:.4f} | Trn Kspace: {kspace_loss:.4f} | Trn UFLoss (Raw): {ufloss:.2e} | Trn UFLoss (W): {ufloss * args.lambda_uf:.2e} | Val Total: {val_loss:.5f} | PSNR: {live_psnr:.4f} | SSIM: {live_ssim:.4f} | NMSE: {live_nmse:.6f}"
    print(log_str)
    with open(os.path.join(directory, 'training_log.txt'), 'a') as f:
        f.write(log_str + '\n')

    sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'trn_loss': total_train_loss, 'val_loss': total_val_loss})

    ep += 1

    

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60

print('Training completed in  ', str(ep), ' epochs, ', elapsed_minutes, ' minutes')

with open(os.path.join(directory, 'training_summary.txt'), 'w') as f:
    f.write(f"Training completed in {ep} epochs, {elapsed_minutes:.2f} minutes\n")

    
test_mask = np.complex64(original_mask)

#generate network input  and reference image

nw_input_inference = utils.sense1(kspace_train * np.tile(test_mask[..., np.newaxis], (1, 1, args.ncoil_GLOB)),np.transpose(sens_maps[0],(1,2,0)))

ref_image= utils.sense1(kspace_train,np.transpose(sens_maps[0],(1,2,0)))

if args.data_opt=='Coronal_PD' :

    test_mask[ :, 0:17] = np.ones((args.nrow_GLOB, 17))

    test_mask[:, 352:args.ncol_GLOB] = np.ones((args.nrow_GLOB, 16))



test_data = Dataset_Inference(utils.complex2real(nw_input_inference[np.newaxis]),test_mask[np.newaxis], test_mask[np.newaxis],  sens_maps[0][np.newaxis])

test_loader = DataLoader(test_data, batch_size=args.batchSize, shuffle=False,num_workers = 6)
# directory = 'saved_models/ZS_SSL_Knee_Saved_Models_300Epochs_Rate4_10Unrolls'

# load the best checkpoint

best_checkpoint = torch.load(os.path.join(directory,'best.pth'))

model.load_state_dict(best_checkpoint["model_state"])

zs_ssl_recon = test(test_loader, model, device)

zs_ssl_recon = utils.real2complex(zs_ssl_recon.to('cpu').numpy())
if args.data_opt == 'Coronal_PD':

    """window levelling in presence of fully-sampled data"""

    factor = np.max(np.abs(ref_image[:]))

else:

    factor = 1



ref_image = np.abs(ref_image) / factor

nw_input_inference = np.abs(nw_input_inference) / factor

zs_ssl_recon = np.abs(zs_ssl_recon) / factor



plt.figure(figsize=(24,8))

psnr_val = peak_signal_noise_ratio(ref_image, zs_ssl_recon, data_range=ref_image.max())
ssim_val = structural_similarity(ref_image, zs_ssl_recon, data_range=ref_image.max())
error_map = np.abs(ref_image - zs_ssl_recon)
mse_val = np.mean((ref_image - zs_ssl_recon)**2)
nmse_val = np.linalg.norm(ref_image - zs_ssl_recon)**2 / np.linalg.norm(ref_image)**2

plt.subplot(1,4,1),plt.imshow(ref_image,cmap='gray',vmax=0.6*np.max(ref_image[:])), plt.title('Ref Image')
plt.subplot(1,4,2),plt.imshow(nw_input_inference,cmap='gray',vmax=0.6*np.max(ref_image[:])), plt.title('Zero-Filled')
plt.subplot(1,4,3),plt.imshow(zs_ssl_recon,cmap='gray',vmax=0.6*np.max(ref_image[:])), plt.title(f'ZS-SSL Recon\nPSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}\nMSE: {mse_val:.6f}, NMSE: {nmse_val:.4f}')
plt.subplot(1,4,4),plt.imshow(error_map,cmap='hot',vmax=0.2*np.max(ref_image[:])), plt.title('Error Map'), plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(directory, 'recon_output.png'))

plt.figure()

plt.plot(np.asarray(total_train_loss).T)

plt.plot(np.asarray(total_val_loss).T)

plt.title('Loss Curves'), plt.xlabel('Epochs'), plt.ylabel('Loss')

plt.legend(['trn loss', 'val loss'])

plt.grid()

plt.savefig(os.path.join(directory, 'loss_curves.png'))
