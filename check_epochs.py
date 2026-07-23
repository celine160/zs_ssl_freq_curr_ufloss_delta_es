import scipy.io as sio

try:
    log = sio.loadmat('saved_models/ZS_SSL_Model_300Epochs_Rate4_10Unrolls_Multimask_frequencies_K25/TrainingLog.mat')
    val_loss = log['val_loss'][0]
    trn_loss = log['trn_loss'][0]
    
    print(f"Total epochs trained: {len(val_loss)}")
    
    # The 'best' model is saved when val_loss is at its minimum
    import numpy as np
    best_epoch = np.argmin(val_loss) + 1  # +1 because epochs are 1-indexed
    print(f"Best model was saved at epoch: {best_epoch}")
    
except Exception as e:
    print(f"Error: {e}")
