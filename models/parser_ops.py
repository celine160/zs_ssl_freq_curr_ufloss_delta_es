import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='ZS-SSL: Zero-Shot Self-Supervised Learning')

    # %% hyperparameters for the  network
    parser.add_argument('--data_opt', type=str, default='AXFLAIR',
                    help='type of dataset')
    parser.add_argument('--data_dir', type=str, default='/home/zs_ssl/data.mat',
                    help='data directory')                
    parser.add_argument('--nrow_GLOB', type=int, default=320,
                        help='number of rows of the slices in the dataset')
    parser.add_argument('--ncol_GLOB', type=int, default=368,
                        help='number of columns of the slices in the dataset')
    parser.add_argument('--ncoil_GLOB', type=int, default=15,
                        help='number of coils of the slices in the dataset')                
    parser.add_argument('--acc_rate', type=int, default=4,
                        help='acceleration rate')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='batch size')
    parser.add_argument('--nb_unroll_blocks', type=int, default=10,
                        help='number of unrolled blocks')
    parser.add_argument('--nb_res_blocks', type=int, default=15,
                        help="number of residual blocks in ResNet")
    parser.add_argument('--CG_Iter', type=int, default=10,
                        help='number of Conjugate Gradient iterations for DC')

    # %% hyperparameters for the zs-ssl
    parser.add_argument('--rho_val', type=float, default=0.2,
                        help='cardinality of the validation mask')                        
    parser.add_argument('--rho_train', type=float, default=0.4,
                        help='cardinality of the loss mask, \ rho = |\ Lambda| / |\ Omega|')
    parser.add_argument('--num_reps', type=int, default=25,
                        help='number of repetions for the remainder mask')
    parser.add_argument('--transfer_learning', type=bool, default=False,
                        help='transfer learning from pretrained model')
    parser.add_argument('--TL_path', type=str, default=None,
                        help='path to pretrained model')                                        
    parser.add_argument('--stop_training', type=int, default=15, help='Patience for Early Stopping')
    parser.add_argument('--use_delta_es', action='store_true', help='Use delta-based early stopping with warmup instead of original simple early stopping')
    parser.add_argument('--min_delta', type=float, default=0.0005, help='Minimum absolute drop (e.g. 0.0005) in validation loss to reset early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=15, help='Number of epochs before delta early stopping can trigger')
    parser.add_argument('--out_dir', type=str, default='', help='Custom base output directory for saving models')
    parser.add_argument('--saved_model_name', type=str, default='ZS_SSL_Model_300Epochs_Rate4_10Unrolls',
                        help='model name to be used for eval')       
    
    # %% hyperparameters for UFLoss
    parser.add_argument('--lambda_uf', type=float, default=0.0,
                        help='weight of the UFLoss perceptual regularization')
    
    # %% hyperparameters for multi-mask experiments
    parser.add_argument("--use_multi_masks", action="store_true", help="Enable multi-mask mode")
    parser.add_argument("--multi_mask_mode", type=str, default="coverage", choices=["random", "coverage", "frequencies", "frequency_balanced", "frequency_curriculum"], help="Type of multi-mask generation")
    parser.add_argument("--seed", type=int, default=42, help="Universal seed for random initialization and masks")
    parser.add_argument("--multi_mask_k", type=int, default=3, help="Number of masks to generate in the multi-mask set (default: 3)")
    
    return parser
